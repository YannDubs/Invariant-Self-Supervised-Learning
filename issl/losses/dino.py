"""Reimplementation of DINO."""
from __future__ import annotations

import copy
from typing import Any, Sequence, Union, Optional

import torch
import torch.nn as nn
from issl.architectures import  get_Architecture
from issl.helpers import (average_dict, freeze_module_, weights_init)
from torch.distributions import Categorical
from torch.nn import functional as F
import numpy as np
import math
import pytorch_lightning as pl

class DINO(nn.Module):
    """Compute the ISSL loss using self distillation (i.e. approximates M(X) using current representation).

    Parameters
    ----------
    z_dim : int
        Dimensionality of the representation.

    n_equivalence_classes : int, optional
        Number of equivalence classes (C in the paper)

    projector_kwargs : dict, optional
        Arguments to get `Projector` from `get_Architecture`.

    encoder : Module, optional
        Optional encoder.

    freeze_Mx_epochs : int, optional
        Freeze the last lasyer that many epochs from the start.

    student_temperature : float, optional
        Temperature for prediction of the student network.

    center_momentum : float, optional
        Ema weight for computing the mean for centering.
    """

    def __init__(
            self,
            z_dim: int,
            encoder: nn.Module,
            n_equivalence_classes: int=1000,  # for imagenet they use 65k
            projector_kwargs: dict[str, Any] = {
                "architecture": "mlp",
                "hid_dim": 1024,
                "n_hid_layers": 1,
                "norm_layer": "batch",
                "activation": "GELU",
                "bottleneck_size": 256,
                "is_cosine": True,
                "is_batchnorm_bottleneck": False
            },
            freeze_Mx_epochs: int=1,
            student_temperature: float=0.1,
            center_momentum: float = 0.9,
            warmup_teacher_temp: float = 0.04,
            warmup_teacher_temp_epochs : int = 50,
            teacher_temperature: float = 0.07,
            n_epochs: Optional[int] = None,
    ) -> None:

        super().__init__()

        self.z_dim = z_dim
        self.n_equivalence_classes = n_equivalence_classes
        self.student_temperature = student_temperature
        self.center_momentum = center_momentum
        self.freeze_Mx_epochs = freeze_Mx_epochs  # will be frozen in ISSL
        self.register_buffer("center", torch.zeros(1, n_equivalence_classes))

        self.projector_kwargs = self.process_shapes(projector_kwargs)
        Projector = get_Architecture(**self.projector_kwargs)
        self.projector = Projector()

        # clone student and freeze it
        # cannot use deepcopy because weight norm => reinstantiate
        TeacherProj = get_Architecture(**self.projector_kwargs)
        self.teacher_proj = TeacherProj()
        self.teacher_encoder = copy.deepcopy(encoder)
        freeze_module_(self.teacher_proj)
        freeze_module_(self.teacher_encoder)

        self.n_epochs = n_epochs
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temperature, warmup_teacher_temp_epochs),
            np.ones(self.n_epochs - warmup_teacher_temp_epochs) * teacher_temperature
        ))

        self.current_epoch = 0
        self.reset_parameters()

    def reset_parameters(self) -> None:
        weights_init(self)
        self.current_epoch = 0

    @property
    def to_freeze(self):
        # only works for MLP
        return [self.projector.post_block.linear]  # will be frozen in ISSL if self.freeze_Mx_epochs > 0

    @property
    def teacher_temperature(self):
        try:
            return self.teacher_temp_schedule[self.current_epoch]
        except IndexError:
            return self.teacher_temp_schedule[-1]

    @torch.no_grad()
    def update_center(self, teacher_output):
        # shape: [batch_size, M_shape]
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def process_shapes(self, kwargs: dict) -> dict:
        kwargs = copy.deepcopy(kwargs)  # ensure mutable object is ok
        kwargs["in_shape"] = kwargs.get("in_shape", self.z_dim)
        kwargs["out_shape"] = kwargs.get("out_shape", self.n_equivalence_classes)
        return kwargs

    def forward(
            self, z: torch.Tensor, x : torch.Tensor, x_tilde : torch.Tensor , parent: Any
    ) -> tuple[torch.Tensor, dict]:
        """Self distillation of examples and compute the upper bound on R[A|Z].

        Parameters
        ----------
        z : Tensor shape=[batch_size, z_dim]
            Sampled representation.

        parent : ISSLModule, optional
            Parent module. Should have attribute `parent.encoder`.

        Returns
        -------
        loss : torch.Tensor shape=[]

        logs : dict
            Additional values to monitor.
        """
        self.current_epoch = parent.current_epoch

        # shape=[batch_size*2, z_dim]
        # have to use x and x_tilde directly because the encoder is different now
        z_tilde = self.teacher_encoder(torch.cat([x, x_tilde], dim=0))

        # shape: [batch_size*2, M_shape].
        student_M = self.projector(z).float()
        teacher_M = self.teacher_proj(z_tilde).float()

        # shape: [].
        loss, logs = self.loss(student_M, teacher_M)

        return loss, logs

    def loss(
        self, z: torch.Tensor, z_tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:

        # shape: [batch_size, n_nequiv].
        z_x, z_a = z.chunk(2, dim=0)
        z_tgt_x, z_tgt_a = z_tgt.chunk(2, dim=0)

        loss1, logs1 = self.asymmetric_loss(z_x, z_tgt_a)
        loss2, logs2 = self.asymmetric_loss(z_a, z_tgt_x)
        loss = (loss1 + loss2) / 2

        logs = average_dict(logs1, logs2)

        return loss, logs

    def asymmetric_loss(self, M_student, M_teacher):

        logits_student = M_student / self.student_temperature
        logits_teacher = (M_teacher - self.center) / self.teacher_temperature

        # p(M|Z). shape: [batch_size, M_shape]
        p_Mlz = F.softmax(logits_teacher, dim=-1)

        # shape: []
        CE_distill_aug = -(p_Mlz * logits_student.log_softmax(-1)).sum(-1).mean()

        self.update_center(M_teacher)

        logs = dict(
            CE_distill_aug=CE_distill_aug,
            H_Mlz=Categorical(probs=p_Mlz).entropy().mean(),
        )
        return CE_distill_aug, logs

# modified from https://github.com/PyTorchLightning/lightning-bolts/blob/ad771c615284816ecadad11f3172459afdef28e3/pl_bolts/callbacks/byol_updates.py
class MAWeightUpdate(pl.Callback):
    """EMA Weight update rule for DINO.

    Notes
    -----
    - BYOL/DINO/... claims this keeps the online_network from collapsing.
    - Automatically increases tau from ``initial_tau`` to 1.0 with every training step
    """

    def __init__(self, initial_tau: float = 0.996):
        """
        Args:
            initial_tau: starting tau. Auto-updates with every training step
        """
        super().__init__()
        self.initial_tau = initial_tau
        self.current_tau = initial_tau

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: pl.LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
    ) -> None:

        # get networks
        student_enc = pl_module.encoder
        teacher_enc = pl_module.loss_decodability.teacher_encoder

        student_proj = pl_module.loss_decodability.projector
        teacher_proj = pl_module.loss_decodability.teacher_proj

        # update weights
        with torch.no_grad():  # not needed
            self.update_weights(student_enc, teacher_enc)
            self.update_weights(student_proj, teacher_proj)

        # update tau after
        self.current_tau = self.update_tau(pl_module, trainer)

    def update_tau(self, pl_module: pl.LightningModule, trainer: pl.Trainer) -> float:
        max_steps = len(trainer.train_dataloader) * trainer.max_epochs
        tau = (
            1
            - (1 - self.initial_tau)
            * (math.cos(math.pi * pl_module.global_step / max_steps) + 1)
            / 2
        )
        return tau

    def update_weights(
        self,
        online_net: Union[nn.Module, torch.Tensor],
        target_net: Union[nn.Module, torch.Tensor],
    ) -> None:
        # apply MA weight update
        for (name, online_p), (_, target_p) in zip(
            online_net.named_parameters(), target_net.named_parameters(),
        ):
            target_p.data = (
                self.current_tau * target_p.data
                + (1 - self.current_tau) * online_p.detach().data
            )
