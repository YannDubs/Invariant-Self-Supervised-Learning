"""Self distillation proxies to minimize R[M(X)|Z] and thus ensure decodability."""
from __future__ import annotations

import copy
import queue
from collections import Sequence
from typing import Any, Optional
import abc

import torch
import torch.nn as nn
from issl.architectures import  get_Architecture
from issl.architectures.basic import FlattenCosine
from issl.helpers import (RunningMean, average_dict, freeze_module_, kl_divergence, prod, queue_push_, sinkhorn_knopp,
                          weights_init, BatchNorm1d)
from torch.distributions import Categorical
from torch.nn import functional as F
import  numpy as np

__all__ = [
    "DISSL",
    "DINO"
]

class BaseDistillationISSL(nn.Module, metaclass=abc.ABCMeta):


    def __init__(
            self,
            z_shape: Sequence[int],
            n_equivalence_classes: float = 16384,
            projector_kwargs: dict[str, Any] = {
                "architecture": "mlp",
                "bottleneck_size": 512,
                "hid_dim": 1024,
                "n_hid_layers": 1,
                "is_cosine": True,
                "norm_layer": "batch"
            },
    ) -> None:
        super().__init__()
        self.z_shape = [z_shape] if isinstance(z_shape, int) else z_shape
        self.z_dim = prod(self.z_shape)
        self.n_equivalence_classes = n_equivalence_classes

        self.projector_kwargs = self.process_shapes(projector_kwargs)
        Projector = get_Architecture(**self.projector_kwargs)
        self.projector = Projector()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        weights_init(self)

    def process_shapes(self, kwargs: dict) -> dict:
        kwargs = copy.deepcopy(kwargs)  # ensure mutable object is ok
        kwargs["in_shape"] = kwargs.get("in_shape", self.z_shape)
        kwargs["out_shape"] = kwargs.get("out_shape", self.n_equivalence_classes)
        return kwargs

    def forward(
            self, z: torch.Tensor, z_tgt: torch.Tensor, _, __, parent: Any
    ) -> tuple[torch.Tensor, dict, dict]:
        """Self distillation of examples and compute the upper bound on R[A|Z].

        Parameters
        ----------
        z : Tensor shape=[batch_size, z_dim]
            Sampled representation.

        z_tgt : Tensor shape=[2 * batch_size, *x_shape]
            Representation from the other branch.

        parent : ISSLModule, optional
            Parent module. Should have attribute `parent.encoder`.

        Returns
        -------
        hat_R_MlA : torch.Tensor shape=[batch_shape]
            Estimate of R[M(X)|Z].

        logs : dict
            Additional values to log.

        other : dict
            Additional values to return.
        """
        loss, logs, other = self.loss(z, z_tgt)
        return loss, logs, other

    @abc.abstractmethod
    def loss(self, z_src: torch.Tensor, z_tgt: torch.Tensor) -> tuple[torch.Tensor, dict, dict]:
        pass



# performs SSL by having one part of the model implementing the M(X)
add_doc_dissl = """              
    lambda_maximality : float, optional
        Parameter that weights the divergence D[p_hat(M) || Unif]
    """

class DISSL(BaseDistillationISSL):
    """Compute the ISSL loss using self distillation (i.e. approximates M(X) using current representation).

    Parameters
    ----------
    z_shape : sequence of int
        Shape of the representation.

    n_equivalence_classes : int, optional
        Number of equivalence classes (C in the paper)

    projector_kwargs : dict, optional
        Arguments to get `Projector` from `get_Architecture`.

    encoder : Module, optional
        Optional encoder.
    """

    def __init__(
        self,
        *args,
        lambda_maximality: float = 2.3,
        beta_det_inv: float=0.8,
        temperature: float=1.0,
        temperature_assign:  Optional[float] = None,
        predictor_kwargs: dict[str, Any] = {
                               "architecture": "linear",
                           },
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)
        self.lambda_maximality = lambda_maximality
        self.beta_det_inv = beta_det_inv
        self.temperature = temperature
        self.temperature_assign = temperature_assign or self.temperature / 2

        # code ready for multi crops
        self.crops_assign = 2
        self.crops_pred = 2

        self.predictor_kwargs = self.process_shapes(predictor_kwargs)
        Predictor = get_Architecture(**self.predictor_kwargs)
        self.predictor = Predictor()

        self.reset_parameters()

    def get_Mx(self, z_tgt):
        """Return the parameters of the categorical dist of predicted M(X). If entropy is 0 then this is Mx."""
        logits_assign = self.projector(z_tgt).float() / self.temperature_assign
        Mx = F.softmax(logits_assign, dim=-1)
        return Mx

    def loss(
        self, z: torch.Tensor, z_tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, dict, dict]:

        # shape: [batch_size, M_shape]. Make sure not half prec
        logits_assign = self.projector(z_tgt).float() / self.temperature_assign
        logits_predict = self.predictor(z).float() / self.temperature

        all_p_Mlz = F.softmax(logits_assign, dim=-1).chunk(self.crops_assign)
        all_log_p_Mlz = F.log_softmax(logits_assign, dim=-1).chunk(self.crops_assign)
        all_log_q_Mlz = F.log_softmax(logits_predict, dim=-1).chunk(self.crops_pred)

        CE_pMlz_qMlza = 0
        H_M = 0
        CE_pMlz_pMlza = 0
        n_CE_pq = 0
        n_CE_pp = 0
        for i_p, p_Mlz in enumerate(all_p_Mlz):

            ##### Ensure maximality #####
            # current marginal estimate p(M). batch shape: [] ; event shape: []
            p_M = p_Mlz.mean(0)

            # D[\hat{p}(M) || Unif(\calM)]. shape: []
            # for unif prior same as maximizing entropy.
            H_M = H_M + Categorical(probs=p_M).entropy()
            #############################

            ##### Ensure invariance and determinism of assignement #####
            for i_log_p, log_p_Mlza in enumerate(all_log_p_Mlz):
                if i_p == i_log_p:
                    continue
                CE_pMlz_pMlza = CE_pMlz_pMlza - (p_Mlz * log_p_Mlza).sum(-1)
                n_CE_pp += 1
            #########################

            for i_q, log_q_Mlza in enumerate(all_log_q_Mlz):
                if i_p == i_q:
                    continue  # skip if same view

                # KL = - H[M|Z] - E_{p(M|Z)}[log q(M|Z)]. As you want to have a deterministic
                # p(M|Z) you want to min H[M|Z]. So min KL + H[M|Z] = - E_{p(M|Z)}[log q(M|Z)]
                CE_pMlz_qMlza = CE_pMlz_qMlza - (p_Mlz * log_q_Mlza).sum(-1)
                n_CE_pq += 1

        CE_pMlz_qMlza /= n_CE_pq
        H_M /= len(all_p_Mlz)
        CE_pMlz_pMlza /= n_CE_pp

        # shape: [batch_size]
        loss = CE_pMlz_qMlza - self.lambda_maximality * H_M + self.beta_det_inv * CE_pMlz_pMlza

        logs = dict(
            fit_pM_Unif=-H_M,
            fit_pMlz_qMlz=CE_pMlz_qMlza.mean(),
            H_M=H_M,
            H_Mlz=Categorical(probs=torch.cat(all_p_Mlz, dim=0).detach()).entropy().mean()
        )
        other = dict()

        return loss, logs, other

    def asymmetric_loss(self, z_src : torch.Tensor, z_tgt: torch.Tensor) -> tuple[torch.Tensor, dict, dict]:
        pass  # not using




class DINO(BaseDistillationISSL):
    """Compute the ISSL loss using self distillation (i.e. approximates M(X) using current representation).

    Parameters
    ----------
    z_shape : sequence of int
        Shape of the representation.

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
            *args,
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
            **kwargs
    ) -> None:

        super().__init__(
            *args,
            n_equivalence_classes=n_equivalence_classes,
            projector_kwargs=projector_kwargs,
            **kwargs,
        )

        self.student_temperature = student_temperature
        self.center_momentum = center_momentum
        self.freeze_Mx_epochs = freeze_Mx_epochs  # will be frozen in ISSL
        self.register_buffer("center", torch.zeros(1, n_equivalence_classes))

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

    def loss(
        self, z: torch.Tensor, z_tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, dict, dict]:

        z_x, z_a = z.chunk(2, dim=0)
        z_tgt_x, z_tgt_a = z_tgt.chunk(2, dim=0)

        loss1, logs1, other = self.asymmetric_loss(z_x, z_tgt_a)
        loss2, logs2, _ = self.asymmetric_loss(z_a, z_tgt_x)
        loss = (loss1 + loss2) / 2

        logs = average_dict(logs1, logs2)

        return loss, logs, other

    def asymmetric_loss(self, M_student, M_teacher):

        logits_student = M_student / self.student_temperature
        logits_teacher = (M_teacher - self.center) / self.teacher_temperature

        # p(M|Z). shape: [batch_size, M_shape]
        p_Mlz = F.softmax(logits_teacher, dim=-1)

        # shape: [batch_size]
        fit_pMlz_qMlz = -(p_Mlz * logits_student.log_softmax(-1)).sum(-1)

        self.update_center(M_teacher)

        logs = dict(
            fit_pMlz_qMlz=fit_pMlz_qMlz.mean(),
            H_Mlz=Categorical(probs=p_Mlz).entropy().mean(),
        )
        other = dict()

        return fit_pMlz_qMlz, logs, other


    def forward(
            self, z: torch.Tensor, z_tgt: torch.Tensor, x : torch.Tensor, a : torch.Tensor , parent: Any
    ) -> tuple[torch.Tensor, dict, dict]:

        self.current_epoch = parent.current_epoch

        # shape: [batch_size, M_shape].
        # have to use x and a directly because the encoder is different now
        student_M = self.projector(z).float()
        teacher_M = self.teacher_proj(parent(torch.cat([x, a]),
                                             encoder=self.teacher_encoder)).float()

        loss, logs, other = super().loss(student_M, teacher_M)

        return loss, logs, other
