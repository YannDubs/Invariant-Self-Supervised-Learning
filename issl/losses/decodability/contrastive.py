"""Contrastive proxies to minimize R[A|Z] and thus ensure decodability."""
from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from issl.architectures import get_Architecture
from issl.helpers import prod, warmup_cosine_scheduler, weights_init

__all__ = ["ContrastiveISSL"]


# TODO should probably write it as a subclass of self distillation because you are reusing a lot of the
class ContrastiveISSL(nn.Module):
    """Computes the ISSL loss using contrastive variational bound (i.e. with positive and negative examples).

    Notes
    -----
    - parts of code taken from https://github.com/lucidrains/contrastive-learner

    Parameters
    ----------
    z_shape : sequence of int
        Shape of the representation.

    temperature : float, optional
        Temperature scaling in InfoNCE. Recommended less than 1.

    is_train_temperature : bool, optional
        Whether to treat the temperature as a parameter. Uses the same same as CLIP.
        If so `temperature` will be used as the initialization.

    min_temperature : float, optional
        Lower bound on the temperature. Only if `is_train_temperature`.

    is_pred_proj_same : bool, optional
        Whether to use the projector for the predictor. Typically, True in SSL, but our theory says that it should be
        False.

    is_self_contrastive : bool, optional
        Whether to keep current example in batch when performing contrastive learning and then have to predict
        proba 0.5 to current and positive.

    is_batchnorm_post : bool, optional
        Whether to add a batchnorm layer after the predictor / projector.

    is_batchnorm_pre : bool, optional
        Whether to add a batchnorm layer pre the predictor / projector.

    projector_kwargs : dict, optional
        Arguments to get `Projector` from `get_Architecture`. Note that is `out_shape` is <= 1
        it will be a percentage of z_dim. To use no projector set `mode=Flatten`.

    predictor_kwargs : dict, optional
        Arguments to get `Predictor` from `get_Architecture`. Note that is `out_shape` is <= 1
        it will be a percentage of z_dim. To use no predictor set `mode=Flatten`.

    References
    ----------
    [1] Song, Jiaming, and Stefano Ermon. "Multi-label contrastive predictive coding." Advances in
    Neural Information Processing Systems 33 (2020).
    """

    def __init__(
        self,
        z_shape: Sequence[int],
        temperature: float = 0.07,
        is_train_temperature: bool = False,
        is_cosine_pos_temperature: bool = False,
        is_cosine_neg_temperature: bool = False,
        n_epochs: Optional[int] = None,
        min_temperature: float = 0.01,
        is_pred_proj_same: bool = False,
        is_self_contrastive: bool = False,
        is_batchnorm_pre: bool = False,
        is_batchnorm_post: bool = True,
        projector_kwargs: dict[str, Any] = {
            "architecture": "mlp",
            "hid_dim": 2048,
            "n_hid_layers": 2,
            "norm_layer": "batch",
        },
        predictor_kwargs: dict[str, Any] = {"architecture": "flatten"},
        is_margin_loss: bool = False,  # DEV
        encoder: Optional[nn.Module] = None  # only used for DINO
    ) -> None:
        super().__init__()
        self.z_shape = [z_shape] if isinstance(z_shape, int) else z_shape
        self.z_dim = prod(self.z_shape)
        self.is_train_temperature = is_train_temperature
        self.is_cosine_neg_temperature = is_cosine_neg_temperature
        self.is_cosine_pos_temperature = is_cosine_pos_temperature
        self.n_epochs = n_epochs
        self.min_temperature = min_temperature
        self.is_pred_proj_same = is_pred_proj_same
        self.is_batchnorm_pre = is_batchnorm_pre
        self.is_batchnorm_post = is_batchnorm_post
        self.predictor_kwargs = self.process_kwargs(predictor_kwargs)
        self.projector_kwargs = self.process_kwargs(projector_kwargs)
        self.is_self_contrastive = is_self_contrastive
        self.is_margin_loss = is_margin_loss

        if self.is_pred_proj_same:
            self.is_self_contrastive = False

        Projector = get_Architecture(**self.projector_kwargs)
        self.projector = self.add_batchnorms(Projector(), projector_kwargs["architecture"])

        if self.is_pred_proj_same:
            self.predictor = self.projector
        else:
            Predictor = get_Architecture(**self.predictor_kwargs)
            self.predictor = self.add_batchnorms(Predictor(), predictor_kwargs["architecture"])

        if self.is_train_temperature:
            self.init_temperature = temperature
            self.log_temperature = nn.Parameter(
                torch.log(torch.tensor(self.init_temperature))
            )
        elif self.is_cosine_neg_temperature:
            self.precomputed_temperature = [warmup_cosine_scheduler(i, (self.n_epochs // 50) + 1, self.n_epochs,
                                                                    boundary=1, optima=self.min_temperature)
                                            for i in range(1000)]
        elif self.is_cosine_pos_temperature:
            self.precomputed_temperature = [warmup_cosine_scheduler(i, (self.n_epochs // 50) + 1, self.n_epochs,
                                                                    boundary=self.min_temperature, optima=1)
                                            for i in range(1000)]
        else:
            self._temperature = temperature

        self.reset_parameters()

    def add_batchnorms(self, head, arch):
        if self.is_batchnorm_pre:
            head = nn.Sequential(nn.BatchNorm1d(self.z_dim), head)

        if self.is_batchnorm_post:
            if not (arch.lower() in ["identity", "flatten"]):
                head = nn.Sequential(head, nn.BatchNorm1d(self.out_dim))

        return head

    def reset_parameters(self) -> None:
        weights_init(self)

        if self.is_train_temperature:
            self.log_temperature = nn.Parameter(
                torch.log(torch.tensor(self.init_temperature))
            )

    def process_kwargs(self, kwargs: dict) -> dict:
        kwargs = copy.deepcopy(kwargs)  # ensure mutable object is ok
        kwargs["in_shape"] = self.z_shape

        if "out_shape" not in kwargs:
            kwargs["out_shape"] = prod(self.z_shape)
        elif kwargs["out_shape"] <= 1:
            kwargs["out_shape"] = max(10, int(self.z_dim * kwargs["out_shape"]))

        self.out_dim = kwargs["out_shape"]

        if self.is_batchnorm_post and kwargs["architecture"] in ["mlp", "linear"]:
            kwargs["bias"] = False  # no bias when batchorm

        return kwargs

    def forward(
        self, z: torch.Tensor, z_tgt: torch.Tensor, _, __, parent: Any
    ) -> tuple[torch.Tensor, dict, dict]:
        """Contrast examples and compute the upper bound on R[A|Z].
    
        Parameters
        ----------
        z : Tensor shape=[2 * batch_size, z_dim]
            Representations.

        z_tgt : Tensor shape=[2 * batch_size, *x_shape]
            Representation from the other branch.

        parent : ISSLModule, optional
            Parent module.
    
        Returns
        -------
        loss : torch.Tensor shape=[batch_shape]
            Estimate of the loss.
    
        logs : dict
            Additional values to log.
    
        other : dict
            Additional values to return.
        """

        if z.ndim != 2:
            raise ValueError(
                f"When using contrastive loss the representation needs to be flattened."
            )

        self.current_epoch = parent.current_epoch

        new_batch_size, z_dim = z.shape
        batch_size = new_batch_size // 2

        # shape: [2 * batch_size, 2 * batch_size]
        logits = self.compute_logits_p_Alz(z, z_tgt)

        # shape: [2 * batch_size]
        hat_H_mlz, logs = self.compute_loss(logits)

        # shape: [batch_size]
        hat_H_mlz = (hat_H_mlz[:batch_size] + hat_H_mlz[batch_size:]) / 2

        other = dict()

        return hat_H_mlz, logs, other

    def compute_logits_p_Alz(
        self, z_src: torch.Tensor, z_tgt: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the logits for the contrastive predictor p(A|Z)."""

        # shape: [2*batch_size, out_shape]
        z_src = self.predictor(z_src)
        z_tgt = self.projector(z_tgt)

        # will use cosine similarity
        z_src = F.normalize(z_src, dim=1, p=2)
        z_tgt = F.normalize(z_tgt, dim=1, p=2)

        # shape: [2*batch_size, 2*batch_size]
        logits = z_src @ z_tgt.T

        return logits

    @property
    def temperature(self):
        if self.is_train_temperature:
            temperature = torch.clamp(
                self.log_temperature.exp(), min=self.min_temperature
            )

        elif self.is_cosine_neg_temperature or self.is_cosine_pos_temperature:
            temperature = self.precomputed_temperature[self.current_epoch]

        else:
            temperature = self._temperature
        return temperature

    def compute_loss(self, logits: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Computes the upper bound H_q[A|Z]."""
        n_classes = logits.size(1)  # 2*batch_size
        new_batch_size = logits.size(0)  # 2*batch_size
        batch_size = new_batch_size // 2
        device = logits.device
        arange = torch.arange(batch_size, device=device)

        # make sure 32 bits
        logits = logits.float() / self.temperature

        if not self.is_self_contrastive:
            # select all but current example.
            mask = ~torch.eye(new_batch_size, device=device).bool()
            n_classes -= 1  # remove the current example due to masking

            # infoNCE is essentially the same as a softmax (where weights are other z => try to predict
            # index of positive z). The label is the index of the positive z, which for each z is the
            # index that comes batch_size - 1 after it (batch_size because you concatenated) -1 because
            # you masked select all but the current z. arange takes care of idx of z which increases
            pos_idx = torch.cat([arange + batch_size - 1, arange], dim=0)

            logits = logits[mask].view(new_batch_size, n_classes)
            if not self.is_margin_loss:
                hat_H_mlz = F.cross_entropy(logits, pos_idx, reduction="none")
            else:
                hat_H_mlz = F.multi_margin_loss(logits, pos_idx, reduction="none")

        else:
            if not self.is_margin_loss:
                # here just predict 0.5 probability for both => no masking needed. This should somehow ensure invariance
                # all the add examples have 0 probability under p
                log_q = logits.log_softmax(-1)[:, :new_batch_size]
                # only keep the probability you assign to the 2 positives then sum and divide by 2
                # essentially multiply and sum by p giving mass 0.5 to each
                mask = torch.eye(batch_size, device=device).bool().repeat(2, 2)
                log_q = log_q[mask].view(new_batch_size, 2)
                hat_H_mlz = - log_q.sum(1) / 2

            else:  # TODO rm if do not use
                pos_idx = torch.eye(batch_size, device=device).repeat(2, 2).long()
                hat_H_mlz = F.multilabel_margin_loss(logits, pos_idx, reduction="none")

        hat_H_m = math.log(n_classes)
        logs = dict(
            I_q_zm=(hat_H_m - hat_H_mlz.mean()),
            hat_H_m=hat_H_m,
            n_negatives=float(n_classes),  # lightning expects float for logging
            temperature=self.temperature
        )

        return hat_H_mlz, logs
