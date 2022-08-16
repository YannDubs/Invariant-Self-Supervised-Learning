"""Contrastive proxies to minimize ISSL log loss."""
from __future__ import annotations

import copy
import math
from typing import Any

import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

from issl.architectures import get_Architecture
from issl.helpers import weights_init

logger = logging.getLogger(__name__)

__all__ = ["SimCLR", "CISSL"]

class BaseContrastiveISSL(nn.Module):
    """Computes the ISSL loss using contrastive variational bound (i.e. with positive and negative examples).

    Notes
    -----
    - parts of code taken from https://github.com/lucidrains/contrastive-learner

    Parameters
    ----------
    z_dim : int
        Dimensionality of the representation.

    temperature : float, optional
        Temperature scaling in InfoNCE. Recommended less than 1.

    is_pred_proj_same : bool, optional
        Whether to use the projector for the predictor. Typically, True in SSL, but our theory says that it should be
        False.

    is_self_contrastive : bool, optional
        Whether to keep current example in batch when performing contrastive learning and then have to predict
        proba 0.5 to current and positive.

    projector_kwargs : dict, optional
        Arguments to get `Projector` from `get_Architecture`. Note that is `out_shape` is <= 1
        it will be a percentage of z_dim. To use no projector set `mode=Identity`.

    predictor_kwargs : dict, optional
        Arguments to get `Predictor` from `get_Architecture`. Note that is `out_shape` is <= 1
        it will be a percentage of z_dim. To use no predictor set `mode=Identity`.

    References
    ----------
    [1] Song, Jiaming, and Stefano Ermon. "Multi-label contrastive predictive coding." Advances in
    Neural Information Processing Systems 33 (2020).
    """

    def __init__(
        self,
        z_dim: int,
        temperature: float = 0.07,
        is_pred_proj_same: bool = False,
        is_self_contrastive: bool = False,
        projector_kwargs: dict[str, Any] = {
            "architecture": "mlp",
            "hid_dim": 1024,
            "n_hid_layers": 1,
            "norm_layer": "batch",
        },
        predictor_kwargs: dict[str, Any] = {"architecture": "linear", "out_shape": 128},
        **kwargs
    ) -> None:

        super().__init__()
        logger.info(f"Unused arguments {kwargs}.")

        self.z_dim = z_dim
        self.is_pred_proj_same = is_pred_proj_same
        self.predictor_kwargs = self.process_kwargs(predictor_kwargs)
        self.projector_kwargs = self.process_kwargs(projector_kwargs)
        self.is_self_contrastive = is_self_contrastive
        self.temperature = temperature

        if self.is_pred_proj_same:
            self.is_self_contrastive = False

        Projector = get_Architecture(**self.projector_kwargs)
        self.projector = self.add_batchnorms(Projector(), self.projector_kwargs)

        if self.is_pred_proj_same:
            self.predictor = self.projector
        else:
            Predictor = get_Architecture(**self.predictor_kwargs)
            self.predictor = self.add_batchnorms(Predictor(), self.predictor_kwargs)

        self.reset_parameters()

    def add_batchnorms(self, head, kwargs):
        if not (kwargs["architecture"].lower() in ["identity", "flatten"]):
            head = nn.Sequential(head, nn.BatchNorm1d(self.out_dim))
        return head

    def reset_parameters(self) -> None:
        weights_init(self)

    def process_kwargs(self, kwargs: dict) -> dict:
        kwargs = copy.deepcopy(kwargs)  # ensure mutable object is ok
        kwargs["in_shape"] = kwargs.get("in_shape", self.z_dim)
        kwargs["out_shape"] = kwargs.get("out_shape", self.z_dim)
        self.out_dim = kwargs["out_shape"]
        return kwargs

    def forward(
        self, z: torch.Tensor, z_tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Contrast examples and compute the upper bound on R[A|Z].
    
        Parameters
        ----------
        z : Tensor shape=[2 * batch_size, z_dim]
            Representations.

        z_tgt : Tensor shape=[2 * batch_size, z_dim]
            Representation from the other branch.
    
        Returns
        -------
        loss : torch.Tensor shape=[batch_shape]
            Estimate of the loss.
    
        logs : dict
            Additional values to log.
        """

        # shape: [2 * batch_size, 2 * batch_size]
        logits = self.compute_logits_p_Alz(z, z_tgt)

        # shape: [2 * batch_size]
        hat_H_mlz, logs = self.compute_loss(logits)

        return hat_H_mlz.mean(), logs

    def compute_logits_p_Alz(
        self, z_src: torch.Tensor, z_tgt: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the logits for the contrastive predictor p(A|Z)."""

        # shape: [2*batch_size, out_shape]
        # normalize to use cosine similarity
        z_src = F.normalize(self.predictor(z_src), dim=1, p=2)
        z_tgt = F.normalize(self.projector(z_tgt), dim=1, p=2)

        # shape: [2*batch_size, 2*batch_size]
        logits = z_src @ z_tgt.T

        # make sure 32 bits
        logits = logits.float() / self.temperature

        return logits

    def compute_loss(self, logits: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Computes the upper bound H_q[A|Z]."""
        n_classes = logits.size(1)  # 2*batch_size
        new_batch_size = logits.size(0)  # 2*batch_size
        batch_size = new_batch_size // 2
        device = logits.device
        arange = torch.arange(batch_size, device=device)

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
            hat_H_mlz = F.cross_entropy(logits, pos_idx, reduction="none")

        else:
            # here just predict 0.5 probability for both => no masking needed. This should somehow ensure invariance
            # all the add examples have 0 probability under p
            log_q = logits.log_softmax(-1)[:, :new_batch_size]
            # only keep the probability you assign to the 2 positives then sum and divide by 2
            # essentially multiply and sum by p giving mass 0.5 to each
            mask = torch.eye(batch_size, device=device).bool().repeat(2, 2)
            log_q = log_q[mask].view(new_batch_size, 2)
            hat_H_mlz = - log_q.sum(1) / 2

        hat_H_m = math.log(n_classes)  # only correct for CE but not important in any case

        logs = dict(
            I_q_zm=(hat_H_m - hat_H_mlz.mean()),
            hat_H_m=hat_H_m,
            n_negatives=float(n_classes),  # lightning expects float for logging
            temperature=self.temperature
        )

        return hat_H_mlz, logs


class CISSL(BaseContrastiveISSL):
    def __init__(self,
                 *args,
                is_pred_proj_same: bool = False,
                is_self_contrastive: bool = True,
                **kwargs):
        super().__init__(*args,
                         is_pred_proj_same=is_pred_proj_same,
                         is_self_contrastive=is_self_contrastive,
                         **kwargs)


class SimCLR(BaseContrastiveISSL):
    def __init__(self,
                 *args,
                is_pred_proj_same: bool = True,
                is_self_contrastive: bool = False,
                **kwargs):
        super().__init__(*args,
                         is_pred_proj_same=is_pred_proj_same,
                         is_self_contrastive=is_self_contrastive,
                         **kwargs)