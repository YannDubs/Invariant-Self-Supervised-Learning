"""Regularizers for H[Z] and thus improve downstream sample efficiency."""
from __future__ import annotations

from collections import Sequence
from typing import Any, Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from issl import get_marginalDist
from issl.helpers import BatchRMSELoss, at_least_ndim, kl_divergence


def get_regularizer(mode: Optional[str], **kwargs) -> Optional[torch.nn.Module]:
    if mode == "coarsener":
        return CoarseningRegularizer(**kwargs)
    elif mode == "prior":
        return PriorRegularizer(**kwargs)
    elif mode == "coarsenerMx":
        return CoarseningRegularizerMx(**kwargs)
    elif mode is None:
        return None
    else:
        raise ValueError(f"Unknown mode={mode}.")


class CoarseningRegularizer(nn.Module):
    """Regularizer that ensures that you have a coarser representation by essentially minimizing I[Z,X|M(X)]. In practice
    this simply brings enc(x) and enc(a) closer together using different losses.

    Parameters
    ---------
    loss : {"kl", "rmse", "cosine", "huber"} or callable, optional
        What loss to use to bring enc(x) and enc(a) closer together. If `RMSE` or `cosine` will sample through the rep.
        If kl should be stochastic representation. Makes more sense to use a symmetric function and thus assume X ~ A.

    is_distributions : bool, optional
        Whether the loss should take in the distributions. Only used if the loss is callable.

    is_aux_already_represented : bool, optional
        Whether the positive examples are already represented => no need to use p_ZlX again.
        In this case `p_ZlX` will be replaced by a placeholder distribution. Useful
        for clip, where the positive examples are text sentences that are already represented.
    """

    def __init__(
        self,
        loss: Any,
        is_distributions: bool = False,
        is_aux_already_represented: bool = False,
    ) -> None:
        super().__init__()
        self.is_distributions = is_distributions
        self.is_aux_already_represented = is_aux_already_represented

        if loss == "kl":
            # use symmetric kl divergence
            self.loss_f = lambda p, q: (kl_divergence(p, q) + kl_divergence(q, p)) / 2
            self.is_distributions = True
        elif loss == "rmse":
            self.loss_f = BatchRMSELoss()
            self.is_distributions = False
        elif loss == "huber":
            self.loss_f = nn.SmoothL1Loss(reduction="none")
            self.is_distributions = False
        elif loss == "cosine":
            self.loss_f = lambda z, z_a: - F.cosine_similarity(
                z.flatten(1, -1), z_a.flatten(1, -1), dim=-1, eps=1e-08
            )
            self.is_distributions = False
        elif not isinstance(loss, str):
            self.loss_f = loss
        else:
            raise ValueError(f"Unknown loss={loss}.")

    def forward(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        p_Zlx: torch.distributions.TransformedDistribution,
        parent: Any,
    ) -> tuple[torch.Tensor, dict, dict]:
        """Contrast examples and compute the upper bound on R[A|Z].

        Parameters
        ----------
        z : Tensor shape=[batch_size, z_dim]
            Samples from the encoder.

        a : Tensor shape=[batch_size, *x_shape]
            Augmented sample.

        p_Zlx : torch.Distribution
            Encoded distribution.

        parent : ISSLModule, optional
            Parent module. Should have attribute `parent.p_ZlX`.

        Returns
        -------
        loss : torch.Tensor shape=[batch_shape]
            Estimate of the loss.

        logs : dict
            Additional values to log.

        other : dict
            Additional values to return.
        """
        if self.is_distributions:
            assert not self.is_aux_already_represented
            in_x = p_Zlx
            in_a = parent.p_ZlX(a)
        else:
            # shape: [batch_size, z_dim]
            if self.is_aux_already_represented:
                # sometimes already represented, e.g., for CLIP the sentences are pre represented.
                z_a = a
            else:
                z_a = parent(a, is_sample=True, is_process=True)

            in_x = z
            in_a = z_a

        # shape : [batch]
        loss = self.loss_f(in_x, in_a)
        loss = einops.reduce(loss, "b ... -> b", "sum")

        logs = dict()
        other = dict()
        return loss, logs, other


class PriorRegularizer(nn.Module):
    """Regularizer of the mutual information I[Z,X] by using a  prior."""

    def __init__(self, family: str, z_shape: Sequence[int]):
        super().__init__()
        # TODO: make it work with z_shape instead of z_dim
        self.q_Z = get_marginalDist(family, z_shape)

    def forward(
        self, z, a, p_Zlx: torch.distributions.TransformedDistribution, parent,
    ) -> tuple[torch.Tensor, dict, dict]:
        """Contrast examples and compute the upper bound on R[A|Z].

        Parameters
        ----------
        z : Tensor shape=[batch_size, z_dim]
            Samples from the encoder.
            
        p_Zlx : torch.Distribution
            Encoded distribution.

        Returns
        -------
        kl : torch.Tensor shape=[batch_shape]
            Estimate of the kl divergence.

        logs : dict
            Additional values to log.

        other : dict
            Additional values to return.
        """
        # batch shape: [] ; event shape: [z_shape]
        q_Z = self.q_Z()

        # E_x[KL[p(Z|x) || q(Z)]]. shape: [batch_size]
        kl = kl_divergence(p_Zlx, q_Z, z_samples=z)

        logs = dict(I_q_ZX=kl.mean(), H_ZlX=p_Zlx.entropy().mean(0))

        # upper bound on H[Z] (cross entropy)
        logs["H_q_Z"] = logs["I_q_ZX"] + logs["H_ZlX"]
        other = dict()
        return kl, logs, other


class CoarseningRegularizerMx(nn.Module):
    """Similar to CoarseningRegularizer but the input is Mx (which can generally not be encoded with p(Z|X)) rather than
    some a \in \mathcal{X}.

    Parameters
    ---------
    loss : {"l2", "cosine", "l1"}, optional
        What loss to use to bring any enc(x) associated with the same Mx closer together. .
    """

    def __init__(self, loss: Any) -> None:
        super().__init__()
        self.loss = loss

    def forward(
        self, z: torch.Tensor, Mx: torch.Tensor, _, __,
    ) -> tuple[torch.Tensor, dict, dict]:
        """Contrast examples and compute the upper bound on R[A|Z].

        Parameters
        ----------
        z : Tensor shape=[batch_size, z_dim]
            Samples from the encoder.

        Mx : Tensor shape=[batch_size, *Mx_shape]
            Maximal invariants.

        Returns
        -------
        loss : torch.Tensor shape=[batch_shape]
            Estimate of the loss. Note that if there are no 2 batches with the same Mx, then will
            filter those examples out. So the final shape might be smaller.

        logs : dict
            Additional values to log.

        other : dict
            Additional values to return.
        """
        # shape=[batch_size, Mx_dim]
        Mx = at_least_ndim(Mx, ndim=2).flatten(1)
        # shape=[batch_size, z_dim]
        z = z.flatten(1)

        # shape=[batch_size, batch_size]
        mask_Mx = torch.eq(Mx.T, Mx).float()
        mask_Mx.fill_diagonal_(0)

        # shape=[batch_size, batch_size]
        if self.loss == "l2":
            dist = torch.cdist(z, z, p=2.0)

        elif self.loss == "l1":
            dist = torch.cdist(z, z, p=1.0)

        elif self.loss == "cosine":
            dist = - F.cosine_similarity(
                z.unsqueeze(-1), z.T.unsqueeze(0), dim=1, eps=1e-08
            )

        else:
            raise ValueError(f"Unknown loss={self.loss}.")

        # shape : between [0] and [batch]. Mask out examples where there are no 2 with same Mx
        n_select = mask_Mx.sum(1)
        mask = n_select > 0
        if mask.any():
            loss = (mask_Mx * dist).sum(1).masked_select(mask)
            loss = loss / n_select.masked_select(mask)
        else:
            loss = None

        logs = dict()
        other = dict()

        return loss, logs, other
