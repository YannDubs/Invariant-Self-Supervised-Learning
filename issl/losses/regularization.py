"""Regularizers for H[Z] and thus improve downstream sample efficiency."""
from __future__ import annotations

from typing import Any, Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from issl.helpers import (BatchRMSELoss, at_least_ndim, eye_like, prod, rel_distance,
                          corrcoeff_to_eye_loss, rayleigh_coeff)


def get_regularizer(mode: Optional[str], **kwargs) -> Optional[torch.nn.Module]:
    if mode == "coarsener":
        return CoarseningRegularizer(**kwargs)
    elif mode == "coarsenerMx":
        return CoarseningRegularizerMx(**kwargs)
    elif mode == "effdim":
        return EffdimRegularizer(**kwargs)
    elif mode is None:
        return None
    else:
        raise ValueError(f"Unknown mode={mode}.")


class CoarseningRegularizer(nn.Module):
    """Regularizer that ensures that you have a coarser representation by essentially minimizing I[Z,X|M(X)]. In practice
    this simply brings enc(x) and enc(a) closer together using different losses.

    Parameters
    ---------
    loss : {"kl", "rmse", "cosine", "huber", "rel_l1"} or callable, optional
        What loss to use to bring enc(x) and enc(a) closer together. If `RMSE` or `cosine` will sample through the rep.
        If kl should be stochastic representation. Makes more sense to use a symmetric function and thus assume X ~ A.
    """

    def __init__(
        self,
        loss: Any,
    ) -> None:
        super().__init__()

        if loss == "rmse":
            self.loss_f = BatchRMSELoss()
        elif loss == "huber":
            self.loss_f = nn.SmoothL1Loss(reduction="none")
        elif loss == "rel_l1_clamp":
            # clamp to 0.01 to avoid distance of negatives to go to infty (numerically unstable)
            # 0.01 means that relative distance of positive is 1% of negatives
            # TODO should only clamp gradients of the negatives. Positives should still be forced to collapse
            self.loss_f = lambda x1,x2: rel_distance(x1,x2, p=1.0).clamp(min=0.01)
        elif loss == "rel_l1":
            self.loss_f = lambda x1,x2: rel_distance(x1,x2, p=1.0)
        elif loss == "corrcoef":
            self.loss_f = corrcoeff_to_eye_loss
        elif loss == "rayleigh":
            self.loss_f = rayleigh_coeff
        elif loss == "cosine":
            self.loss_f = lambda z, z_a: - F.cosine_similarity(
                z.flatten(1, -1), z_a.flatten(1, -1), dim=-1, eps=1e-08
            )
        elif not isinstance(loss, str):
            self.loss_f = loss
        else:
            raise ValueError(f"Unknown loss={loss}.")

    def forward(
            self, z: torch.Tensor, _, __
    ) -> tuple[torch.Tensor, dict, dict]:
        """Contrast examples and compute the upper bound on R[A|Z].

        Parameters
        ----------
        z : Tensor shape=[2 * batch_size, z_dim]
            Samples from the encoder.

        Returns
        -------
        loss : torch.Tensor shape=[batch_shape]
            Estimate of the loss.

        logs : dict
            Additional values to log.

        other : dict
            Additional values to return.
        """
        z_x, z_a = z.chunk(2, dim=0)

        # shape : [batch]
        loss = self.loss_f(z_x, z_a)
        loss = einops.reduce(loss, "b ... -> b", "sum")

        logs = dict()
        other = dict()
        return loss, logs, other

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
        self, z: torch.Tensor, Mx: torch.Tensor, _,
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


class EffdimRegularizer(nn.Module):
    """Increases effective dimensionality by each dimension independent.

    Parameters
    ---------
    z_shape : list or int
        Shape of representation.

    is_use_augmented : bool, optional
        Whether to compute the cross correlation between examples with different augmentations rather than same
        augmentations, Both give optimal representations.
    """

    def __init__(
        self,
        z_shape,
        is_use_augmented: bool = True,
    ) -> None:
        super().__init__()
        self.is_use_augmented = is_use_augmented

        z_dim = z_shape if isinstance(z_shape, int) else prod(z_shape)
        self.corr_coef_bn = torch.nn.BatchNorm1d(z_dim, affine=False)

    def forward(
        self, z: torch.Tensor, _, __
    ) -> tuple[torch.Tensor, dict, dict]:
        z_x, z_a = z.chunk(2, dim=0)

        batch_size, dim = z_x.shape
        z_a = z_a if self.is_use_augmented else z_x

        corr_coeff = (self.corr_coef_bn(z_x).T @ self.corr_coef_bn(z_a)) / batch_size

        pos_loss = (corr_coeff.diagonal() - 1).pow(2)
        neg_loss = corr_coeff.masked_select(~eye_like(corr_coeff).bool()).view(dim, dim - 1).pow(2).mean(1)
        logs = dict()
        other = dict()

        return pos_loss + neg_loss, logs, other