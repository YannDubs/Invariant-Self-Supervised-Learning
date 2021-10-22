"""Regularizers for H[Z] and thus improve downstream sample efficiency."""
from __future__ import annotations

from typing import Any, Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from issl.helpers import kl_divergence


def get_regularizer(mode: Optional[str], **kwargs) -> Optional[torch.nn.Module]:
    if mode == "coarsener":
        return CoarseningRegularizer(**kwargs)
    elif mode is None:
        return None
    else:
        raise ValueError(f"Unknown mode={mode}.")


class CoarseningRegularizer(nn.Module):
    """Regularizer that ensures that you have a coarser representation by essentially minimizing I[Z,X|M(X)]. In practice
    this simply brings enc(x) and enc(a) closer together using different losses.

    Parameters
    ---------
    loss : {"kl", "mse", "cosine", "huber"} or callable, optional
        What loss to use to bring enc(x) and enc(a) closer together. If `MSE` or `cosine` will sample through the rep.
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
        elif loss == "mse":
            self.loss_f = nn.MSELoss()
            self.is_distributions = False
        elif loss == "huber":
            self.loss_f = nn.SmoothL1Loss()
            self.is_distributions = False
        elif loss == "cosine":
            self.loss_f = lambda z, z_a: F.cosine_similarity(
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
            Reconstructed representations.

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
                z_a = parent(a, is_sample=True)

            in_x = z
            in_a = z_a

        # shape : [batch]
        loss = self.loss_f(in_x, in_a)
        loss = einops.reduce(loss, "b ... -> b", "sum")

        logs = dict()
        other = dict()
        return loss, logs, other
