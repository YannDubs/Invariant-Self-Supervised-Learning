"""Generative proxies to minimize R[A|Z] and thus ensure decodability."""
from __future__ import annotations

from collections import Sequence

import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = ["GenerativeISSL"]

from issl.architectures import get_Architecture
from issl.helpers import (
    UnNormalizer,
    is_colored_img,
    is_img_shape,
    prediction_loss,
    weights_init,
)
import einops


class GenerativeISSL(nn.Module):
    """Computes the ISSL loss using generative variational bound (i.e. generate the augmented example).

    Parameters
    ----------
    z_shape : sequence of int
        Shape of the representation.

    a_shape : tuple of int or int
        Shape of auxiliary target to generate.

    decoder_kwargs : dict, optional
        Arguments to get `Decoder` from `get_Architecture`.

    normalized : str, optional
        Name of the normalization to undo. If `None` then nothing.  This is important to know whether needs to be
        unnormalized when comparing in case you are reconstructing the input. Currently only works for colored
        images.

    pred_loss_kwargs : dict, optional
        Additional arguments to `prediction_loss`.
    """

    def __init__(
        self,
        z_shape: Sequence[int],
        a_shape: Sequence[int],
        decoder_kwargs: dict = {"architecture": "linear"},
        normalized=None,
        pred_loss_kwargs: dict = {},
    ) -> None:
        super().__init__()

        Decoder = get_Architecture(**decoder_kwargs)

        # this will return the sufficient statistics for q(Y|Z)
        self.suff_stat_AlZ = Decoder(z_shape, a_shape)
        self.normalized = normalized
        self.is_img_out = is_img_shape(a_shape)
        self.pred_loss_kwargs = pred_loss_kwargs

        if self.normalized is not None:
            self.unnormalizer = UnNormalizer(self.normalized)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        weights_init(self)

    def forward(
        self, z: torch.Tensor, aux_target: torch.Tensor, _
    ) -> tuple[torch.Tensor, dict, dict]:
        """Generate A and compute the upper bound on R[A|Z].

        Parameters
        ----------
        z : Tensor shape=[batch_size, z_dim]
            Sampled representation.

        aux_target : Tensor shape=[batch_size, *aux_shape]
            Auxiliary target to predict.

        Returns
        -------
        loss : torch.Tensor shape=[batch_shape]
            Estimate of the loss.

        logs : dict
            Additional values to log.

        other : dict
            Additional values to return.
        """

        # shape: [batch_size, *a_shape]
        a_hat = self.suff_stat_AlZ(z)

        # -log p(ai|zi). shape: [batch_size, *a_shape]
        # all of the following should really be written in a single line using log_prob where P_{A|Z}
        # is an actual conditional distribution (categorical if cross entropy, and gaussian for mse),
        # but this might be less understandable for usual deep learning + less numerically stable
        if self.is_img_out:
            if is_colored_img(aux_target):
                if self.is_normalized:
                    # compare in unnormalized space
                    aux_target = self.unnormalizer(aux_target)

                # output is linear but data is in 0,1 => for a better comparison put in [0,1]
                a_hat = torch.sigmoid(a_hat)

                # color image => uses gaussian distribution
                neg_log_q_alz = F.mse_loss(a_hat, aux_target, reduction="none")
            else:
                # black white image => uses categorical distribution, with logits for stability
                neg_log_q_alz = F.binary_cross_entropy_with_logits(
                    a_hat, aux_target, reduction="none"
                )

                # but for saving you still want the image in [0,1]
                a_hat = torch.sigmoid(a_hat)
        else:  # normal pred if not images
            neg_log_q_alz = prediction_loss(a_hat, aux_target, **self.pred_loss_kwargs)

        # -log p(y|z). shape: [batch_size]
        # mathematically should take a sum (log prod proba -> sum log proba), but usually people take mean
        neg_log_q_alz = einops.reduce(neg_log_q_alz, "b ... -> b", reduction="sum",)

        # T for auxiliary task to distinguish from task Y
        logs = dict(H_q_AlZ=neg_log_q_alz.mean())

        other = dict()
        # for plotting (note that they are already unormalized)
        other["A_hat"] = a_hat[0].detach().cpu()
        other["A"] = aux_target[0].detach().cpu()

        return neg_log_q_alz, logs, other
