"""cross entropy to minimize directly R[M(X)|Z] and thus ensure decodability."""
from __future__ import annotations

from collections import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einops
from torchmetrics.functional import accuracy

from issl.architectures import get_Architecture
from issl.helpers import weights_init

__all__ = [
    "ExactISSL",
]


class ExactISSL(nn.Module):
    """Compute the ISSL loss directly by predicting the real M(X). This is essentially
    an exact (but unrealistic) version of selfdistillation_prior

    Parameters
    ----------
    z_shape : sequence of int
        Shape of the representation.

    m_shape : sequence of int
        Shape of the max invariant to predict.

    loss : str, optional
        Loss with represent to which to compute R[M(X)|Z]. Should be a name of a module in torch.nn
        that takes `reduction=None` as input.

    is_to_one_hot : bool, optional
        Whether to one hot encoder the maximal invariant before giving it to the loss.

    predictor_kwargs : dict, optional
        Arguments to get `Predictor` from `get_Architecture`.
    """

    def __init__(
        self,
        z_shape: Sequence[int],
        m_shape: Sequence[int],
        loss: str = "CrossEntropyLoss",
        is_to_one_hot: bool = False,
        predictor_kwargs: dict[str, Any] = {"architecture": "linear"},
        encoder: Optional[nn.Module] = None  # only used for DINO
    ) -> None:
        super().__init__()
        self.z_shape = z_shape
        self.is_to_one_hot = is_to_one_hot
        self.compute_loss = getattr(nn, loss)(reduction="none")
        self.m_shape = m_shape
        self.predictor_kwargs = predictor_kwargs

        Predictor = get_Architecture(
            in_shape=self.z_shape, out_shape=self.m_shape, **self.predictor_kwargs
        )
        self.predictor = Predictor()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        weights_init(self)

    def forward(
        self, z: torch.Tensor, _, __, m: torch.Tensor, parent
    ) -> tuple[torch.Tensor, dict, dict]:
        """Self distillation of examples and compute the upper bound on R[A|Z].

        Parameters
        ----------
        z : Tensor shape=[batch_size, *z_shape]
            Sampled representation.

        m : Tensor shape=[batch_size, *m_shape]
            Maximal invariants.

        Returns
        -------
        hat_R_MlZ : torch.Tensor shape=[batch_shape]
            Estimate of R[M(X)|Z].

        logs : dict
            Additional values to log.

        other : dict
            Additional values to return.
        """

        # shape: [batch_size, M_shape]
        M_pred = self.predictor(z)

        m_input = m
        if self.is_to_one_hot:
            m_input = F.one_hot(m_input, num_classes=self.m_shape[0]).float() * 2 - 1

        # shape: [batch_size]
        hat_R_mlz = self.compute_loss(M_pred, m_input)
        hat_R_mlz = einops.reduce(hat_R_mlz, "b ... -> b", reduction="mean")

        logs = dict()
        logs["Mx_acc"] = accuracy(M_pred.argmax(dim=-1), m)
        logs["Mx_err"] = 1 - logs["Mx_acc"]

        other = dict()

        return hat_R_mlz, logs, other
