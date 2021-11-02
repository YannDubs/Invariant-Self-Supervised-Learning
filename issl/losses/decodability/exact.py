"""cross entropy to minimize directly R[M(X)|Z] and thus ensure decodability."""
from __future__ import annotations

import copy
import queue
from collections import Callable, Sequence
from typing import Any, Optional

import torch
import torch.nn as nn
from issl.architectures import get_Architecture
from issl.helpers import weights_init
from torch.nn import functional as F

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

    predictor_kwargs : dict, optional
        Arguments to get `Predictor` from `get_Architecture`.
    """

    def __init__(
        self,
        z_shape: Sequence[int],
        m_shape: Sequence[int],
        loss: str = "CrossEntropyLoss",
        predictor_kwargs: dict[str, Any] = {"architecture": "linear"},
    ) -> None:
        super().__init__()
        self.z_shape = z_shape
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
        self, z: torch.Tensor, m: torch.Tensor, _
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

        # shape: [batch_size, M_shape]
        hat_R_mlz = self.compute_loss(M_pred, m)

        logs = dict()
        other = dict()

        return hat_R_mlz, logs, other
