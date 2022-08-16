from __future__ import annotations


import logging
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

import pytorch_lightning as pl
import torch
from torchmetrics.functional import accuracy

from .architectures import get_Architecture
from .helpers import (
    append_optimizer_scheduler_,
    prediction_loss,
    weights_init,
)

__all__ = ["Predictor", "OnlineEvaluator"]

logger = logging.getLogger(__name__)



class Predictor(pl.LightningModule):
    """Main network for downstream prediction."""

    def __init__(self, hparams) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        cfg_pred = self.hparams.predictor.kwargs
        Architecture = get_Architecture(cfg_pred.architecture, **cfg_pred.arch_kwargs)
        self.predictor = Architecture(self.hparams.data.shape)
        self.stage = self.hparams.stage

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Perform prediction for `x`.

        Parameters
        ----------
        z : torch.Tensor of shape=[batch_size, z_dim]
            Data to represent.

        Returns
        -------
        Y_pred : torch.Tensor of shape=[batch_size, *target_shape]
            First is for predicting the task, all others are for the agg tasks.

        Y_preds_agg : list of tensors of shape=[batch_size, *target_shape]
            First is for predicting the task, all others are for the agg tasks.
        """
        # shape: [batch_size,  *target_shape]
        Y_pred = self.predictor(z)
        return Y_pred

    def step(self, batch: torch.Tensor) -> tuple[torch.Tensor, dict]:
        x, y = batch

        # list of Y_hat. Each Y_hat shape: [batch_size,  *target_shape]
        Y_hat, Ys_hat_agg = self(x)

        # Shape: [batch, 1]
        loss, logs = self.loss(Y_hat, y)

        return loss.mean(), logs

    def loss(self, Y_hat: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Compute the MSE or cross entropy loss."""

        loss = prediction_loss(Y_hat, y)

        logs = dict()
        logs["loss"] = loss.mean()
        logs["acc"] = accuracy(Y_hat.argmax(dim=-1), y)
        logs["err"] = 1 - logs["acc"]

        return loss, logs


    def training_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> Optional[torch.Tensor]:

        loss, logs = self.step(batch)
        self.log_dict(
            {
                f"train/{self.stage}/{self.hparams.task}/{k}": v
                for k, v in logs.items()
            },
            sync_dist=True,
        )
        return loss

    def test_val_step(
        self, batch: torch.Tensor, batch_idx: int, mode: str
    ) -> Optional[torch.Tensor]:
        loss, logs = self.step(batch)

        self.log_dict(
            {
                f"{mode}/{self.stage}/{self.hparams.task}/{k}": v
                for k, v in logs.items()
            },
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        return self.test_val_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.test_val_step(batch, batch_idx, "test")

    def configure_optimizers(self):

        optimizers, schedulers = [], []

        append_optimizer_scheduler_(
            self.hparams.optimizer_pred,
            self.hparams.scheduler_pred,
            self.parameters(),
            optimizers,
            schedulers,
            name="lr_predictor",
        )

        return optimizers, schedulers


class OnlineEvaluator(torch.nn.Module):
    """
    Attaches MLP/linear predictor for evaluating the quality of a representation as usual in self-supervised.

    Notes
    -----
    -  generalizes `pl_bolts.callbacks.ssl_online.SSLOnlineEvaluator` for multilabel clf and regression
    and does not use a callback as pytorch lightning doesn't work well with trainable callbacks.

    Parameters
    ----------
    in_shape : int
        Input dimension.

    out_shape : tuple of int
        Shape of the output

    architecture : str or Callable
        If module should be instantiated using `Architecture(in_shape, n_equivalence_classes)`. If str will be given to
        `get_Architecture`.

    arch_kwargs : dict, optional
        Arguments to `get_Architecture`.
    """

    def __init__(
        self,
        in_shape: Sequence[int],
        out_shape: Sequence[int],
        architecture: Union[str, Callable],
        arch_kwargs: dict[str, Any] = {},
    ) -> None:
        super().__init__()
        Architecture = get_Architecture(architecture, **arch_kwargs)
        self.model = Architecture(in_shape, out_shape)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        weights_init(self)

    def forward(
        self, batch: torch.Tensor, encoder: pl.LightningModule
    ) -> tuple[torch.Tensor, dict]:
        x, y = batch

        if isinstance(y, Sequence):
            y = y[0]  # only return the real label assumed to be first

        with torch.no_grad():
            # Shape: [batch, z_dim]
            z = encoder(x)

        z = z.detach()

        # Shape: [batch, *Y_shape]
        Y_hat = self.model(z)

        # Shape: [batch]
        loss = prediction_loss(Y_hat, y)

        # Shape: []
        loss = loss.mean()

        logs = dict(eval_loss=loss)
        logs["eval_acc"] = accuracy(Y_hat.argmax(dim=-1), y)
        logs["eval_err"] = 1 - logs["eval_acc"]

        return loss, logs
