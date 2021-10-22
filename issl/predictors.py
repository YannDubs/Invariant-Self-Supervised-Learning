from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from sklearn.pipeline import Pipeline
from torchmetrics.functional import accuracy

from .architectures import get_Architecture
from .helpers import (
    append_optimizer_scheduler_,
    namespace2dict,
    prediction_loss,
    weights_init,
)

__all__ = [
    "Predictor",
    "OnlineEvaluator",
    "get_representor_predictor",
    "SklearnPredictor",
]

logger = logging.getLogger(__name__)


def get_representor_predictor(representor: pl.LightningModule):
    """
    Helper function that returns a Predictor with correct representor. Cannot use partial because
    need lightning module and cannot give as kwargs to load model because not pickable)
    """

    class FeatPred(Predictor):
        def __init__(self, hparams, representor=representor):
            super().__init__(hparams, representor=representor)

    return FeatPred


class Predictor(pl.LightningModule):
    """Main network for downstream prediction."""

    def __init__(
        self, hparams: Any, representor: Optional[pl.LightningModule] = None
    ) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.is_clf = self.hparams.data.target_is_clf

        if representor is not None:
            # ensure not saved in checkpoint and frozen
            representor.set_represent_mode_()
            representor.stage = "pred_repr"
            self.representor = representor
            pred_in_shape = self.hparams.encoder.z_shape

        else:
            self.representor = torch.nn.Identity()
            pred_in_shape = self.hparams.data.shape

        cfg_pred = self.hparams.predictor.kwargs
        Architecture = get_Architecture(cfg_pred.architecture, **cfg_pred.arch_kwargs)
        self.predictor = Architecture(pred_in_shape)

        self.stage = self.hparams.stage

    def forward(self, x: torch.Tensor, is_logits: bool = True) -> torch.Tensor:
        """Perform prediction for `x`.

        Parameters
        ----------
        x : torch.Tensor of shape=[batch_size, *data.shape]
            Data to represent.

        is_logits : bool, optional
            Whether to return the logits instead of classes probability in case you are using using
            classification.

        Returns
        -------
        Y_pred : torch.Tensor of shape=[batch_size, *target_shape]

        if is_return_logs:
            logs : dict
                Dictionary of values to log.
        """
        with torch.no_grad():
            # shape: [batch_size,  *z.out_shape]
            z = self.representor(x)

        z = z.detach()  # shouldn't be needed

        # shape: [batch_size,  *target_shape]
        Y_pred = self.predictor(z)

        if not is_logits and self.is_clf:
            out = Y_pred.softmax(-1)
        else:
            out = Y_pred

        return out

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: Optional[int] = None
    ):
        """
        Predict function, this will represent the data and also return the correct label.
        Which is useful in case you want to create a represented dataset.
        """
        x, y = batch
        y_hat = self(x)
        return y_hat.cpu(), y.cpu()

    def add_balanced_logs_(
        self, loss: torch.Tensor, y: torch.Tensor, Y_hat: torch.Tensor, logs: dict
    ) -> None:
        """Add evaluation using balanced loss (ie mean per class) which can be good if imbalanced."""
        mapper = self.hparams.data.balancing_weights

        sample_weights = torch.tensor(
            [mapper[str(yi.item())] for yi in y], device=y.device
        )
        logs["balanced_loss"] = (loss * sample_weights).mean()

        if self.is_clf:
            is_same = (Y_hat.argmax(dim=-1) == y).float()
            balanced_acc = (is_same * sample_weights).mean()
            logs["balanced_acc"] = balanced_acc
            logs["balanced_err"] = 1 - logs["balanced_acc"]

    def step(self, batch: torch.Tensor) -> tuple[torch.Tensor, dict]:
        x, y = batch

        # shape: [batch_size,  *target_shape]
        Y_hat = self(x)

        # Shape: [batch, 1]
        loss, logs = self.loss(Y_hat, y)

        if not self.training and len(self.hparams.data.balancing_weights) > 0:
            # for some datasets we have to evaluate using the mean per class loss / accuracy
            # we don't train it using that (because shouldn't have access to those weights during train)
            # but we compute it during evaluation
            self.add_balanced_logs_(loss, y, Y_hat, logs)

        # Shape: []
        loss = loss.mean()

        logs["loss"] = loss
        if self.is_clf:
            logs["acc"] = accuracy(Y_hat.argmax(dim=-1), y)
            logs["err"] = 1 - logs["acc"]

        return loss, logs

    def loss(self, Y_hat: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Compute the MSE or cross entropy loss."""

        loss = prediction_loss(Y_hat, y, self.is_clf)
        logs = dict()

        return loss, logs

    def training_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> Optional[torch.Tensor]:
        loss, logs = self.step(batch)
        self.log_dict(
            {f"train/{self.stage}/{k}": v for k, v in logs.items()}, sync_dist=True
        )
        return loss

    def test_val_step(
        self, batch: torch.Tensor, batch_idx: int, mode: str
    ) -> Optional[torch.Tensor]:
        loss, logs = self.step(batch)

        self.log_dict(
            {f"{mode}/{self.stage}/{k}": v for k, v in logs.items()}, sync_dist=True,
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


class SklearnPredictor(Pipeline):
    """Sklearn predictor."""

    def __init__(self, cfgp):
        self.cfgp = cfgp
        steps = []

        dict_cfgp = namespace2dict(self.cfgp)
        if self.cfgp.is_scale:
            steps += [("scaler", instantiate(dict_cfgp["scaler"]))]

        steps += [("model", instantiate(dict_cfgp["model"]))]

        super().__init__(steps)


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
        If module should be instantiated using `Architecture(in_shape, out_dim)`. If str will be given to
        `get_Architecture`.

    arch_kwargs : dict, optional
        Arguments to `get_Architecture`.

    is_classification : bool, optional
        Whether or not the task is a classification one.
    """

    def __init__(
        self,
        in_shape: Sequence[int],
        out_shape: Sequence[int],
        architecture: Union[str, Callable],
        arch_kwargs: dict[str, Any] = {},
        is_classification: bool = True,
    ) -> None:
        super().__init__()
        Architecture = get_Architecture(architecture, **arch_kwargs)
        self.model = Architecture(in_shape, out_shape)
        self.is_classification = is_classification

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
            # Shape: [batch, *z_shape]
            z = encoder(x)

        z = z.detach()

        # Shape: [batch, *Y_shape]
        Y_hat = self.model(z)

        # Shape: [batch]
        loss = prediction_loss(Y_hat, y, self.is_classification)

        # Shape: []
        loss = loss.mean()

        logs = dict(eval_loss=loss)
        if self.is_classification:
            logs["eval_acc"] = accuracy(Y_hat.argmax(dim=-1), y)
            logs["eval_err"] = 1 - logs["eval_acc"]

        return loss, logs
