from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

from sklearn.pipeline import Pipeline

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from torch import nn
from torchmetrics.functional import accuracy

from .architectures import get_Architecture
from .helpers import (
    aggregate_dicts,
    append_optimizer_scheduler_,
    mean,
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
        self.is_agg_target = self.hparams.data.aux_target == "agg_target"

        if representor is not None:
            # ensure not saved in checkpoint and frozen
            representor.set_represent_mode_()
            representor.stage = "pred_repr"
            self.representor = representor
            pred_in_shape = self.hparams.kwargs.in_shape # use in_shape instead of z_shape in case you modify z (eg using get_Mx)

        else:
            self.representor = torch.nn.Identity()
            pred_in_shape = self.hparams.data.shape

        cfg_pred = self.hparams.predictor.kwargs
        Architecture = get_Architecture(cfg_pred.architecture, **cfg_pred.arch_kwargs)
        self.predictor = Architecture(pred_in_shape)

        if self.is_agg_target:
            cfgp = copy.deepcopy(cfg_pred)
            del cfgp.arch_kwargs.out_shape  # use same hparams besides out shape
            Arch = get_Architecture(cfgp.architecture, **cfgp.arch_kwargs)
            self.agg_predictors = nn.ModuleList(
                [Arch(pred_in_shape, out_shape=k) for k in self.hparams.data.aux_shape]
            )

        self.stage = self.hparams.stage

    def forward(
        self, x: torch.Tensor, is_logits: bool = True
    ) -> tuple[torch.Tensor, list]:
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
            First is for predicting the task, all others are for the agg tasks.

        Y_preds_agg : list of tensors of shape=[batch_size, *target_shape]
            First is for predicting the task, all others are for the agg tasks.
        """
        with torch.no_grad():
            # shape: [batch_size,  *z.out_shape]
            z = self.representor(x)

        z = z.detach()  # shouldn't be needed

        # shape: [batch_size,  *target_shape]
        Y_pred = self.apply_predictor(self.predictor, z, is_logits)

        if self.is_agg_target:
            # currently not in parallel. Waiting for pytorch/issues/36459
            Ys_pred_agg = [
                self.apply_predictor(p, z, is_logits) for p in self.agg_predictors
            ]
        else:
            Ys_pred_agg = []

        return Y_pred, Ys_pred_agg

    def apply_predictor(
        self, predictor: nn.Module, z: torch.Tensor, is_logits: bool
    ) -> torch.Tensor:
        Y_pred = predictor(z)

        if not is_logits and self.is_clf:
            out = Y_pred.softmax(-1)
        else:
            out = Y_pred

        return out

    def step(self, batch: torch.Tensor) -> tuple[torch.Tensor, dict]:
        x, y = batch

        if self.is_agg_target:
            y, y_agg = y

            if isinstance(y_agg, torch.Tensor):
                # sometimes will be list of labels and other time tensor => make all list
                y_agg = y_agg.unbind(-1)

        # list of Y_hat. Each Y_hat shape: [batch_size,  *target_shape]
        Y_hat, Ys_hat_agg = self(x)

        # Shape: [batch, 1]
        loss, logs = self.loss(Y_hat, y)

        if self.is_agg_target:
            loss_agg = []
            logs_agg = []
            for i in range(len(self.agg_predictors)):
                curr_loss, curr_logs = self.loss(Ys_hat_agg[i], y_agg[i])
                loss_agg.append(curr_loss)
                logs_agg.append(curr_logs)

            for k, v in aggregate_dicts(logs_agg, operation=mean).items():
                logs[f"{k}_agg"] = v  # agg is avg over all agg_tasks

            for k, v in aggregate_dicts(logs_agg, operation=max).items():
                logs[f"{k}_agg_max"] = v  # agg is max over all agg_tasks

            for k, v in aggregate_dicts(logs_agg, operation=min).items():
                logs[f"{k}_agg_min"] = v  # agg is min over all agg_tasks

            loss = loss + sum(loss_agg)  # sum all losses

        if not self.training and len(self.hparams.data.balancing_weights) > 0:
            # for some datasets we have to evaluate using the mean per class loss / accuracy
            # we don't train it using that (because shouldn't have access to those weights during train)
            # but we compute it during evaluation. Not adding for agg task, although could.
            self.add_balanced_logs_(loss, y, Y_hat, logs)

        # Shape: []
        loss = loss.mean()

        return loss, logs

    def loss(self, Y_hat: torch.Tensor, y: torch.Tensor,) -> tuple[torch.Tensor, dict]:
        """Compute the MSE or cross entropy loss."""

        loss = prediction_loss(Y_hat, y)

        logs = dict()
        logs["loss"] = loss.mean()
        logs["acc"] = accuracy(Y_hat.argmax(dim=-1), y)
        logs["err"] = 1 - logs["acc"]

        return loss, logs

    def add_balanced_logs_(
        self, loss: torch.Tensor, y: torch.Tensor, Y_hat: torch.Tensor, logs: dict
    ) -> None:
        """Add evaluation using balanced loss (ie mean per class) which can be good if imbalanced."""
        mapper = self.hparams.data.balancing_weights

        sample_weights = torch.tensor(
            [mapper[str(yi.item())] for yi in y], device=y.device
        )
        logs["balanced_loss"] = (loss * sample_weights).mean()

        is_same = (Y_hat.argmax(dim=-1) == y).float()
        balanced_acc = (is_same * sample_weights).mean()
        logs["balanced_acc"] = balanced_acc
        logs["balanced_err"] = 1 - logs["balanced_acc"]

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


class SklearnPredictor(Pipeline):
    """Sklearn predictor."""

    def __init__(self, cfgp, old_predictor=None):
        self.cfgp = cfgp
        self.old_predictor = None  # trick to make cloning work with sklearn (pipeline will be clones)

        steps = []

        dict_cfgp = namespace2dict(self.cfgp)
        if self.cfgp.is_scale:
            steps += [("scaler", instantiate(dict_cfgp["scaler"]))]

        if self.cfgp.is_preprocess:
            steps += [("preprocesser", instantiate(dict_cfgp["preprocesser"]))]

        if old_predictor is None:
            steps += [("model", instantiate(dict_cfgp["model"]))]
        else:
            # TODO : clean and add logistic regression CV instead
            logger.info("Using warm start for Sklearn. Currently this assumes that `C` was the only modified hyperparameter")
            old_predictor.C = dict_cfgp["model"]["C"]
            steps += [("model", old_predictor)]

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
            # Shape: [batch, *z_shape]
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
