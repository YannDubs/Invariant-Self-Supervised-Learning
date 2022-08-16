from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Optional
import logging

import pytorch_lightning as pl
import torch

from issl import get_Architecture
from issl.helpers import DistToEtf, OrderedSet, append_optimizer_scheduler_
from issl.losses import get_loss_decodability
from issl.predictors import OnlineEvaluator

__all__ = ["ISSLModule"]
logger = logging.getLogger(__name__)

class ISSLModule(pl.LightningModule):
    """Main module for invariant SSL module."""

    def __init__(self, hparams: Any) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)

        cfge = self.hparams.encoder.kwargs
        Architecture = get_Architecture(cfge.architecture, **cfge.arch_kwargs)
        self.encoder = Architecture(cfge.in_shape, cfge.out_shape)

        self.loss_decodability = get_loss_decodability(encoder=self.encoder,
            **self.hparams.decodability.kwargs
        )

        if self.hparams.evaluation.representor.is_online_eval:
            # replaces pl_bolts.callbacks.SSLOnlineEvaluator because training
            # as a callback was not well support by lightning
            self.evaluator = OnlineEvaluator(**self.hparams.online_evaluator.kwargs)

        self.stage = self.hparams.stage  # allow changing to stages

        # input example to get shapes for summary
        self.example_input_array = torch.randn(10, *self.hparams.data.shape).sigmoid()

        self.dist_to_etf = DistToEtf(z_dim=self.hparams.encoder.z_dim)

    def load_state_dict(self, state_dict, *args, **kwargs):
        # TODO rm this is just for backward compatibility
        new_state_dict = {k.replace("p_ZlX.mapper","encoder"): v for k,v in state_dict.items()}
        try:
            super().load_state_dict(new_state_dict, *args, **kwargs)
        except:
            logger.exception("Using strict=False after:")
            super().load_state_dict(new_state_dict, strict=False)


    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """
        Predict function, this will represent the data and also return the correct label.
        Which is useful in case you want to create a represented dataset.
        """
        x, y = batch

        if isinstance(y, Sequence):
            y = torch.stack([y[0]] + y[1], dim=-1)

        return self(x).cpu(), y.cpu()

    def forward(
        self, x: torch.Tensor, encoder=None,  **kwargs
    ):
        """Represents the data `x`.

        Parameters
        ----------
        x : torch.Tensor of shape=[batch_size, *data.shape]
            Data to represent.

        encoder : Module, optional
            Encoder to use instead of the default `self.encoder`. Useful to use an EMA encoder.

        Returns
        -------
        z : torch.Tensor of shape=[batch_size, z_dim]
            Represented data.
        """

        if encoder is None:
            encoder = self.encoder

        # shape: [batch_size, *z_dim]
        z = encoder(x, **kwargs)

        return z

    def step(self, batch: list) -> tuple[torch.Tensor, dict]:

        x, (_, aux_target) = batch

        z = self(x)  # only encode the input (ie if asymmetry between X and A)
        z_tgt = None

        loss, logs = self.loss_decodability(z, z_tgt, x, aux_target, self)

        # to log (dict)
        logs["loss"] = loss
        pos_loss, neg_loss = self.dist_to_etf(*z.detach().chunk(2, dim=0))
        logs["etf_pos"] = pos_loss
        logs["etf_neg"] = neg_loss
        logs["dist_to_etf"] = pos_loss + neg_loss

        return loss, logs

    def training_step(
        self, batch: torch.Tensor, batch_idx: int, optimizer_idx: int = 0
    ) -> Optional[torch.Tensor]:

        curr_opt = self.idcs_to_opt[optimizer_idx]

        # MODEL
        if curr_opt == "issl":
            loss, logs = self.step(batch)

        # ONLINE EVALUATOR
        elif curr_opt == "evaluator":
            loss, logs = self.evaluator(batch, self)

        else:
            raise ValueError(f"Unknown curr_opt={curr_opt}.")

        self.log_dict(
            {
                f"train/{self.stage}/{self.hparams.task}/{k}": v
                for k, v in logs.items()
            },
            sync_dist=True,
        )
        return loss

    def test_val_step(
        self, batch: torch.Tensor, batch_idx: int, step: str
    ) -> Optional[torch.Tensor]:
        loss, logs = self.step(batch)

        if self.hparams.evaluation.representor.is_online_eval:
            _, online_logs = self.evaluator(batch, self)
            logs.update(online_logs)

        self.log_dict(
            {
                f"{step}/{self.stage}/{self.hparams.task}/{k}": v
                for k, v in logs.items()
            },
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        return self.test_val_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.test_val_step(batch, batch_idx, "test")

    def get_specific_parameters(self, mode: str) -> OrderedSet:
        """Returns an iterator over the desired model parameters."""
        all_param = OrderedSet(self.parameters())

        if self.hparams.evaluation.representor.is_online_eval:
            eval_param = OrderedSet(self.evaluator.parameters())
        else:
            eval_param = OrderedSet([])

        issl_param = OrderedSet(all_param - eval_param)
        if mode == "all":
            return all_param
        elif mode == "evaluator":
            return eval_param
        elif mode == "issl":
            return issl_param
        else:
            raise ValueError(f"Unknown parameter mode={mode}.")

    def configure_optimizers(self) -> tuple[list[Any], list[Any]]:

        self.idcs_to_opt = {}
        optimizers, schedulers = [], []
        n_opt = 0

        # Encoder OPTIMIZER
        self.idcs_to_opt[n_opt] = "issl"
        n_opt += 1
        append_optimizer_scheduler_(
            self.hparams.optimizer_issl,
            self.hparams.scheduler_issl,
            self.get_specific_parameters("issl"),
            optimizers,
            schedulers,
            name="lr_issl",
        )

        # ONLINE EVALUATOR
        if self.hparams.evaluation.representor.is_online_eval:
            self.idcs_to_opt[n_opt] = "evaluator"
            n_opt += 1
            append_optimizer_scheduler_(
                self.hparams.optimizer_eval,
                self.hparams.scheduler_eval,
                self.get_specific_parameters("evaluator"),
                optimizers,
                schedulers,
                name="lr_evaluator",
            )

        return optimizers, schedulers

    def set_represent_mode_(self):
        """Set as a representor."""

        # this ensures that nothing is persistent, i.e. will not be saved in checkpoint when
        # part of predictor
        for model in self.modules():
            params = dict(model.named_parameters(recurse=False))
            buffers = dict(model.named_buffers(recurse=False))
            for name, param in params.items():
                del model._parameters[name]
                model.register_buffer(name, param.data, persistent=False)

            for name, param in buffers.items():
                del model._buffers[name]
                model.register_buffer(name, param, persistent=False)

        self.freeze()
        self.eval()

    def on_after_backward(self):
        dec = self.loss_decodability
        if hasattr(dec, "to_freeze") and self.current_epoch < dec.freeze_Mx_epochs:
            for to_freeze in dec.to_freeze:
                for name, p in to_freeze.named_parameters():
                    p.grad = None