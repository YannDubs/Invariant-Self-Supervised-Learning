from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Optional
import logging

import pytorch_lightning as pl
import torch
from issl.distributions import CondDist
from issl.helpers import Annealer, OrderedSet, append_optimizer_scheduler_, prod, rel_distance
from issl.losses import get_loss_decodability, get_regularizer
from issl.predictors import OnlineEvaluator
from torch.nn import functional as F

__all__ = ["ISSLModule"]
logger = logging.getLogger(__name__)

class ISSLModule(pl.LightningModule):
    """Main module for invariant SSL module."""

    def __init__(self, hparams: Any) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)

        self.p_ZlX = CondDist(**self.hparams.encoder.kwargs)
        self.loss_decodability = get_loss_decodability(p_ZlX=self.p_ZlX,
            **self.hparams.decodability.kwargs
        )
        self.loss_regularizer = get_regularizer(**self.hparams.regularizer.kwargs)

        if self.hparams.evaluation.representor.is_online_eval:
            # replaces pl_bolts.callbacks.SSLOnlineEvaluator because training
            # as a callback was not well support by lightning
            self.evaluator = OnlineEvaluator(**self.hparams.online_evaluator.kwargs)

        self.stage = self.hparams.stage  # allow changing to stages

        # input example to get shapes for summary
        self.example_input_array = torch.randn(10, *self.hparams.data.shape).sigmoid()

        self._save= dict()


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
        self, x: torch.Tensor, is_sample: bool = False, is_return_p_ZlX: bool = False, p_ZlX=None, rm_out_chan: bool = False,
    ):
        """Represents the data `x`.

        Parameters
        ----------
        x : torch.Tensor of shape=[batch_size, *data.shape]
            Data to represent.

        is_sample : bool, optional
            Whether to sample the representation rather than return expected representation.

        is_return_p_ZlX : bool, optional
            Whether to return the encoder in addition to the representation.

        p_ZlX : CondDist, optional
            Encoder to use instead of the default `self.p_ZlX`. Useful to use an EMA encoder.

        Returns
        -------
        z : torch.Tensor of shape=[batch_size, z_dim]
            Represented data.
        """

        if p_ZlX is None:
            p_ZlX = self.p_ZlX

        # batch shape: [batch_size, *z_shape[:-1]] ; event shape: [z_dim]
        if rm_out_chan:
            p_Zlx = p_ZlX(x, rm_out_chan=True)
        else:
            p_Zlx = p_ZlX(x)

        # shape: [batch_size, *z_shape]
        if is_sample:
            z = p_Zlx.rsample()
        else:
            z = p_Zlx.mean

        if is_return_p_ZlX:
            return z, p_Zlx

        return z

    def step(self, batch: list) -> tuple[torch.Tensor, dict, dict]:

        x, (_, aux_target) = batch

        if self.hparams.representor.is_switch_x_aux_trgt and self.stage == "repr":
            # switch x and aux_target. Useful if want augmentations as inputs. Only during representations.
            x, aux_target = aux_target, x

        # z shape: [batch_size, *z_shape]
        # p_Zlx batch shape: [batch_size, *z_shape[:-1]] ; event shape: [z_dim]
        z, p_Zlx = self(x, is_sample=True, is_return_p_ZlX=True)

        try:
            breakpoint()
            # TODO if keeping rm_out_chan then for regularizer will need to give the augmentation after out chan
            z_a = self(aux_target, is_sample=False, rm_out_chan=self.hparams.encoder.rm_out_chan_aug)
        except:
            z_a = None

        beta = self.hparams.representor.loss.beta
        if self.loss_regularizer is not None or math.isclose(beta, 0):
            # `sample_eff` is proxy to ensure supp(p(Z|M(X))) = supp(p(Z|X)). shape: [batch_size]
            regularize, s_logs, s_other = self.loss_regularizer(
                z, z_a, aux_target, p_Zlx, self
            )
        else:
            regularize, s_logs, s_other = None, dict(), dict()

        # `decodability` is proxy for R_f[A|Z]. shape: [batch_size]
        decodability, d_logs, d_other = self.loss_decodability(z, z_a, aux_target,  x, self)
        loss, logs, other = self.loss(decodability, regularize)

        # to log (dict)
        logs.update(d_logs)
        logs.update(s_logs)
        logs["z_norm_l2"] = z.norm(dim=1, p=2).mean()
        logs["z_max"] = z.abs().max(dim=-1)[0].mean()
        logs["z_norm_l1"] = z.norm(dim=1, p=1).mean()

        if z_a is not None:
            logs["rel_distance"] = rel_distance(z.detach(), z_a.detach()).mean()  # estimate neural collapse

        # any additional information that can be useful (dict)
        other.update(d_other)
        other.update(s_other)
        other["X"] = x[0].detach().float().cpu()
        other["Z"] = z.detach().float().cpu()

        return loss, logs, other

    def loss(
        self, decodability: torch.Tensor, regularize: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, dict, dict]:

        # E_x[...]. shape: shape: []
        # ensure that using float32 because small / large beta can cause issues in half precision
        decodability = decodability.float().mean(0)
        regularize = regularize.float().mean(0) if regularize is not None else 0.0
        loss = decodability + self.hparams.representor.loss.beta * regularize

        logs = dict(loss=loss, decodability=decodability, regularize=regularize)

        other = dict()

        return loss, logs, other

    def training_step(
        self, batch: torch.Tensor, batch_idx: int, optimizer_idx: int = 0
    ) -> Optional[torch.Tensor]:

        curr_opt = self.idcs_to_opt[optimizer_idx]

        # MODEL
        if curr_opt == "issl":

            loss, logs, other = self.step(batch)

            # Imp: waiting for torch lightning #1243
            self._save["train"] = other

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
        loss, logs, other = self.step(batch)

        self._save[step] = other

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
            for name, p in dec.to_freeze.named_parameters():
                p.grad = None