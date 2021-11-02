from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Optional

import pytorch_lightning as pl
import torch
from issl.architectures import get_Architecture
from issl.distributions import CondDist
from issl.helpers import Annealer, OrderedSet, append_optimizer_scheduler_
from issl.losses import get_loss_decodability, get_regularizer
from issl.losses.decodability import ClusterSelfDistillationISSL
from issl.predictors import OnlineEvaluator
from torch.nn import functional as F

__all__ = ["ISSLModule"]


class ISSLModule(pl.LightningModule):
    """Main module for invariant SSL module."""

    def __init__(self, hparams: Any) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)

        self.p_ZlX = CondDist(**self.hparams.encoder.kwargs)
        self.Z_processor = get_Architecture(**self.hparams.Z_processor.kwargs)()
        self.loss_decodability = get_loss_decodability(
            **self.hparams.decodability.kwargs
        )
        self.loss_regularizer = get_regularizer(**self.hparams.regularizer.kwargs)

        self.beta_annealer = self.get_beta_annealer()

        if self.hparams.evaluation.representor.is_online_eval:
            # replaces pl_bolts.callbacks.SSLOnlineEvaluator because training
            # as a callback was not well support by lightning
            self.evaluator = OnlineEvaluator(**self.hparams.online_evaluator.kwargs)

        self.stage = self.hparams.stage  # allow changing to stages

    @property
    def final_beta(self):
        """Return the final beta to use."""
        cfg = self.hparams
        beta = (
            cfg.representor.loss.beta
            * cfg.regularizer.factor_beta
            * cfg.decodability.factor_beta
        )
        return beta

    def get_beta_annealer(self) -> Annealer:
        """Return the annealer for the weight of the regularizer."""
        cfg = self.hparams
        beta_annealer = Annealer(
            self.final_beta * 1e-5,  # don't use 0 in case geometric
            self.final_beta,
            n_steps_anneal=math.ceil(1 / 10 * cfg.data.max_steps),  # arbitrarily 1/10th
            mode=cfg.representor.loss.beta_anneal,
        )
        return beta_annealer

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
        self, x: torch.Tensor, is_sample: bool = False, is_process_Z: bool = True
    ) -> torch.Tensor:
        """Represents the data `x`.

        Parameters
        ----------
        x : torch.Tensor of shape=[batch_size, *data.shape]
            Data to represent.

        is_sample : bool, optional
            Whether to sample the representation rather than return expected representation.

        is_process_Z : bool, optional
            Whether to use post processing of the representation.

        Returns
        -------
        z : torch.Tensor of shape=[batch_size, z_dim]
            Represented data.
        """
        # batch shape: [batch_size, *z_shape[:-1]] ; event shape: [z_dim]
        p_Zlx = self.p_ZlX(x)

        # shape: [batch_size, *z_shape]
        if is_sample:
            z = p_Zlx.rsample()
        else:
            z = p_Zlx.mean

        if is_process_Z:
            # shape: [batch_size, *z_shape]
            z = self.Z_processor(z)

        if self.hparams.encoder.is_normalize_Z:
            z = F.normalize(z, dim=1, p=2)

        return z

    def step(self, batch: list) -> tuple[torch.Tensor, dict, dict]:

        x, (_, aux_target) = batch

        if self.hparams.representor.is_switch_x_aux_trgt and self.stage == "repr":
            # switch x and aux_target. Useful if want augmentations as inputs. Only turing representations.
            x, aux_target = aux_target, x

        # batch shape: [batch_size, *z_shape[:-1]] ; event shape: [z_dim]
        p_Zlx = self.p_ZlX(x)

        # shape: [batch_size, *z_shape]
        z = p_Zlx.rsample()

        # shape: [batch_size, *z_shape]
        z = self.Z_processor(z)

        if self.hparams.encoder.is_normalize_Z:
            z = F.normalize(z, dim=1, p=2)

        if self.loss_regularizer is not None:
            # `sample_eff` is proxy to ensure supp(p(Z|M(X))) = supp(p(Z|X)). shape: [batch_size]
            regularize, s_logs, s_other = self.loss_regularizer(
                z, aux_target, p_Zlx, self
            )
        else:
            regularize, s_logs, s_other = None, dict(), dict()

        # `decodability` is proxy for R_f[A|Z]. shape: [batch_size]
        decodability, d_logs, d_other = self.loss_decodability(z, aux_target, self)
        loss, logs, other = self.loss(decodability, regularize)

        # to log (dict)
        logs.update(d_logs)
        logs.update(s_logs)
        logs["z_norm_l2"] = F.normalize(z, dim=1, p=2).mean()
        logs["z_max"] = z.abs().max(dim=-1)[0].mean()
        logs["z_norm_l1"] = F.normalize(z, dim=1, p=1).mean()

        # any additional information that can be useful (dict)
        other.update(d_other)
        other.update(s_other)
        other["X"] = x[0].detach().cpu()

        return loss, logs, other

    def loss(
        self, decodability: torch.Tensor, regularize: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, dict, dict]:

        curr_beta = self.beta_annealer(n_update_calls=self.global_step)

        # E_x[...]. shape: shape: []
        # ensure that using float32 because small / large beta can cause issues in half precision
        decodability = decodability.float().mean(0)

        if regularize is not None:
            regularize = regularize.float().mean(0)

            # use actual (annealed) beta for the gradients, but still want the loss to be in terms of
            # final beta for plotting and checkpointing => use trick
            beta_sample_eff = curr_beta * regularize  # actual gradients
            beta_sample_eff = (
                beta_sample_eff
                - beta_sample_eff.detach()
                + (self.final_beta * regularize.detach())
            )
        else:
            beta_sample_eff = 0
            regularize = 0

        loss = decodability + beta_sample_eff

        logs = dict(
            loss=loss, decodability=decodability, regularize=regularize, beta=curr_beta,
        )

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
            # Dev: might not be needed if no save
            self._save = other

        # ONLINE EVALUATOR
        elif curr_opt == "evaluator":
            loss, logs = self.evaluator(batch, self)

        else:
            raise ValueError(f"Unknown curr_opt={curr_opt}.")

        self.log_dict(
            {
                f"train/{self.stage}/{self.hparams.data.name}/{k}": v
                for k, v in logs.items()
            },
            sync_dist=True,
        )
        return loss

    def test_val_step(
        self, batch: torch.Tensor, batch_idx: int, step: str
    ) -> Optional[torch.Tensor]:
        # TODO for some reason validation step for wandb logging after resetting is not correct
        loss, logs, _ = self.step(batch)

        if self.hparams.evaluation.representor.is_online_eval:
            _, online_logs = self.evaluator(batch, self)
            logs.update(online_logs)

        self.log_dict(
            {
                f"{step}/{self.stage}/{self.hparams.data.name}/{k}": v
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
                self.evaluator.parameters(),
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
        is_cluster_slfdstl = isinstance(dec, ClusterSelfDistillationISSL)
        if is_cluster_slfdstl and self.current_epoch < dec.freeze_Mx_epochs:
            for name, p in dec.named_parameters():
                if "Mx_logits" in name:
                    p.grad = None
