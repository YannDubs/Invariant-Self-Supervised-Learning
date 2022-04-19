from __future__ import annotations

import logging
from typing import Any, Sequence, Union

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
import numpy as np
import seaborn as sns
import math

from torch import nn

from .helpers import (
    cont_tuple_to_tuple_cont,
    plot_config,
    prod,
    to_numpy,
    is_pow_of_k,
    check_import
)

try:
    import wandb
except ImportError:
    pass

try:
    import umap
except ImportError:
    pass

try:
    import pandas as pd
except ImportError:
    pass

logger = logging.getLogger(__name__)


def save_img(pl_module, trainer, img, name, caption):
    """Save an image on logger. Currently only  wandb."""
    experiment = trainer.logger.experiment
    if isinstance(trainer.logger, WandbLogger):
        wandb_img = wandb.Image(img, caption=caption)
        experiment.log({name: [wandb_img]}, commit=False)

    else:
        err = f"Plotting images is only available on  Wandb but you are using {type(trainer.logger)}."
        raise ValueError(err)


def is_plot(trainer, plot_interval):
    epoch = trainer.current_epoch
    if plot_interval == -1:
        # check if power of 2 (starting from 4)
        is_plot_interval = is_pow_of_k(int(epoch) + 2, k=2)
    else:
        is_plot_interval = (trainer.current_epoch + 1) % plot_interval == 0
    is_last_epoch = trainer.current_epoch == trainer.max_epochs - 1
    return is_plot_interval or is_last_epoch


class PlottingCallback(Callback):
    """Base classes for callbacks that plot.

    Parameters
    ----------
    plot_interval : int, optional
        Every how many epochs to plot. If -1 will plot every power of 3 starting from 3.

    plot_config_kwargs : dict, optional
            General config for plotting, e.g. arguments to matplotlib.rc, sns.plotting_context,
            matplotlib.set ...
    """

    def __init__(self, plot_interval: int = -1, plot_config_kwargs: dict = {}) -> None:
        super().__init__()
        self.plot_interval = plot_interval
        self.plot_config_kwargs = plot_config_kwargs

    # noinspection PyBroadException
    @rank_zero_only  # only plot on one machine
    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule,
    ) -> None:
        if is_plot(trainer, self.plot_interval):
            try:
                for fig, kwargs in self.yield_figs_kwargs(trainer, pl_module):
                    if "caption" not in kwargs:
                        kwargs["caption"] = f"ep: {trainer.current_epoch}"

                    save_img(pl_module, trainer, fig, **kwargs)

                    plt.close(fig)
            except:
                logger.exception(f"Couldn't plot for {type(PlottingCallback)}, error:")

    def yield_figs_kwargs(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        raise NotImplementedError()


class EffectiveDim(PlottingCallback):
    """Logs the (log of abs) eigenspectrum of the cross correlation matrix of Z, to check effective dimensionality.

    Parameters
    ----------
    z_shape : list or int
        Shape of z.
    """
    def __init__(self, z_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_corr_coef = 0
        self.val_corr_coef = 0
        self.n_train_steps = 0
        self.n_val_steps = 0

        z_dim = z_shape if isinstance(z_shape, int) else prod(z_shape)
        self.corr_coef_bn = torch.nn.BatchNorm1d(z_dim, affine=False)

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        Z = pl_module._save["train"]["Z"]
        corr_coef = (self.corr_coef_bn(Z).T @ self.corr_coef_bn(Z))
        self.train_corr_coef += corr_coef
        self.n_train_steps += 1

    def on_validation_batch_end(self, trainer, pl_module, *args, **kwargs):
        Z = pl_module._save["val"]["Z"]
        corr_coef = (self.corr_coef_bn(Z).T @ self.corr_coef_bn(Z))
        self.val_corr_coef += corr_coef
        self.n_val_steps += 1

    def on_train_epoch_end(self, trainer, pl_module, *args, **kwargs):
        corr_coef = self.train_corr_coef / self.n_train_steps

        try:
            rank = torch.linalg.matrix_rank(corr_coef, atol=1e-4, rtol=0.01, hermitian=True).float()
            pl_module.log(f"train/{pl_module.stage}/{pl_module.hparams.task}/rank", rank, on_epoch=True)
        except:
            logger.exception("could not compute rank:")

        super().on_train_epoch_end(trainer, pl_module, *args, **kwargs)

        self.n_train_steps = 0
        self.train_corr_coef = 0

    def on_validation_epoch_end(self, trainer, pl_module, *args, **kwargs):
        corr_coef = self.val_corr_coef / self.n_val_steps

        try:
            rank = torch.linalg.matrix_rank(corr_coef, atol=1e-4, rtol=0.01, hermitian=True).float()
            pl_module.log(f"train/{pl_module.stage}/{pl_module.hparams.task}/rank", rank, on_epoch=True)
        except:
            logger.exception("could not compute rank:")

        self.n_val_steps = 0
        self.val_corr_coef = 0

    def yield_figs_kwargs(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        with torch.no_grad():
            corr_coef = self.train_corr_coef / self.n_train_steps


            eigenv = torch.linalg.eigvalsh(corr_coef)
            eigenv = eigenv.abs().log().sort(descending=True)[0]

            with plot_config(**self.plot_config_kwargs):
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                ax.plot(eigenv)
                ax.set_xlabel("Singular value rank")
                ax.set_ylabel("Log of singular value")
                fig.tight_layout()

                yield fig, dict(name="Effective dim")


class RepresentationUMAP(PlottingCallback):
    """Plot the UMAP 2D plot of the representations, for different hyperparameters.

    Parameters
    ----------
    is_test: bool, optional
        Whether to plot representations for the test set rather than train.

    n_samples: int, optional
        Number of samples to plot from the dataset.

    n_labels: int, optional
        Number of labels to plot.

    n_neighbors: list of int, optional
        N neighbours parameters for UMAP. Should be one per UMAP plot. Low values will focus more on local structure,
        while high values on global structure.

    min_dists: list of int, optional
        Min dist parameters for UMAP. Should be one per UMAP plot.

    kwargs:
        Any additional arguments to `PlottingCallback`.
    """
    def __init__(
        self,
        dm : pl.LightningDataModule,
        is_test: bool = True,
        n_samples: int = 1000,
        n_labels: int = 10,
        n_neighbors: list[int] = [5,30,100],
        min_dists: list[int] = [0.05,0.1,0.5],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        check_import("umap", "RepresentationUMAP")
        check_import("pandas", "RepresentationUMAP")

        self.is_test = is_test
        self.n_samples = n_samples
        self.n_labels = n_labels
        self.n_neighbors = n_neighbors
        self.min_dists = min_dists

        dataset = dm.test_dataset if self.is_test else dm.train_dataset
        targets = dataset.cached_targets

        selected_Y = np.random.choice(np.unique(targets), size=self.n_labels, replace=False)
        mask = np.isin(targets,selected_Y)
        idcs_y = np.arange(len(targets))[mask]
        if len(idcs_y) < self.n_samples:
            logger.info(f"Plotting only only len(idcs)={len(idcs_y)}<{self.n_samples} for UMAP.")
            self.n_samples = len(idcs_y)
        idcs = np.random.choice(idcs_y, size=self.n_samples, replace=False)
        XY = [(dataset[i][0].cpu(),dataset[i][1][0]) for i in idcs]
        X,Y = cont_tuple_to_tuple_cont(XY)
        self.X = torch.stack(X)

        if hasattr(dataset, "idx_to_class"):
            self.y = [dataset.idx_to_class[y] for y in Y]
        else:
            self.y = Y

    def get_UMAP_embeddings(self, X, sup=None):
        return {f"UMAP K={k}, D={d} ": umap.UMAP(n_neighbors=k, min_dist=d).fit_transform(X, y=sup)
                for k, d in zip(self.n_neighbors, self.min_dists)}

    def yield_figs_kwargs(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        with torch.no_grad():
            pl_module.eval()
            Z = np.concatenate([to_numpy(pl_module(x.to(pl_module.device))) for x in self.X.split(512)], axis=0)
            Z_umaps = self.get_UMAP_embeddings(Z)

            name_plots = []
            for name, Z_umap in Z_umaps.items():
                with plot_config(**self.plot_config_kwargs, font_scale=2, is_ax_off=True):
                    # need correct label and color. maybe just use seaborn ?
                    df = pd.DataFrame({"x1": Z_umap[:,0], "x2": Z_umap[:,1], "Label": self.y})
                    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
                    sns.scatterplot(data=df, x="x1", y="x2", hue="Label", alpha=0.7, ax=ax)
                    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    name_plots.append((name, fig))

        pl_module.train()

        for name, fig in name_plots:
            yield fig, dict(name=name)

class Freezer(BaseFinetuning):
    """Freeze entire model.

    Parameters
    ----------
    model_name : string
        Name of the module to freeze from pl module. Can use dots.
    """

    def __init__(
        self, model_name,
    ):
        super().__init__()
        self.model_name = model_name.split(".")

    def get_model(self, pl_module):
        model = pl_module

        for model_name in self.model_name:
            model = getattr(model, model_name)

        return model

    def freeze_before_training(self, pl_module):
        model = self.get_model(pl_module)
        self.freeze(modules=model, train_bn=False)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        pass

# modified from https://github.com/PyTorchLightning/lightning-bolts/blob/ad771c615284816ecadad11f3172459afdef28e3/pl_bolts/callbacks/byol_updates.py
class MAWeightUpdate(pl.Callback):
    """EMA Weight update rule for DINO.

    Notes
    -----
    - BYOL/DINO/... claims this keeps the online_network from collapsing.
    - Automatically increases tau from ``initial_tau`` to 1.0 with every training step
    """

    def __init__(self, initial_tau: float = 0.996):
        """
        Args:
            initial_tau: starting tau. Auto-updates with every training step
        """
        super().__init__()
        self.initial_tau = initial_tau
        self.current_tau = initial_tau

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: pl.LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
    ) -> None:

        # get networks
        student_enc = pl_module.encoder
        teacher_enc = pl_module.loss_decodability.teacher_encoder

        student_proj = pl_module.loss_decodability.projector
        teacher_proj = pl_module.loss_decodability.teacher_proj

        # update weights
        with torch.no_grad():  # not needed
            self.update_weights(student_enc, teacher_enc)
            self.update_weights(student_proj, teacher_proj)

        # update tau after
        self.current_tau = self.update_tau(pl_module, trainer)

    def update_tau(self, pl_module: pl.LightningModule, trainer: pl.Trainer) -> float:
        max_steps = len(trainer.train_dataloader) * trainer.max_epochs
        tau = (
            1
            - (1 - self.initial_tau)
            * (math.cos(math.pi * pl_module.global_step / max_steps) + 1)
            / 2
        )
        return tau

    def update_weights(
        self,
        online_net: Union[nn.Module, torch.Tensor],
        target_net: Union[nn.Module, torch.Tensor],
    ) -> None:
        # apply MA weight update
        for (name, online_p), (_, target_p) in zip(
            online_net.named_parameters(), target_net.named_parameters(),
        ):
            target_p.data = (
                self.current_tau * target_p.data
                + (1 - self.current_tau) * online_p.detach().data
            )
