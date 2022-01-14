from __future__ import annotations

import logging
from typing import Optional

import einops
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
import numpy as np
import seaborn as sns

from .helpers import (
    UnNormalizer,
    cont_tuple_to_tuple_cont, is_colored_img,
    plot_config,
    tensors_to_fig,
    to_numpy,
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
    is_plot_interval = (trainer.current_epoch + 1) % plot_interval == 0
    is_last_epoch = trainer.current_epoch == trainer.max_epochs - 1
    return is_plot_interval or is_last_epoch


class PlottingCallback(Callback):
    """Base classes for callbacks that plot.

    Parameters
    ----------
    plot_interval : int, optional
        Every how many epochs to plot.

    plot_config_kwargs : dict, optional
            General config for plotting, e.g. arguments to matplotlib.rc, sns.plotting_context,
            matplotlib.set ...
    """

    def __init__(self, plot_interval: int = 10, plot_config_kwargs: dict = {}) -> None:
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


class ReconstructImages(PlottingCallback):
    """Logs some reconstructed images.

    Notes
    -----
    - the model should return a dictionary after each training step, containing
    a tensor "Y_hat" and a tensor "Y" both of image shape.
    - this will log one reconstructed image (+real) after each training epoch.
    """

    def yield_figs_kwargs(self, trainer: pl.Trainer, pl_module: pl.LightningModule):

        cfg = pl_module.hparams
        #! waiting for torch lightning #1243
        a_hat = pl_module._save["A_hat"].float()
        a = pl_module._save["A"].float()
        x = pl_module._save["X"].float()

        if is_colored_img(x):
            if cfg.data.normalized:
                # undo normalization for plotting
                unnormalizer = UnNormalizer(cfg.data.normalized)
                x = unnormalizer(x)
                a = unnormalizer(a)

        yield a_hat, dict(name="rec_img")

        yield a, dict(name="trgt_img")

        yield x, dict(name="input_img")

class ReconstructMx(PlottingCallback):
    """Reconstruct the estimated M(X).

    Notes
    -----
    - The model should have attribute `f_ZhatlM` and `suff_stat_AlZhat`.
    """
    def yield_figs_kwargs(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        n_Mx = pl_module.loss_decodability.predecode_n_Mx

        with torch.no_grad():
            pl_module.eval()
            with plot_config(**self.plot_config_kwargs, font_scale=2):
                Ms = torch.eye(n_Mx, device=pl_module.device)
                Zhat = pl_module.loss_decodability.f_ZhatlM(Ms)
                img = pl_module.loss_decodability.suff_stat_AlZhat(Zhat)
                img = torch.sigmoid(img) # put back on [0,1]

                fig = tensors_to_fig(img,n_cols=10)

        pl_module.train()

        yield fig, dict(name="rec_Mx")

class LatentDimInterpolator(PlottingCallback):
    """Logs interpolated images.

    Parameters
    ----------
    z_dim : int 
        Number of dimensions for latents.

    range_start : float, optional
        Start of the interpolating range.

    range_end : float, optional
        End of the interpolating range.

    n_per_lat : int, optional
        Number of traversal to do for each latent.

    n_lat_traverse : int, optional
        Number of latent to traverse for traversal 1_d. Max is `z_dim`.

    kwargs :
        Additional arguments to PlottingCallback.
    """

    def __init__(
        self,
        z_dim: int,
        range_start: float = -5,
        range_end: float = 5,
        n_per_lat: int = 7,
        n_lat_traverse: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.z_dim = z_dim
        self.range_start = range_start
        self.range_end = range_end
        self.n_per_lat = n_per_lat
        self.n_lat_traverse = n_lat_traverse

    def yield_figs_kwargs(self, trainer: pl.Trainer, pl_module: pl.LightningModule):

        with torch.no_grad():
            pl_module.eval()
            with plot_config(**self.plot_config_kwargs, font_scale=2):
                traversals_2d = self.latent_traverse_2d(pl_module)

            with plot_config(**self.plot_config_kwargs, font_scale=1.5):
                traversals_1d = self.latent_traverse_1d(pl_module)

        pl_module.train()

        yield traversals_2d, dict(name="traversals_2d")
        yield traversals_1d, dict(name="traversals_1d")

    def _traverse_line(
        self, idx: int, pl_module: pl.LightningModule, z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return a (size, latent_size) latent sample, corresponding to a traversal
        of a latent variable indicated by idx."""

        if z is None:
            z = torch.zeros(1, self.n_per_lat, self.z_dim, device=pl_module.device)

        traversals = torch.linspace(
            self.range_start,
            self.range_end,
            steps=self.n_per_lat,
            device=pl_module.device,
        )
        for i in range(self.n_per_lat):
            z[:, i, idx] = traversals[i]

        z = einops.rearrange(z, "r c ... -> (r c) ...")
        img = pl_module.loss_decodability.suff_stat_AlZ(z)

        # put back to [0,1]
        img = torch.sigmoid(img)
        return img

    def latent_traverse_2d(self, pl_module: pl.LightningModule) -> plt.Figure:
        """Traverses the first 2 latents TOGETHER."""
        traversals = torch.linspace(
            self.range_start,
            self.range_end,
            steps=self.n_per_lat,
            device=pl_module.device,
        )
        z_2d = torch.zeros(
            self.n_per_lat, self.n_per_lat, self.z_dim, device=pl_module.device
        )
        for i in range(self.n_per_lat):
            z_2d[i, :, 0] = traversals[i]  # fill first latent

        imgs = self._traverse_line(1, pl_module, z=z_2d)  # fill 2nd latent and rec.
        fig = tensors_to_fig(
            imgs,
            n_cols=self.n_per_lat,
            x_labels=["1st Latent"],
            y_labels=["2nd Latent"],
        )

        return fig

    def latent_traverse_1d(self, pl_module: pl.LightningModule) -> plt.Figure:
        """Traverses the first `self.n_lat` latents separately."""
        n_lat_traverse = min(self.n_lat_traverse, self.z_dim)
        imgs = [self._traverse_line(i, pl_module) for i in range(n_lat_traverse)]
        imgs = torch.cat(imgs, dim=0)
        fig = tensors_to_fig(
            imgs,
            n_cols=self.n_per_lat,
            x_labels=["Sweeps"],
            y_labels=[f"Lat. {i}" for i in range(n_lat_traverse)],
        )
        return fig

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
        n_samples: int = 5000,
        n_labels: int = 5,
        n_neighbors: list[int] = [5,30,100],
        min_dists: list[int] = [0.05,0.1,0.5],
        plot_interval: int = 50,
        cmap: str ='tab10',
        **kwargs,
    ) -> None:
        super().__init__(**kwargs, plot_interval=plot_interval)

        check_import("umap", "RepresentationUMAP")
        check_import("pandas", "RepresentationUMAP")

        self.is_test = is_test
        self.n_samples = n_samples
        self.n_labels = n_labels
        self.n_neighbors = n_neighbors
        self.min_dists = min_dists
        self.cmap = cmap

        dataset = dm.test_dataset if self.is_test else dm.train_dataset
        targets = to_numpy(dataset.get_targets())
        selected_Y = np.random.choice(np.unique(targets), size=self.n_labels, replace=False)
        mask = np.isin(targets,selected_Y)
        idcs_y = np.arange(len(targets))[mask]
        if len(idcs_y) < self.n_samples:
            logger.info(f"Plotting only only len(idcs)={len(idcs_y)}<{self.n_samples} for UMAP.")
            self.n_samples = len(idcs_y)
        idcs = np.random.choice(idcs_y, size=self.n_samples, replace=False)
        Xy = [(dataset[i][0].cpu(),dataset[i][1][0]) for i in idcs]
        X,y = cont_tuple_to_tuple_cont(Xy)
        self.X = torch.stack(X)

        if hasattr(dataset, "idx_to_class"):
            self.y = [dataset.idx_to_class[y] for y in y]
        else:
            self.y = y

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
                    sns.scatterplot(data=df, x="x1", y="x2", hue="Label", alpha=0.7, ax=ax) #palette=self.cmap
                    plt.legend(bbox_to_anchor=(1.25, 1), borderaxespad=0)
                    name_plots.append((name, fig))

        pl_module.train()

        for name, fig in name_plots:
            yield fig, dict(name=name)


def plot_embeddings(embeddings, labels, label_names, cmap='tab10', title=None):

    fig, ax = plt.subplots(1, len(embeddings), figsize=(26, 6))
    if title is not None:
        fig.suptitle(title, fontsize=18);
    plt.setp(ax, xticks=[], yticks=[]);

    for i, (k, embedding) in enumerate(embeddings.items()):
        im = ax[i].scatter(*embedding.T, s=0.3, c=labels, cmap=cmap, alpha=1.0)
        ax[i].set_title(k, fontsize=14)

    cbaxes = fig.add_axes([0.12, 0.05, 0.78, 0.05])
    cbar = fig.colorbar(im, boundaries=np.arange(num_labels + 1) - 0.5, orientation="horizontal", cax=cbaxes)
    cbar.set_ticks(np.arange(num_labels))
    cbar.set_ticklabels(label_names)
    plt.show()

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
