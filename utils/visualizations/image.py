import random

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from issl.helpers import set_seed, tmp_seed, to_numpy

DFLT_FIGSIZE = (17, 9)

__all__ = ["plot_dataset_samples_imgs"]


def plot_dataset_samples_imgs(
    dataset, n_plots=4, figsize=DFLT_FIGSIZE, ax=None, pad_value=1, seed=123, title=None
):
    """Plot `n_samples` samples of the a datset."""
    set_seed(seed)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    img_tensor = torch.stack(
        [dataset[random.randint(0, len(dataset) - 1)][0] for _ in range(n_plots)], dim=0
    )
    grid = make_grid(img_tensor, nrow=2, pad_value=pad_value)

    ax.imshow(to_numpy(grid.permute(1, 2, 0)))
    ax.axis("off")

    if title is not None:
        ax.set_title(title)


def plot_dataset_samples_X_A(
    dataset,
    n_plots=3,
    figsize=DFLT_FIGSIZE,
    ax=None,
    pad_value=1,
    seed_img=123,
    seed_trnsf=123,
    title=None,
):
    """Plot `n_samples` samples of the a datset."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    with tmp_seed(seed_img):
        idcs = [random.randint(0, len(dataset) - 1) for _ in range(n_plots)]

    with tmp_seed(seed_trnsf):
        Xs = [dataset[i][0] for i in idcs]
        As = [dataset[i][1][1] for i in idcs]

    img_tensor = torch.stack(Xs + As, dim=0)
    grid = make_grid(img_tensor, nrow=n_plots, pad_value=pad_value)

    ax.imshow(to_numpy(grid.permute(1, 2, 0)))
    ax.axis("off")

    if title is not None:
        ax.set_title(title)
