from __future__ import annotations

import contextlib
import math
import numbers
import operator
import random
import sys
import warnings
from argparse import Namespace
from collections.abc import MutableMapping, MutableSet, Sequence
from functools import reduce
from queue import Queue
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.distributed as dist
from matplotlib.cbook import MatplotlibDeprecationWarning
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence
from torchvision import transforms as transform_lib


def namespace2dict(namespace):
    """
    Converts recursively namespace to dictionary. Does not work if there is a namespace whose
    parent is not a namespace.
    """
    d = dict(**namespace)
    for k, v in d.items():
        if isinstance(v, NamespaceMap):
            d[k] = namespace2dict(v)
    return d


class NamespaceMap(Namespace, MutableMapping):
    """Namespace that can act like a dict."""

    def __init__(self, d):
        # has to take a single argument as input instead of a dictionary as namespace usually do
        # because from pytorch_lightning.utilities.apply_func import apply_to_collection doesn't work
        # with namespace (even though they think it does)
        super().__init__(**d)

    def select(self, k):
        """Allows selection using `.` in string."""
        to_return = self
        for subk in k.split("."):
            to_return = to_return[subk]
        return to_return

    def __getitem__(self, k):
        return self.__dict__[k]

    def __setitem__(self, k, v):
        self.__dict__[k] = v

    def __delitem__(self, k):
        del self.__dict__[k]

    def __len__(self):
        return len(self.__dict__)

    def __iter__(self):
        return iter(self.__dict__)


def replicate_shape(shape: Sequence[int], n_rep: int) -> Sequence[int]:
    """Replicate the last dimension of a shape `n_rep` times."""
    T = type(shape)
    out_tuple = tuple(shape[:-1]) + (shape[-1] * n_rep,)
    return T(out_tuple)


def check_import(module: str, to_use: Optional[str] = None):
    """Check whether the given module is imported."""
    if module not in sys.modules:
        if to_use is None:
            error = '{} module not imported. Try "pip install {}".'.format(
                module, module
            )
            raise ImportError(error)
        else:
            error = 'You need {} to use {}. Try "pip install {}".'.format(
                module, to_use, module
            )
            raise ImportError(error)


# modified from https://github.com/skorch-dev/skorch/blob/92ae54b/skorch/utils.py#L106
def to_numpy(X) -> np.array:
    """Convert tensors,list,tuples,dataframes to numpy arrays."""
    if isinstance(X, np.ndarray):
        return X

    # the sklearn way of determining pandas dataframe
    if hasattr(X, "iloc"):
        return X.values

    if isinstance(X, (tuple, list, numbers.Number)):
        return np.array(X)

    if not isinstance(X, (torch.Tensor, PackedSequence)):
        raise TypeError(f"Cannot convert {type(X)} to a numpy array.")

    if X.is_cuda:
        X = X.cpu()

    if X.requires_grad:
        X = X.detach()

    return X.numpy()


def set_seed(seed: Optional[int]) -> None:
    """Set the random seed."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)


@contextlib.contextmanager
def tmp_seed(seed: Optional[int], is_cuda: bool = torch.cuda.is_available()):
    """Context manager to use a temporary random seed with `with` statement."""
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    random_state = random.getstate()
    if is_cuda:
        torch_cuda_state = torch.cuda.get_rng_state()

    set_seed(seed)
    try:
        yield
    finally:
        if seed is not None:
            # if seed is None do as if no tmp_seed
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
            random.setstate(random_state)
            if is_cuda:
                torch.cuda.set_rng_state(torch_cuda_state)


def weights_init(module: nn.Module, nonlinearity: str = "relu") -> None:
    """Initialize a module and all its descendents.

    Parameters
    ----------
    module : nn.Module
       module to initialize.

    nonlinearity : str, optional
        Name of the nn.functional activation. Used for initialization.
    """
    # loop over direct children (not grand children)
    for m in module.children():

        # all standard layers
        if isinstance(m, torch.nn.modules.conv._ConvNd):
            # used in https://github.com/brain-research/realistic-ssl-evaluation/
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=nonlinearity)
            try:
                nn.init.zeros_(m.bias)
            except AttributeError:
                pass

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)
            try:
                nn.init.zeros_(m.bias)
            except AttributeError:
                pass

        elif isinstance(m, nn.BatchNorm2d):
            try:
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            except AttributeError:
                # if affine = False
                pass

        elif hasattr(m, "reset_parameters"):
            # if has a specific reset
            # Imp: don't go in grand children because you might have specific weights you don't want to reset
            m.reset_parameters()

        else:
            weights_init(m, nonlinearity=nonlinearity)  # go to grand children


def batch_flatten(x: torch.Tensor) -> tuple[torch.Tensor, Sequence[int]]:
    """Batch wise flattening of an array."""
    shape = x.shape
    return x.reshape(-1, shape[-1]), shape


def batch_unflatten(x: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """Revert `batch_flatten`."""
    return x.reshape(*shape[:-1], -1)


def prod(iterable: Sequence[float]) -> float:
    """Take product of iterable like."""
    return reduce(operator.mul, iterable, 1)


def mean(array):
    """Take mean of array like."""
    return sum(array) / len(array)


def aggregate_dicts(dicts, operation=mean):
    """
    Aggregate a sequence of dictionaries to a single dictionary using `operation`. `Operation` should
    reduce a list of all values with the same key. Keys that are not found in one dictionary will
    be mapped to `None`, `operation` can then chose how to deal with those.
    """
    all_keys = set().union(*[el.keys() for el in dicts])
    return {k: operation([dic.get(k, None) for dic in dicts]) for k in all_keys}


def kl_divergence(p, q, z_samples=None, is_lower_var=False):
    """Computes KL[p||q], analytically if possible but with MC if not."""
    try:
        kl_pq = torch.distributions.kl_divergence(p, q)

    except NotImplementedError:
        # removes the event shape
        log_q = q.log_prob(z_samples)
        log_p = p.log_prob(z_samples)
        if is_lower_var:
            # http://joschu.net/blog/kl-approx.html
            log_r = log_q - log_p
            # KL[p||q] = (r−1) - log(r)
            kl_pq = log_r.exp() - 1 - log_r
        else:
            # KL[p||q] = E_p[log p] - E_p[log q]
            kl_pq = log_p - log_q

    return kl_pq


MEANS = dict(
    imagenet=[0.485, 0.456, 0.406],
    cifar10=[0.4914009, 0.48215896, 0.4465308],
    # this is galaxy 128 but shouldn't change much as resizing shouldn't impact much
    galaxy=[0.03294565, 0.04387402, 0.04995899],
    clip=[0.48145466, 0.4578275, 0.40821073],
    stl10=[0.43, 0.42, 0.39],
    stl10_unlabeled=[0.43, 0.42, 0.39],
)
STDS = dict(
    imagenet=[0.229, 0.224, 0.225],
    cifar10=[0.24703279, 0.24348423, 0.26158753],
    galaxy=[0.07004886, 0.07964786, 0.09574898],
    clip=[0.26862954, 0.26130258, 0.27577711],
    stl10=[0.27, 0.26, 0.27],
    stl10_unlabeled=[0.27, 0.26, 0.27],
)


class Normalizer(torch.nn.Module):
    def __init__(self, dataset, is_raise=True):
        super().__init__()
        self.dataset = dataset.lower()
        try:
            self.normalizer = transform_lib.Normalize(
                mean=MEANS[self.dataset], std=STDS[self.dataset]
            )
        except KeyError:
            if is_raise:
                raise KeyError(
                    f"dataset={self.dataset} wasn't found in MEANS={MEANS.keys()} or"
                    f"STDS={STDS.keys()}. Please add mean and std."
                )
            else:
                self.normalizer = None

    def forward(self, x):
        if self.normalizer is None:
            return x

        return self.normalizer(x)


# requires python 3.7+
# taken from https://github.com/bustawin/ordered-set-37
class OrderedSet(MutableSet):
    """A set that preserves insertion order by internally using a dict."""

    def __init__(self, iterable):
        self._d = dict.fromkeys(iterable)

    def add(self, x):
        self._d[x] = None

    def discard(self, x):
        self._d.pop(x)

    def __contains__(self, x):
        return self._d.__contains__(x)

    def __len__(self):
        return self._d.__len__()

    def __iter__(self):
        return self._d.__iter__()


class UnNormalizer(torch.nn.Module):
    def __init__(self, dataset, is_raise=True):
        super().__init__()
        self.dataset = dataset.lower()
        try:
            mean, std = MEANS[self.dataset], STDS[self.dataset]
            self.unnormalizer = transform_lib.Normalize(
                [-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
            )
        except KeyError:
            if is_raise:
                raise KeyError(
                    f"dataset={self.dataset} wasn't found in MEANS={MEANS.keys()} or"
                    f"STDS={STDS.keys()}. Please add mean and std."
                )
            else:
                self.normalizer = None

    def forward(self, x):
        if self.unnormalizer is None:
            return x

        return self.unnormalizer(x)


def is_img_shape(shape):
    """Whether a shape is from an image."""
    try:
        return len(shape) == 3 and (shape[-3] in [1, 3])
    except TypeError:
        return False  # if shape is not list


def is_colored_img(x: torch.Tensor) -> bool:
    """Check if an image or batch of image is colored."""
    if x.shape[-3] not in [1, 3]:
        raise ValueError(f"x doesn't seem to be a (batch of) image as shape={x.shape}.")
    return x.shape[-3] == 3


def at_least_ndim(x: torch.Tensor, ndim: int) -> torch.Tensor:
    """Reshapes a tensor so that it has at least n dimensions."""
    if x is None:
        return None
    return x.view(list(x.shape) + [1] * (ndim - x.ndim))


def tensors_to_fig(
    x: torch.Tensor,
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
    x_labels: list = [],
    y_labels: list = [],
    imgsize: tuple[int, int] = (4, 4),
    small_font: int = 16,
    large_font: int = 20,
) -> plt.Figure:
    """Make a grid-like figure from tensors and labels. Return figure."""
    b, c, h, w = x.shape
    assert (n_rows is not None) or (n_cols is not None)
    if n_cols is None:
        n_cols = b // n_rows
    elif n_rows is None:
        n_rows = b // n_cols

    n_x_labels = len(x_labels)
    n_y_labels = len(y_labels)
    assert n_x_labels in [0, 1, n_cols]
    assert n_y_labels in [0, 1, n_rows]

    figsize = (imgsize[0] * n_cols, imgsize[1] * n_rows)

    constrained_layout = True

    # TODO : remove once use matplotlib 3.4
    # i.e. use fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, sharex=True, sharey=True, figsize=figsize, constrained_layout=True)
    if n_x_labels == 1 or n_y_labels == 1:
        constrained_layout = False

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        squeeze=False,
        sharex=True,
        sharey=True,
        figsize=figsize,
        constrained_layout=constrained_layout,
    )

    for i in range(n_cols):
        for j in range(n_rows):
            xij = x[i * n_rows + j]
            xij = xij.permute(1, 2, 0)
            axij = axes[j, i]
            if xij.size(2) == 1:
                axij.imshow(to_numpy(xij.squeeze()), cmap="gray")
            else:
                axij.imshow(to_numpy(xij))

            axij.get_xaxis().set_ticks([])
            axij.get_xaxis().set_ticklabels([])
            axij.get_yaxis().set_ticks([])
            axij.get_yaxis().set_ticklabels([])

            if n_x_labels == n_cols and j == (n_rows - 1):
                axij.set_xlabel(x_labels[i], fontsize=small_font)
            if n_y_labels == n_rows and i == 0:
                axij.set_ylabel(y_labels[j], fontsize=small_font)

    # TODO : remove all the result once use matplotlib 3.4
    # i.e. use:
    #     if n_x_labels == 1:
    #         fig.supxlabel(x_labels[0])

    #     if n_y_labels == 1:
    #         fig.supylabel(y_labels[0])
    # fig.set_constrained_layout_pads(w_pad=0, h_pad=0.,hspace=hspace, wspace=wspace)

    large_ax = fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)

    if n_x_labels == 1:
        large_ax.set_xlabel(x_labels[0], fontsize=large_font)

    if n_y_labels == 1:
        large_ax.set_ylabel(y_labels[0], fontsize=large_font)

    # TODO : remove once use matplotlib 3.4
    if n_x_labels == 1 or n_y_labels == 1:
        plt.tight_layout()

    return fig


def prediction_loss(
    Y_hat: torch.Tensor, y: torch.Tensor, is_classification: bool = True,
) -> torch.Tensor:
    """Compute the prediction loss for a task.

    Parameters
    ----------
    Y_hat : Tensor
        Predictions.  Should be shape (batch_size, *Y_shape).

    y : Tensor
        Targets.

    is_classification : bool, optional
        Whether we are in a classification task, in which case we use log loss instead of (r)mse.
    """
    # shape : [batch_size, *Y_shape]
    if is_classification:
        loss = F.cross_entropy(Y_hat, y.squeeze().long(), reduction="none")
    else:
        loss = F.mse_loss(Y_hat, y, reduction="none")

    batch_size = y.size(0)

    # shape : [batch_size]
    loss = loss.view(batch_size, -1).mean(keepdim=False, dim=1)

    return loss


def queue_push_(queue: Queue, el: torch.Tensor) -> None:
    """Pushes to the queue without going past the limit."""
    if queue.full():
        queue.get(0)
    queue.put_nowait(el)


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: Optional[str],
    epochs: Optional[int] = None,
    decay_factor: int = 1000,
    k_steps: int = 3,
    name: Optional[str] = None,
    kwargs_config_scheduler: dict[str, Any] = {},
    **kwargs,
):
    """Return the correct lr scheduler as a dictionary as required by pytorch lightning.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer to wrap.

    scheduler_type : {None, "expdecay","UniformMultiStepLR"}U{any torch lr_scheduler}
        Name of the scheduler to use. "expdecay" uses an exponential decay scheduler where the lr
        is decayed by `decay_factor` during training. Needs to be given `epochs`. "UniformMultiStepLR"
        decreases learning by `decay_factor` but as step functions where `k_steps` is number of steps.
        If another `str` it must be a `torch.optim.lr_scheduler` in which case the arguments are given by `kwargs`.

    epochs : int, optional
        Number of epochs during training.

    decay_factor : int, optional
        By how much to reduce learning rate during training. Only if
        `name in ["expdecay","UniformMultiStepLR"]`.

    k_steps : int, optional
        Number of steps for decreasing the learning rate in `"UniformMultiStepLR"`.

    name : str, optional
        Name of the scheduler for logging.

    kwargs_config_scheduler : dict, optional
        Additional kwargs to be passed to pytorch lightning, e.g., monitor / interval / frequency...

    kwargs :
        Additional arguments to any `torch.optim.lr_scheduler`.
    """
    if scheduler_type is None:
        scheduler = None
    elif scheduler_type == "expdecay":
        gamma = (1 / decay_factor) ** (1 / epochs)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    elif scheduler_type == "UniformMultiStepLR":
        delta_epochs = epochs // (k_steps + 1)
        milestones = [delta_epochs * i for i in range(1, k_steps + 1)]
        gamma = (1 / decay_factor) ** (1 / k_steps)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
    else:
        Scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)
        scheduler = Scheduler(optimizer, **kwargs)

    return dict(scheduler=scheduler, name=name, **kwargs_config_scheduler)


def get_optimizer(parameters: Any, mode: str, **kwargs):
    """Return an instantiated optimizer.

    Parameters
    ----------
    parameters : Iterator over tensors
        Parameters of the model used to instantiate optimizer.
    
    mode : {"gdn"}U{any torch.optim optimizer}
        Optimizer to use.

    kwargs :
        Additional arguments to the optimizer.
    """
    Optimizer = getattr(torch.optim, mode)
    if "lr_factor" in kwargs:
        kwargs["lr"] = kwargs["lr"] * kwargs.pop("lr_factor")
    optimizer = Optimizer(parameters, **kwargs)
    return optimizer


def append_optimizer_scheduler_(
    hparams_opt: Namespace,
    hparams_sch: Namespace,
    parameters: Any,
    optimizers: list[Any],
    schedulers: list[Any],
    name: Optional[str] = None,
) -> tuple[list[Any], list[Any]]:
    """Return the correct optimizer and scheduler."""

    # only use parameters that are trainable
    train_params = parameters
    if isinstance(train_params, list) and isinstance(train_params[0], dict):
        # in case you have groups
        for group in train_params:
            group["params"] = list(filter(lambda p: p.requires_grad, group["params"]))
    else:
        train_params = list(filter(lambda p: p.requires_grad, train_params))

    optimizer = get_optimizer(train_params, hparams_opt.mode, **hparams_opt.kwargs)
    optimizers += [optimizer]

    for mode in hparams_sch.modes:
        sch_kwargs = hparams_sch.kwargs.get(mode, {})
        scheduler = get_lr_scheduler(optimizer, mode, name=name, **sch_kwargs)
        schedulers += [scheduler]

    return optimizers, schedulers


@contextlib.contextmanager
def plot_config(
    style="ticks",
    context="notebook",
    palette="colorblind",
    font_scale=1.5,
    font="sans-serif",
    is_ax_off=False,
    is_rm_xticks=False,
    is_rm_yticks=False,
    rc={"lines.linewidth": 2},
    set_kwargs=dict(),
    despine_kwargs=dict(),
    # pretty_renamer=dict(), #TODO
):
    """Temporary seaborn and matplotlib figure style / context / limits / ....

    Parameters
    ----------
    style : dict, None, or one of {darkgrid, whitegrid, dark, white, ticks}
        A dictionary of parameters or the name of a preconfigured set.

    context : dict, None, or one of {paper, notebook, talk, poster}
        A dictionary of parameters or the name of a preconfigured set.

    palette : string or sequence
        Color palette, see :func:`color_palette`

    font : string
        Font family, see matplotlib font manager.

    font_scale : float, optional
        Separate scaling factor to independently scale the size of the
        font elements.

    is_ax_off : bool, optional
        Whether to turn off all axes.

    is_rm_xticks, is_rm_yticks : bool, optional
        Whether to remove the ticks and labels from y or x axis.

    rc : dict, optional
        Parameter mappings to override the values in the preset seaborn
        style dictionaries.

    set_kwargs : dict, optional
        kwargs for matplotlib axes. Such as xlim, ylim, ...

    despine_kwargs : dict, optional
        Arguments to `sns.despine`.
    """
    defaults = plt.rcParams.copy()

    try:
        rc["font.family"] = font
        plt.rcParams.update(rc)

        with sns.axes_style(style=style, rc=rc), sns.plotting_context(
            context=context, font_scale=font_scale, rc=rc
        ), sns.color_palette(palette):
            yield
            last_fig = plt.gcf()
            for i, ax in enumerate(last_fig.axes):
                ax.set(**set_kwargs)

                if is_ax_off:
                    ax.axis("off")

                if is_rm_yticks:
                    ax.axes.yaxis.set_ticks([])

                if is_rm_xticks:
                    ax.axes.xaxis.set_ticks([])

        sns.despine(**despine_kwargs)

    finally:
        with warnings.catch_warnings():
            # filter out depreciation warnings when resetting defaults
            warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
            # reset defaults
            plt.rcParams.update(defaults)


class Annealer:
    """Helper class to perform annealing

    Parameter
    ---------
    initial_value : float
        Start of annealing.

    final_value : float
        Final value after annealing.

    n_steps_anneal : int
        Number of steps before reaching `final_value`. If negative, will swap final and initial.

    start_step : int, optional
        Number of steps to wait for before starting annealing. During the waiting time, the
        hyperparameter will be `default`.

    default : float, optional
        Default hyperparameter value that will be used for the first `start_step`s. If `None` uses
        `initial_value`.

    mode : {"linear", "geometric", "constant"}, optional
        Interpolation mode.
    """

    def __init__(
        self,
        initial_value: float,
        final_value: float,
        n_steps_anneal: int,
        start_step: int = 0,
        default: Optional[float] = None,
        mode: str = "geometric",
    ) -> None:
        if n_steps_anneal < 0:
            # quick trick to swap final / initial
            n_steps_anneal *= -1
            initial_value, final_value = final_value, initial_value

        self.initial_value = initial_value
        self.final_value = final_value
        self.n_steps_anneal = n_steps_anneal
        self.start_step = start_step
        self.default = default if default is not None else self.initial_value
        self.mode = mode.lower()

        if self.mode == "linear":
            delta = self.final_value - self.initial_value
            self.factor = delta / self.n_steps_anneal
        elif self.mode == "constant":
            pass  # nothing to do
        elif self.mode == "geometric":
            delta = self.final_value / self.initial_value
            self.factor = delta ** (1 / self.n_steps_anneal)
        else:
            raise ValueError(f"Unknown mode : {mode}.")

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the interpolator."""
        self.n_training_calls = 0

    def is_annealing(self, n_update_calls: int) -> bool:
        not_const = self.mode != "constant"
        is_not_finished = n_update_calls < (self.n_steps_anneal + self.start_step)
        return not_const and is_not_finished

    def __call__(
        self, is_update: bool = False, n_update_calls: Optional[int] = None
    ) -> float:
        """Return the current value of the hyperparameter.

        Parameter
        ---------
        is_update : bool, optional
            Whether to update the value.

        n_update_calls : int, optional
            Number of updated calls. If given then will override the default counter.
        """
        if is_update:
            self.n_training_calls += 1

        if n_update_calls is None:
            n_update_calls = self.n_training_calls

        if self.start_step > n_update_calls:
            return self.default

        n_actual_training_calls = n_update_calls - self.start_step

        if self.is_annealing(n_update_calls):
            current = self.initial_value
            if self.mode == "geometric":
                current *= self.factor ** n_actual_training_calls
            elif self.mode == "linear":
                current += self.factor * n_actual_training_calls
            else:
                raise ValueError(f"Unknown mode : {self.mode}.")
        else:
            current = self.final_value

        return current


# modified from https://github.com/PyTorchLightning/lightning-bolts/blob/ad771c615284816ecadad11f3172459afdef28e3/pl_bolts/callbacks/byol_updates.py
class MAWeightUpdate(pl.Callback):
    """EMA Weight update rule from BYOL.

        Notes
        -----
        - model should have `p_ZlX` and `ema_p_ZlX`.
        - BYOL claims this keeps the online_network from collapsing.
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
        dataloader_idx: int,
    ) -> None:
        # get networks
        online_net = pl_module.p_ZlX
        target_net = pl_module.ema_p_ZlX

        # update weights
        self.update_weights(online_net, target_net)

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
                + (1 - self.current_tau) * online_p.data
            )


# modified from: https://github.com/facebookresearch/vissl/blob/aa3f7cc33b3b7806e15593083aedc383d85e4a53/vissl/losses/distibuted_sinkhornknopp.py#L11
def sinkhorn_knopp(
    logits,
    eps=0.05,
    n_iter=3,
    is_hard_assignment: bool = False,
    world_size: int = 1,
    is_double_prec: bool = True,
    is_force_no_gpu: bool = False,
):
    """Sinkhorn knopp algorithm to find an equipartition giving logits."""

    # we follow the u, r, c and Q notations from
    # https://arxiv.org/abs/1911.05371

    is_gpu = (not is_force_no_gpu) and torch.cuda.is_available()

    if is_double_prec:
        logits = logits.double()

    # shape: [n_Mx, n_samples]
    Q = (logits / eps).exp().T

    # remove potential infs in Q. Replace by max non inf.
    Q = torch.nan_to_num(Q, posinf=Q.masked_fill(torch.isinf(Q), 0).max().item())

    # number of clusters, and examples to be clustered
    sum_Q = torch.sum(Q, dtype=Q.dtype)
    n_Mx, n_samples = Q.shape
    n_samples_world = n_samples * world_size

    if world_size > 1:
        dist.all_reduce(sum_Q, op=dist.ReduceOp.SUM)

    # make the matrix sum to 1
    Q /= sum_Q

    # Shape: [n_Mx]
    r = torch.ones(n_Mx) / n_Mx
    c = torch.ones(n_samples) / n_samples_world
    if is_double_prec:
        r, c = r.double(), c.double()

    if is_gpu:
        r = r.cuda(non_blocking=True)
        c = c.cuda(non_blocking=True)

    for _ in range(n_iter):
        # for numerical stability, add a small epsilon value for zeros
        if (Q == 0).any():
            Q += 1e-12

        # normalize each row: total weight per prototype must be 1/K. Shape : [n_Mx]
        sum_rows = torch.sum(Q, dim=1, dtype=Q.dtype)
        if world_size > 1:
            dist.all_reduce(sum_rows, op=dist.ReduceOp.SUM)

        #  Shape : [n_Mx]
        u = r / sum_rows

        # remove potential infs in u. Replace by max non inf.
        u = torch.nan_to_num(u, posinf=u.masked_fill(torch.isinf(u), 0).max().item())

        # normalize each row: total weight per prototype must be 1/n_Mx
        Q *= u.unsqueeze(1)

        # normalize each column: total weight per sample must be 1/n_samples_world
        Q *= (c / torch.sum(Q, dim=0, dtype=Q.dtype)).unsqueeze(0)

    Q = (Q / torch.sum(Q, dim=0, keepdim=True, dtype=Q.dtype)).T.float()

    if is_hard_assignment:
        # shape : [n_samples]
        index_max = torch.max(Q, dim=1)[1]
        Q.zero_()
        Q.scatter_(1, index_max.unsqueeze(1), 1)

    return Q


# from : https://github.com/open-mmlab/OpenSelfSup/blob/696d04950e55d504cf33bc83cfadbb4ece10fbae/openselfsup/models/utils/gather_layer.py
class GatherFromGpus(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, tensor):
        ctx.save_for_backward(tensor)
        gathered_tensor = [
            torch.zeros_like(tensor) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_tensor, tensor)
        return tuple(gathered_tensor)

    @staticmethod
    def backward(ctx, *grads):
        (tensor,) = ctx.saved_tensors
        grad_out = torch.zeros_like(tensor)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


gather_from_gpus = GatherFromGpus.apply
