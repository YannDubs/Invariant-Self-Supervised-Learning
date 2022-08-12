from __future__ import annotations

import contextlib
import copy
import json
import math
import numbers
import operator
from pathlib import Path
import random
import sys
import warnings
from argparse import Namespace
from collections.abc import MutableMapping, MutableSet, Sequence
from functools import reduce, wraps
from queue import Queue
from typing import Any, Optional, Union
import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.cbook import MatplotlibDeprecationWarning
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torchvision import transforms as transform_lib

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence

try:
    from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
except:
    pass

logger = logging.getLogger(__name__)

def is_pow_of_k(n, k):
    """Check if `n` is a power of k. Can be wrong for huge n."""
    return math.log(n, k).is_integer()

class RunningMean(nn.Module):
    """Keep track of an exponentially moving average"""
    def __init__(self, init: torch.tensor, alpha_use: float=0.5, alpha_store: float=0.1):
        super().__init__()

        assert 0.0 <= alpha_use <= 1.0
        assert 0.0 <= alpha_store <= 1.0
        self.alpha_use = alpha_use
        self.alpha_store = alpha_store
        self.init = init.double()
        self.register_buffer('running_mean', self.init)

    def reset_parameters(self) -> None:
        self.running_mean = self.init

    def forward(self, x):
        out = self.alpha_use * x + (1 - self.alpha_use) * self.running_mean.float()
        # don't keep all the computational graph to avoid memory++
        self.running_mean = (self.alpha_store * x.detach().double() + (1 - self.alpha_store) * self.running_mean).detach().double()
        return out


def int_or_ratio(alpha: float, n: int) -> int:
    """Return an integer alpha. If float, it's seen as ratio of `n`."""
    if isinstance(alpha, int):
        return alpha
    return int(alpha * n)


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




def cont_tuple_to_tuple_cont(container):
    """Converts a container (list, tuple, dict) of tuple to a tuple of container."""
    if isinstance(container, dict):
        return tuple(dict(zip(container, val)) for val in zip(*container.values()))
    elif isinstance(container, list) or isinstance(container, tuple):
        return tuple(zip(*container))
    else:
        raise ValueError("Unknown container type: {}.".format(type(container)))

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

def file_cache(filename):
    """Decorator to cache the output of a function to disk."""
    def decorator(f):
        @wraps(f)
        def decorated(self, directory, *args, **kwargs):
            filepath = Path(directory) / filename
            if filepath.is_file():
                out = json.loads(filepath.read_text())
            else:
                logger.info(f"Precomputing cache at {filepath}")
                out = f(self, directory, *args, **kwargs)
                filepath.write_text(json.dumps(out))
            return out
        return decorated
    return decorator


# taken from https://github.com/rwightman/pytorch-image-models/blob/d5ed58d623be27aada78035d2a19e2854f8b6437/timm/models/layers/weight_init.py
def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='truncated_normal'):
    """Initialization by scaling the variance."""
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        nn.init.trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")

def init_std_modules(module: nn.Module, nonlinearity: str = "relu", is_JL_init=False) -> bool:
    """Initialize standard layers and return whether was initialized."""
    # all standard layers
    if isinstance(module, nn.modules.conv._ConvNd):
        variance_scaling_(module.weight)
        try:
            nn.init.zeros_(module.bias)
        except AttributeError:
            pass

    elif isinstance(module, nn.Linear):
        if is_JL_init and (module.weight.shape[0] < module.weight.shape[1]):
            johnson_lindenstrauss_init_(module)

        else:
            nn.init.trunc_normal_(module.weight, std=0.02)
            try:
                nn.init.zeros_(module.bias)
            except AttributeError: # no bias
                pass

    elif isinstance(module, nn.modules.batchnorm._NormBase):
        if module.affine:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    else:
        return False

    return True

def johnson_lindenstrauss_init_(m):
    """Initialization for low dimension projection => johnson lindenstrauss lemma."""
    torch.nn.init.normal_(m.weight, std=1 / math.sqrt(m.weight.shape[0]))

def weights_init(module: nn.Module, nonlinearity: str = "relu", is_JL_init=False) -> None:
    """Initialize a module and all its descendents.

    Parameters
    ----------
    module : nn.Module
       module to initialize.

    nonlinearity : str, optional
        Name of the nn.functional activation. Used for initialization.
    """
    init_std_modules(module, is_JL_init=is_JL_init)  # in case you gave a standard module

    # loop over direct children (not grand children)
    for m in module.children():

        if init_std_modules(m):
            pass
        elif hasattr(m, "reset_parameters"):
            # if has a specific reset
            # Imp: don't go in grand children because you might have specific weights you don't want to reset
            m.reset_parameters()
        else:
            weights_init(m, nonlinearity=nonlinearity, is_JL_init=is_JL_init)  # go to grand children


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

def average_dict(*dicts):
    """Return a dictionary, where every value is avg over the dicts."""
    keys = set(k for d in dicts for k in d.keys() )
    return {k: mean([d[k] for d in dicts if k in d]) for k in keys}

def freeze_module_(model):
    """"Freeze the module."""
    for p in model.parameters():
        p.requires_grad = False

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


MEANS = {
    "imagenet": [0.485, 0.456, 0.406],
    "tiny-imagenet-200": [0.480, 0.448, 0.398],
    "cifar10": [0.4914009, 0.48215896, 0.4465308],
    "clip": [0.48145466, 0.4578275, 0.40821073],
    "stl10": [0.43, 0.42, 0.39],
    "stl10_unlabeled": [0.43, 0.42, 0.39],
}
STDS = {
    "imagenet": [0.229, 0.224, 0.225],
    "tiny-imagenet-200": [0.277, 0.269, 0.282],
    "cifar10": [0.24703279, 0.24348423, 0.26158753],
    # cifar10=[0.2023, 0.1994, 0.2010],
    # whitening paper actually uses the one from pytorch
    "clip": [0.26862954, 0.26130258, 0.27577711],
    "stl10": [0.27, 0.26, 0.27],
    "stl10_unlabeled": [0.27, 0.26, 0.27],
}


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
    k_steps: Optional[int] = None,
    name: Optional[str] = None,
    kwargs_config_scheduler: dict[str, Any] = {},
    is_warmup_lr: bool = False,
    warmup_epochs: float = 10,
    warmup_multiplier: float = 1.0,
    decay_per_step: Optional[float] = None,
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
        Number of steps for decreasing the learning rate in `"UniformMultiStepLR"`. If `None` uses 3 if number of
        epochs is more than 300 and 200 else.

    name : str, optional
        Name of the scheduler for logging.

    kwargs_config_scheduler : dict, optional
        Additional kwargs to be passed to pytorch lightning, e.g., monitor / interval / frequency...

    is_warmup_lr : bool, optional
        Whether to warmup to lr.

    warmup_epochs : float, optional
        For how many epochs to warmup the learning rate if `is_warmup_lr`. If int it's a number of epoch. If
        in ]0,1[ it's percentage of epochs.

    warmup_multiplier : float, optional
        Target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0
        and ends up with the base_lr. For CosineAnnealingLR scheduler need to be warmup_multiplier=1.0.

    decay_per_step : float, optional
        Decay to use per step. If given will replace `decay_factor` which is over training.

    kwargs :
        Additional arguments to any `torch.optim.lr_scheduler`.
    """
    if is_warmup_lr:
        warmup_epochs = int_or_ratio(warmup_epochs, epochs)
        # remove the warmup
        epochs = epochs - warmup_epochs

    if "milestones" in kwargs:
        # allow negative milestones which are subtracted to the last epoch
        kwargs["milestones"] = [
            m if m > 0 else epochs + m for m in kwargs["milestones"]
        ]

    if scheduler_type is None:
        scheduler = None
    elif scheduler_type == "expdecay":
        if decay_per_step is not None:
            gamma = (1 / decay_per_step)
        else:
            gamma = (1 / decay_factor) ** (1 / epochs)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    elif scheduler_type == "UniformMultiStepLR":
        if k_steps is None:
            k_steps = (5 + (epochs // 1000)) if epochs > 300 else 3
        delta_epochs = epochs // (k_steps + 1)
        milestones = [delta_epochs * i for i in range(1, k_steps + 1)]
        if decay_per_step is not None:
            gamma = (1 / decay_per_step)
        else:
            gamma = (1 / decay_factor) ** (1 / k_steps)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
    else:
        Scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)
        scheduler = Scheduler(optimizer, **kwargs)

    # TODO: test plateau

    if is_warmup_lr:
        if scheduler_type == "CosineAnnealingLR":
            # TODO: test
            assert warmup_multiplier == 1.0
            check_import("pl_bolts", "CosineAnnealingLR with warmup")
            kwargs = copy.deepcopy(kwargs)
            # the following will remove warmup_epochs => should no give
            # epochs = epochs - warmup_epochs
            T_max = kwargs.pop("T_max")
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=warmup_epochs, max_epochs=T_max, **kwargs
            )
        else:
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=warmup_multiplier,
                total_epoch=warmup_epochs,
                after_scheduler=scheduler,
            )

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

    # TODO warming up likely does not work well when using multiple schedulers
    for i, mode in enumerate(hparams_sch.modes):
        sch_kwargs = hparams_sch.kwargs.get(mode, {})
        sch_kwargs.update(hparams_sch.kwargs.base)
        sch_kwargs = copy.deepcopy(sch_kwargs)

        is_warmup_lr = sch_kwargs.get("is_warmup_lr", False)
        if i < len(hparams_sch.modes) - 1 and is_warmup_lr:
            # TODO: currently not correct. because if multiple schedulers the first ones will act normally but they
            # TODO: should wait until warmup is done need to add some waiting scheduler
            # never warmup besides the last
            sch_kwargs["is_warmup_lr"] = False

        is_plat = mode == "ReduceLROnPlateau"
        if is_warmup_lr and is_plat:
            # pytorch lightning will not work with the current code for plateau + warmup
            # because they use `isinstance` to know whether to give a metric
            # => instead just append a linear warming up
            lin = LinearLR(
                optimizer,
                start_factor=1 / 100,
                total_iters=sch_kwargs["warmup_epochs"],
            )
            schedulers += [dict(scheduler=lin, name=name)]
            sch_kwargs["is_warmup_lr"] = False

        scheduler = get_lr_scheduler(optimizer, mode, name=name, **sch_kwargs)
        schedulers += [scheduler]

    return optimizers, schedulers


# modified from https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/6b5e8953a80aef5b324104dc0c2e9b8c34d622bd/warmup_scheduler/scheduler.py#L5
class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError("multiplier should be greater thant or equal to 1.")
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False

        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.after_scheduler._last_lr = self.after_scheduler.base_lrs
                    for i, group in enumerate(self.optimizer.param_groups):
                        group["lr"] = self.after_scheduler.base_lrs[i]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        else:
            # before finished warming up do not call underlying scheduler
            if self.multiplier == 1.0:
                return [
                    base_lr * (float(self.last_epoch) / self.total_epoch)
                    for base_lr in self.base_lrs
                ]
            else:
                return [
                    base_lr
                    * (
                        (self.multiplier - 1.0) * self.last_epoch / self.total_epoch
                        + 1.0
                    )
                    for base_lr in self.base_lrs
                ]

    def step_ReduceLROnPlateau(self, metrics):
        self.last_epoch += 1
        if self.last_epoch < self.total_epoch:
            warmup_lr = self.get_lr()
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            self.after_scheduler.step(metrics)

    def step(self, metrics=None):
        if not isinstance(self.after_scheduler, ReduceLROnPlateau):
            if self.finished and self.after_scheduler:
                self.after_scheduler.step()
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step()
        else:
            self.step_ReduceLROnPlateau(metrics)


# TODO remove in pytorch 1.10 as this is copied from there
class LinearLR(_LRScheduler):
    """Decays the learning rate of each parameter group by linearly changing small
    multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        start_factor (float): The number we multiply learning rate in the first epoch.
            The multiplication factor changes towards end_factor in the following epochs.
            Default: 1./3.
        end_factor (float): The number we multiply learning rate at the end of linear changing
            process. Default: 1.0.
        total_iters (int): The number of iterations that multiplicative factor reaches to 1.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(
        self,
        optimizer,
        start_factor=1.0 / 3,
        end_factor=1.0,
        total_iters=5,
        last_epoch=-1,
        verbose=False,
    ):
        if start_factor > 1.0 or start_factor < 0:
            raise ValueError(
                "Starting multiplicative factor expected to be between 0 and 1."
            )

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError(
                "Ending multiplicative factor expected to be between 0 and 1."
            )

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super(LinearLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [
                group["lr"] * self.start_factor for group in self.optimizer.param_groups
            ]

        if self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        return [
            group["lr"]
            * (
                1.0
                + (self.end_factor - self.start_factor)
                / (
                    self.total_iters * self.start_factor
                    + (self.last_epoch - 1) * (self.end_factor - self.start_factor)
                )
            )
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        return [
            base_lr
            * (
                self.start_factor
                + (self.end_factor - self.start_factor)
                * min(self.total_iters, self.last_epoch)
                / self.total_iters
            )
            for base_lr in self.base_lrs
        ]


@contextlib.contextmanager
def plot_config(
    style="ticks",
    context="talk",
    palette="colorblind",
    font_scale=1.15,
    font="sans-serif",
    is_ax_off=False,
    is_rm_xticks=False,
    is_rm_yticks=False,
    rc={"lines.linewidth": 4},
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

class BatchNorm1d(nn.BatchNorm1d):
    def __init__(self, *args, is_bias=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_bias = is_bias
        if self.affine and not self.is_bias:
            self.bias.requires_grad = False


# modified from: https://github.com/facebookresearch/vissl/blob/aa3f7cc33b3b7806e15593083aedc383d85e4a53/vissl/losses/distibuted_sinkhornknopp.py#L11
def sinkhorn_knopp(
    logits: torch.Tensor,
    eps: float = 0.05,
    n_iter: int = 3,
    is_hard_assignment: bool = False,
    is_double_prec: bool = True,
    is_force_no_gpu: bool = False,
):
    """Sinkhorn knopp algorithm to find an equipartition giving logits.
    
    Parameters
    ----------
    logits : torch.Tensor
        Logits of shape (n_samples, n_Mx).

    eps : float, optional
        Regularization parameter for Sinkhorn-Knopp algorithm. Reducing epsilon parameter encourages
        the assignments to be sharper (i.e. less uniform), which strongly helps avoiding collapse. 
        However, using a too low value for epsilon may lead to numerical instability.

    n_iter : int, optional
        Nubmer of iterations. Larger is better but more compute.

    is_hard_assignment : bool, optional
        Whether to use hard assignements rather than soft ones.
    
    is_double_prec : bool, optional
        Whether to use double precision to ensure that no instabilities.

    is_force_no_gpu : bool, optional
        Forcing computation on CPU even if GPU available.
    """

    # we follow the u, r, c and Q notations from
    # https://arxiv.org/abs/1911.05371

    is_gpu = (not is_force_no_gpu) and torch.cuda.is_available()

    logits = logits.float()
    if is_double_prec:
        logits = logits.double()

    # Q shape: [n_Mx, n_samples]
    # log sum exp trick for stability
    logits = logits / eps
    M = torch.max(logits)
    Q = (logits - M).exp().T

    # remove potential infs in Q. Replace by max non inf.
    Q = torch.nan_to_num(Q, posinf=Q.masked_fill(torch.isinf(Q), 0).max().item())

    # make the matrix sum to 1
    sum_Q = torch.sum(Q, dtype=Q.dtype)
    Q /= sum_Q

    # number of clusters, and examples to be clustered
    n_Mx, n_samples = Q.shape

    # Shape: [n_Mx]
    r = torch.ones(n_Mx) / n_Mx
    c = torch.ones(n_samples) / n_samples
    if is_double_prec:
        r, c = r.double(), c.double()

    if is_gpu:
        r = r.cuda(non_blocking=True)
        c = c.cuda(non_blocking=True)

    for _ in range(n_iter):
        # normalize each row: total weight per prototype must be 1/K. Shape : [n_Mx]
        sum_rows = torch.sum(Q, dim=1, dtype=Q.dtype)

        # for numerical stability, add a small epsilon value for zeros
        if len(torch.nonzero(sum_rows == 0)) > 0:
            Q += 1e-12
            sum_rows = torch.sum(Q, dim=1, dtype=Q.dtype)

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

def warmup_cosine_scheduler(step, warmup_steps, total_steps, boundary=0, optima=1):
    """Computes scheduler for cosine with warmup."""
    if step < warmup_steps:
        return boundary + (float(step) / float(max(1, warmup_steps))) * (optima - boundary)

    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))

    return boundary + 0.5 * (1.0 + math.cos(math.pi * progress)) * (optima - boundary)

def eye_like(x):
    """Return an identity like `x`."""
    return torch.eye(*x.size(), out=torch.empty_like(x))

def rel_distance(x1, x2, detach_at=None, **kwargs):
    """
    Return the relative distance of positive examples compaired to negative.
    ~0 means that positives are essentially the same compared to negatives.
    ~1 means that positives and negatives are essentially indistinguishable.
    """
    dist = torch.cdist(x1, x2, **kwargs)
    dist_inter_class = dist[~eye_like(dist).bool()].mean()
    dist_intra_class = dist.diag().mean()
    rel_dist = dist_intra_class / (dist_inter_class + 1e-5)
    if detach_at is not None and detach_at > rel_dist:
        # only detach negatives, positives can still go to 0 var
        rel_dist = dist_intra_class / (dist_inter_class.detach() + 1e-5)
    return rel_dist

def rel_variance(x1, x2, detach_at=None):
    """
    Return the relative distance of positive examples compaired to negative.
    ~0 means that positives are essentially the same compared to negatives.
    ~1 means that positives and negatives are essentially indistinguishable.
    """
    # Var[X] =  E[(X - X')^2] / 2 for iid X,X'
    paired_variance = torch.cdist(x1, x2, p=2) ** 2 / 2
    var_inter_class = paired_variance[~eye_like(paired_variance).bool()].mean()
    var_intra_class = paired_variance.diag().mean()
    rel_var = var_intra_class / (var_inter_class + 1e-5)
    if detach_at is not None and detach_at > rel_var:
        # only detach negatives, positives can still go to 0 var
        rel_var = var_intra_class / (var_inter_class.detach() + 1e-5)
    return rel_var

def corrcoeff_to_eye_loss(x1,x2):
    batch_size, dim = x1.shape
    x1_norm = (x1 - x1.mean(1, keepdim=True)) / x1.std(1, keepdim=True)
    x2_norm = (x2 - x2.mean(1, keepdim=True)) / x2.std(1, keepdim=True)
    corr_coeff = x1_norm @ x2_norm.T / dim
    pos_loss = (corr_coeff.diagonal() - 1).pow(2)
    neg_loss_1 = corr_coeff.masked_select(~eye_like(corr_coeff).bool()).view(batch_size, batch_size - 1).pow(2).mean(1)
    neg_loss_2 = corr_coeff.T.masked_select(~eye_like(corr_coeff).bool()).view(batch_size, batch_size - 1).pow(2).mean(1)
    neg_loss = (neg_loss_1 + neg_loss_2) / 2  # symmetrize
    return pos_loss + neg_loss


class DistToEtf(nn.Module):
    def __init__(
        self,
        z_shape,
        is_exact_etf=False,
        is_already_normalized=False,
    ) :
        super().__init__()
        self.is_already_normalized = is_already_normalized
        self.is_exact_etf = is_exact_etf
        self.z_dim = z_shape if isinstance(z_shape, int) else prod(z_shape)
        self.running_mean = RunningMean(torch.zeros(self.z_dim))

    def get_etf_rep(self, z):
        z_mean = self.running_mean(z.mean(0))
        z_centered = z - z_mean
        return F.normalize(z_centered, p=2, dim=1)

    def __call__(self, zx, za, is_return_pos_neg=False):
        z_dim = zx.shape[1]

        if not self.is_already_normalized:
            zx = self.get_etf_rep(zx)
            za = self.get_etf_rep(za)

        MtM = zx @ za.T
        if self.is_exact_etf:
            pos_loss = (MtM.diagonal() - 1).pow(2).mean()  # want it to be 1
            neg_loss = (1 / z_dim + MtM.masked_select(~eye_like(MtM).bool())).pow(2).mean()  # want it to be - 1 /dim
        else:
            pos_loss = (1 - MtM.diagonal()).mean()  # directly maximize
            neg_loss = MtM.masked_select(~eye_like(MtM).bool()).mean()  # directly minimize

        if is_return_pos_neg:
            return pos_loss, neg_loss, pos_loss + neg_loss

        return pos_loss + neg_loss
