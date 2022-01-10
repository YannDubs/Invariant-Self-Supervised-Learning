from __future__ import annotations

import contextlib
import copy
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
import seaborn as sns
from matplotlib.cbook import MatplotlibDeprecationWarning
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torchvision import transforms as transform_lib

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence

try:
    from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
except ImportError:
    pass


def int_or_ratio(alpha: float, n: int) -> int:
    """Return an integer alpha. If float, it's seen as ratio of `n`."""
    if isinstance(alpha, int):
        return alpha
    return int(alpha * n)


class BatchRMSELoss(nn.Module):
    """Batch root mean squared error."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.eps = eps

    def forward(self, yhat, y):
        batch_mse = self.mse(yhat, y).flatten(1, -1).mean(-1)
        loss = torch.sqrt(batch_mse + self.eps)
        return loss


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


def init_std_modules(module: nn.Module, nonlinearity: str = "relu") -> bool:
    """Initialize standard layers and return whether was intitialized."""
    # all standard layers
    if isinstance(module, nn.modules.conv._ConvNd):
        # used in https://github.com/brain-research/realistic-ssl-evaluation/
        nn.init.kaiming_normal_(
            module.weight, mode="fan_out", nonlinearity=nonlinearity
        )
        try:
            nn.init.zeros_(module.bias)
        except AttributeError:
            pass

    elif isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity=nonlinearity)
        try:
            nn.init.zeros_(module.bias)
        except AttributeError:
            pass

    elif isinstance(module, nn.BatchNorm2d):
        try:
            module.weight.data.fill_(1)
            module.bias.data.zero_()
        except AttributeError:
            # if affine = False
            pass

    else:
        return False

    return True


def weights_init(module: nn.Module, nonlinearity: str = "relu") -> None:
    """Initialize a module and all its descendents.

    Parameters
    ----------
    module : nn.Module
       module to initialize.

    nonlinearity : str, optional
        Name of the nn.functional activation. Used for initialization.
    """
    init_std_modules(module)  # in case you gave a standard module

    # loop over direct children (not grand children)
    for m in module.children():

        if init_std_modules(m):
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


class GrammRBF(nn.Module):
    """Compute a gramm matrix using gaussian RBF.

    Parameters
    ----------
    is_normalize : bool, optional
        Whether to row normalize the output gram matrix.

    pre_gamma_init : float, optional
        Initialization of the gamma parameter.

    p : int, optional
        Which norm to use.

    is_linear : bool, optional
        Whether to pointwise tranform each element of the gram matrix.
    """

    def __init__(
        self, is_normalize=False, pre_gamma_init=0.0, p=2, is_linear=True, **kwargs
    ):
        super().__init__()
        self.is_normalize = is_normalize
        self.pre_gamma_init = pre_gamma_init
        self.p = p
        self.is_linear = is_linear

        self.reset_parameters()

    def reset_parameters(self):
        self.pre_gamma = nn.Parameter(torch.tensor([self.pre_gamma_init]).float())

        if self.is_linear:
            self.scale = nn.Parameter(torch.tensor([1.0]))
            self.bias = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x1, x2):

        # shape : [x1_dim, x2_dim]
        dist = torch.cdist(x1, x2, p=self.p) ** self.p

        gamma = 1e-5 + F.softplus(self.pre_gamma)
        inp = -gamma * dist

        if self.is_normalize:
            # numerically stable normalization of the weights by density
            out = inp.softmax(-1)
        else:
            out = inp.exp()

        if self.is_linear:
            out = self.scale * out + self.bias

        return out


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
    clip=[0.48145466, 0.4578275, 0.40821073],
    stl10=[0.43, 0.42, 0.39],
    stl10_unlabeled=[0.43, 0.42, 0.39],
)
STDS = dict(
    imagenet=[0.229, 0.224, 0.225],
    cifar10=[0.24703279, 0.24348423, 0.26158753],
    # cifar10=[0.2023, 0.1994, 0.2010],
    # whitening paper actually uses the one from pytorch
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
    is_warmup_lr: bool = False,
    warmup_epochs: float = 10,
    warmup_multiplier: float = 1.0,
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

    is_warmup_lr : bool, optional
        Whether to warmup to lr.

    warmup_epochs : float, optional
        For how many epochs to warmup the learning rate if `is_warmup_lr`. If int it's a number of epoch. If
        in ]0,1[ it's percentage of epochs.

    warmup_multiplier : float, optional
        Target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0
        and ends up with the base_lr. For CosineAnnealingLR scheduler need to be warmup_multiplier=1.0.

    kwargs :
        Additional arguments to any `torch.optim.lr_scheduler`.
    """
    if is_warmup_lr:
        # remove the warmup
        epochs = epochs - warmup_epochs
        raw_epochs = epochs

    if "milestones" in kwargs:
        # allow negative milestones which are subtracted to the last epoch
        kwargs["milestones"] = [
            m if m > 0 else epochs + m for m in kwargs["milestones"]
        ]

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

    # TODO: test plateau

    if is_warmup_lr:
        warmup_epochs = int_or_ratio(warmup_epochs, raw_epochs)
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
    logits: torch.Tensor,
    eps: float = 0.05,
    n_iter: int = 3,
    is_hard_assignment: bool = False,
    world_size: int = 1,
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

    world_size : int, optional
        Number of GPUs.
    
    is_double_prec : bool, optional
        Whether to use double precision to ensure that no instabilities.

    is_force_no_gpu : bool, optional
        Forcing computation on CPU even if GPU available.
    """

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



class LearnedSoftmax(nn.Module):
    def __init__(self, dim=-1, temperature=3, is_train_temp=True, is_anneal_temp=False, n_steps=None, min_temperature=0.05, is_gumbel=False, is_hard=False):
        super().__init__()

        self.dim = dim
        self.is_gumbel = is_gumbel
        self.is_hard = is_hard

        self.init_temperature = temperature
        self.is_train_temp = is_train_temp
        self.is_anneal_temp = is_anneal_temp
        self.min_temperature = min_temperature

        if self.is_train_temp:
            assert not is_anneal_temp
            self.log_temperature = nn.Parameter(
                torch.log(torch.tensor(self.init_temperature))
            )
        elif self.is_anneal_temp:
            self.annealer = Annealer(
                self.init_temperature,
                self.min_temperature,
                n_steps_anneal=n_steps,
                mode="geometric",
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        weights_init(self)

        if self.is_train_temp:
            self.log_temperature = nn.Parameter(
                torch.log(torch.tensor(self.init_temperature))
            )

    def get_temperature(self, is_update=False):
        if self.is_train_temp:
            temperature = torch.clamp(
                self.log_temperature.exp(), min=self.min_temperature, max=5
            )
        elif self.is_anneal_temp:
            temperature = self.annealer(is_update=is_update)
        else:
            temperature = self.init_temperature
        return temperature

    def forward(self, logits):
        temperature = self.get_temperature(is_update=True)

        if self.is_gumbel:
            y_hat = F.gumbel_softmax(logits, tau=temperature, is_hard=self.is_hard)
        else:
            y_hat = self.softmax(logits / temperature)

        return y_hat