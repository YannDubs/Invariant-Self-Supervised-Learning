from __future__ import annotations

import contextlib
import copy
import math
import numbers
import operator
import random
import sys
import json
from pathlib import Path
from argparse import Namespace
from collections.abc import MutableMapping, MutableSet, Sequence
from functools import reduce, wraps

from typing import Any, Optional
import logging

import numpy as np
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

def init_std_modules(module: nn.Module) -> bool:
    """Initialize standard layers and return whether was initialized."""
    # all standard layers
    if isinstance(module, nn.modules.conv._ConvNd):
        variance_scaling_(module.weight)
        try:
            nn.init.zeros_(module.bias)
        except AttributeError:
            pass

    elif isinstance(module, nn.Linear):
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

def average_dict(*dicts):
    """Return a dictionary, where every value is avg over the dicts."""
    keys = set(k for d in dicts for k in d.keys() )
    return {k: mean([d[k] for d in dicts if k in d]) for k in keys}

def freeze_module_(model):
    """"Freeze the module."""
    for p in model.parameters():
        p.requires_grad = False


MEANS = {
    "imagenet": [0.485, 0.456, 0.406],
    "tiny-imagenet-200": [0.480, 0.448, 0.398],
    "cifar10": [0.4914009, 0.48215896, 0.4465308],
}
STDS = {
    "imagenet": [0.229, 0.224, 0.225],
    "tiny-imagenet-200": [0.277, 0.269, 0.282],
    "cifar10": [0.24703279, 0.24348423, 0.26158753],
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

    if is_warmup_lr:
        assert scheduler_type == "CosineAnnealingLR"
        assert warmup_multiplier == 1.0
        check_import("pl_bolts", "CosineAnnealingLR with warmup")
        kwargs = copy.deepcopy(kwargs)
        # the following will remove warmup_epochs => should no give
        # epochs = epochs - warmup_epochs
        T_max = kwargs.pop("T_max")
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=warmup_epochs, max_epochs=T_max, **kwargs
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


        scheduler = get_lr_scheduler(optimizer, mode, name=name, **sch_kwargs)
        schedulers += [scheduler]

    return optimizers, schedulers



class BatchNorm1d(nn.BatchNorm1d):
    def __init__(self, *args, is_bias=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_bias = is_bias
        if self.affine and not self.is_bias:
            self.bias.requires_grad = False

def eye_like(x):
    """Return an identity like `x`."""
    return torch.eye(*x.size(), out=torch.empty_like(x))

class DistToEtf(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.running_mean = RunningMean(torch.zeros(self.z_dim))

    def get_etf_rep(self, z):
        z_mean = self.running_mean(z.mean(0))
        z_centered = z - z_mean
        return F.normalize(z_centered, p=2, dim=1)

    def __call__(self, zx, za):
        zx = self.get_etf_rep(zx)
        za = self.get_etf_rep(za)

        MtM = zx @ za.T

        pos_loss = (1 - MtM.diagonal()).mean()  # best: 1
        neg_loss = MtM.masked_select(~eye_like(MtM).bool()).mean()  # best: - 1 /dim

        return pos_loss, neg_loss
