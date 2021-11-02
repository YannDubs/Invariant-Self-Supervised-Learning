from __future__ import annotations

from collections.abc import Callable, Sequence
from numbers import Number
from typing import Any, Optional, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (Distribution, Independent, Normal,
                                 RelaxedOneHotCategorical, constraints)
from torch.distributions.utils import broadcast_all

from .architectures import get_Architecture
from .helpers import replicate_shape, weights_init

__all__ = ["CondDist", "get_marginalDist", "GumbelCategorical"]


### FAMILY OF DISTRIBUTIONS ###
def get_Distribution(family: str) -> Any:
    """Return the correct uninstantiated family of distribution."""
    if family == "diag_gaussian":
        Family = DiagGaussian
    elif family == "deterministic":
        Family = Deterministic
    elif family == "gumbel_categorical":
        Family = GumbelCategorical
    else:
        raise ValueError(f"Unknown family={family}.")
    return Family


class Distributions:
    """Base class for distributions that can be instantiated with joint suff stat."""

    n_param = None  # needs to be defined in each class

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_suff_params(
        cls, concat_suff_params: torch.Tensor, **kwargs
    ) -> Distributions:
        """Initialize the distribution using the concatenation of sufficient parameters (output of NN)."""
        # shape: [batch_size, out_shape] * n_param
        suff_params = einops.rearrange(
            concat_suff_params, "b ... (d p) -> b ... d p", p=cls.n_param
        ).unbind(-1)
        suff_params = cls.preprocess_suff_params(*suff_params)
        return cls(*suff_params, **kwargs)

    @classmethod
    def preprocess_suff_params(cls, *suff_params: torch.Tensor) -> tuple[torch.Tensor]:
        """Preprocesses parameters outputted from network (usually to satisfy some constraints)."""
        return suff_params

    def detach(self, is_grad_flow: bool = False) -> Distributions:
        """
        Detaches all the parameters. With optional `is_grad_flow` that would ensure pytorch does
        not complain about no grad (by setting grad to 0.
        """
        raise NotImplementedError()


class DiagGaussian(Distributions, Independent):
    """Gaussian with diagonal covariance.

    Parameters
    ----------
    diag_loc : torch.Tensor
        Mean of the last dimension of the desired distribution.

    diag_scale : torch.Tensor
        Standard deviation of the last dimension of the desired distribution.
    """

    n_param = 2
    min_std = 1e-5  # need to be class attribute for `preprocess_suff_params`

    def __init__(self, diag_loc: torch.Tensor, diag_scale: torch.Tensor) -> None:
        super().__init__(Normal(diag_loc, diag_scale), 1)

    @classmethod
    def preprocess_suff_params(
        cls, diag_loc: torch.Tensor, diag_log_var: torch.Tensor
    ) -> tuple[torch.Tensor]:
        # usually exp()**0.5, but you don't want to explode
        diag_scale = F.softplus(diag_log_var) + cls.min_std
        return diag_loc, diag_scale

    def detach(self, is_grad_flow: bool = False) -> DiagGaussian:
        loc = self.base_dist.loc.detach()
        scale = self.base_dist.scale.detach()

        if is_grad_flow:
            loc = loc + 0 * self.base_dist.loc
            scale = scale + 0 * self.base_dist.scale

        return DiagGaussian(loc, scale)


class RelaxedCategorical(Distributions, Independent):
    n_param = 1

    def __init__(
        self, param: torch.Tensor, temperature: float = 1, is_hard: bool = False
    ) -> None:
        super().__init__(
            GumbelCategorical(temperature=temperature, logits=param, is_hard=is_hard), 1
        )

    def detach(self, is_grad_flow: bool = False) -> RelaxedCategorical:
        logits = self.base_dist.logits.detach()

        if is_grad_flow:
            logits = logits + 0 * self.base_dist.logits

        return RelaxedCategorical(logits)


class Deterministic(Distributions, Independent):
    """Delta function distribution (i.e. no stochasticity)."""

    n_param = 1

    def __init__(self, param: torch.Tensor) -> None:
        super().__init__(Delta(param), 1)

    def detach(self, is_grad_flow: bool = False) -> Deterministic:
        loc = self.base_dist.loc.detach()

        if is_grad_flow:
            loc = loc + 0 * self.base_dist.loc

        return Deterministic(loc)


# class VonMisesFisher(Independent)
# TODO: would be great to add given relation with contrastive learning. But currently not in pytorch
# https://github.com/pytorch/pytorch/issues/13811


### CONDITIONAL DISTRIBUTIONS ###
class CondDist(nn.Module):
    """Return the (uninstantiated) correct conditional distribution.

    Parameters
    ----------
    in_shape : sequence of int

    out_shape : sequence of int

    architecture : str or Callable
        If module should be instantiated using `Architecture(in_shape, out_dim)`. If str will be given to
        `get_Architecture`.

    arch_kwargs : dict, optional
        Arguments to `get_Architecture`.

    family : {"diaggaussian","deterministic"}
        Family of the distribution (after conditioning), this can be easily extendable to any
        distribution in `torch.distribution`.

    fam_kwargs : dict, optional
        Additional arguments to the `Family`.
    """

    def __init__(
        self,
        in_shape: Sequence[int],
        out_shape: Sequence[int],
        architecture: Union[str, Callable],
        arch_kwargs: dict[str, Any] = {},
        family: str = "deterministic",
        fam_kwargs: dict[str, Any] = {},
    ) -> None:
        super().__init__()

        self.Family = get_Distribution(family)
        self.in_shape = in_shape
        self.out_shape = [out_shape] if isinstance(out_shape, int) else out_shape
        self.fam_kwargs = fam_kwargs

        Architecture = get_Architecture(architecture, **arch_kwargs)
        params_outs_shape = replicate_shape(self.out_shape, self.Family.n_param)
        self.mapper = Architecture(in_shape, params_outs_shape)

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> Distributions:
        """Compute the distribution conditioned on `x`.

        Parameters
        ----------
        x: torch.Tensor, shape: [batch_size, *in_shape]
            Input on which to condition the output distribution.

        Return
        ------
        p(.|x) : torch.Distribution, batch shape: [batch_size] event shape: [out_dim]
        """

        # shape: [batch_size, out_shape[:-1], out_dim * n_param]
        suff_params = self.mapper(x)

        # batch shape: [batch_size, out_shape[:-1]] ; event shape: [out_dim]
        p__lx = self.Family.from_suff_params(suff_params, **self.fam_kwargs)

        return p__lx

    def reset_parameters(self) -> None:
        weights_init(self)


### MARGINAL DISTRIBUTIONS ###
def get_marginalDist(family: str, out_dim: int) -> nn.Module:
    """Return an approximate marginal distribution. Useful for priors.

    Parameters
    ---------
    family : {"unitgaussian"}

    out_dim : int
        Shape of the output distribution.

    Notes
    -----
    - Marginal distributions are Modules that TAKE NO ARGUMENTS and return the correct distribution
    as they are modules, they ensure that parameters are on the correct device.
    """
    if family == "unitgaussian":
        marginal = MarginalUnitGaussian(out_dim)
    else:
        raise ValueError(f"Unknown family={family}.")
    return marginal


class MarginalUnitGaussian(nn.Module):
    """Mean 0 covariance 1 Gaussian."""

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.out_dim = out_dim

        self.register_buffer("loc", torch.as_tensor([0.0] * self.out_dim))
        self.register_buffer("scale", torch.as_tensor([1.0] * self.out_dim))

    def forward(self) -> Distribution:
        return Independent(Normal(self.loc, self.scale), 1)


### torch.Distribution extensions ###
# modified from: http://docs.pyro.ai/en/stable/_modules/pyro/distributions/delta.html#Delta
class Delta(Distribution):
    """
    Degenerate discrete distribution (a single point).

    Parameters
    ----------
    loc: torch.Tensor
        The single support element.

    log_density: torch.Tensor, optional
        An optional density for this Delta. This is useful to keep the class of :class:`Delta`
        distributions closed under differentiable transformation.
    """

    has_rsample = True
    arg_constraints = {"loc": constraints.real, "log_density": constraints.real}
    support = constraints.real

    def __init__(
        self, loc: torch.Tensor, log_density: float = 0.0, validate_args=None
    ) -> None:
        self.loc, self.log_density = broadcast_all(loc, log_density)

        if isinstance(loc, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()

        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self) -> torch.Tensor:
        return self.loc

    @property
    def variance(self) -> torch.Tensor:
        return torch.zeros_like(self.loc)

    def expand(
        self, batch_shape: Sequence[int], _instance: Optional[Delta] = None
    ) -> torch.Tensor:
        new = self._get_checked_instance(Delta, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.log_density = self.log_density.expand(batch_shape)
        super().__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape: Sequence[int] = torch.Size()) -> torch.Tensor:
        shape = list(sample_shape) + list(self.loc.shape)
        return self.loc.expand(shape)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_prob = (x == self.loc).type(x.dtype).log()
        return log_prob + self.log_density


class GumbelCategorical(RelaxedOneHotCategorical):
    """
    Extended RelaxedOneHotCategorical to give the possibility of using hard during forward.

    Parameters
    ----------
    temperature: float or torch.Tensor, optional
        Relaxation temperature. Smaller means better approximation of categorical.

    probs: torch.Tensor, optional
        Event probabilities.

    logits: torch.Tensor, optional
        Unnormalized log probability for each event.

    is_hard: bool, optional
        Whether to round during the forward pass.
    """

    def __init__(
        self,
        temperature: Union[float, torch.Tensor] = 1.0,
        probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        is_hard: bool = False,
    ) -> None:
        self.is_hard = is_hard
        super().__init__(
            temperature,
            probs=probs,
            logits=logits,
        )

    def rsample(self, sample_shape: Sequence[int] = torch.Size()) -> torch.Tensor:
        samples = super().rsample(sample_shape=sample_shape)

        if self.is_hard:
            dim = -1  # softmax is along last dim
            soft = samples
            index = samples.max(dim, keepdim=True)[1]
            hard = torch.zeros_like(
                samples, memory_format=torch.legacy_contiguous_format
            ).scatter_(dim, index, 1.0)
            samples = hard - soft.detach() + soft

        return samples
