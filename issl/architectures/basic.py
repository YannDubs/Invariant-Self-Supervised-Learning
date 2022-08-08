from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from issl.architectures.helpers import get_Activation, get_Normalization
from issl.helpers import (batch_flatten, batch_unflatten, freeze_module_, johnson_lindenstrauss_init_, prod,
                          weights_init, BatchNorm1d)

__all__ = ["FlattenMLP", "FlattenLinear", "Resizer", "Flatten", "FlattenCosine", "OurProjectionHead"]


class MLP(nn.Module):
    """Multi Layer Perceptron.

    Parameters
    ----------
    in_dim : int

    out_dim : int

    hid_dim : int, optional
        Number of hidden neurones.

    n_hid_layers : int, optional
        Number of hidden layers.

    norm_layer : nn.Module or {"identity","batch"}, optional
        Normalizing layer to use.

    activation : {any torch.nn activation}, optional
        Activation to use.

    is_skip_hidden : bool, optional
        Whether to skip all the hidden layers with a residual connection.

    is_cosine : bool, optional
        Whether the last layer should be cosine similarity instead of linear.

    kwargs_prelinear : bool, optional
        Additional arguments to the first linear layer which is a `FlattenLinear`.

    kwargs:
        Additional arguments of the last linear layer which is a `FlattenLinear`.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_hid_layers: int = 2,
        hid_dim: int = 2048,
        norm_layer: str = "batch",
        activation: str = "ReLU",
        is_skip_hidden: bool = False,
        is_cosine: bool= False,
        kwargs_prelinear: dict = {},
        **kwargs
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_hid_layers = n_hid_layers
        self.hid_dim = hid_dim
        Activation = get_Activation(activation)
        Norm = get_Normalization(norm_layer, dim=1)
        # don't use bias with batch_norm https://twitter.com/karpathy/status/1013245864570073090?l...
        bias_hidden = Norm == nn.Identity
        self.is_skip_hidden = is_skip_hidden
        self.is_cosine = is_cosine


        PreLinear = FlattenLinear if len(kwargs_prelinear) > 0 else nn.Linear # TODO use only FlattenLinear (currently backward compatibility)
        self.pre_block = nn.Sequential(
            PreLinear(in_dim, hid_dim, bias=bias_hidden, **kwargs_prelinear),
            Norm(hid_dim),
            Activation(),
        )

        layers = []
        # start at 1 because pre_block
        for _ in range(1, self.n_hid_layers):
            layers += [
                nn.Linear(hid_dim, hid_dim, bias=bias_hidden),
                Norm(hid_dim),
                Activation(),
            ]
        self.hidden_block = nn.Sequential(*layers)

        # using flatten linear to have bottleneck size
        PostBlock = FlattenCosine if self.is_cosine else FlattenLinear
        self.post_block = PostBlock(hid_dim, out_dim, **kwargs)

        self.reset_parameters()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # flatten to make for normalizing layer => only 2 dim
        X, shape = batch_flatten(X)
        X = self.pre_block(X)

        if self.is_skip_hidden:
            # use a residual connection for all the hidden block
            X = self.hidden_block(X) + X
        else:
            X = self.hidden_block(X)

        X = self.post_block(X)
        X = batch_unflatten(X, shape)
        return X

    def reset_parameters(self):
        weights_init(self)
        
        if self.MLP_bottleneck_prelinear is not None:
            johnson_lindenstrauss_init_(self.pre_block[0])


class FlattenMLP(MLP):
    """
    MLP that can take a multi dimensional array as input and output (i.e. can be used with same
    input and output shape as CNN but permutation invariant.). E.g. for predicting an image use
    `out_shape=(32,32,3)` and this will predict 32*32*3 and then reshape.

    Parameters
    ----------
    in_shape : tuple or int

    out_shape : tuple or int

    kwargs :
        Additional arguments to `MLP`.
    """

    def __init__(
        self, in_shape: Sequence[int], out_shape: Sequence[int], **kwargs
    ) -> None:
        self.in_shape = [in_shape] if isinstance(in_shape, int) else in_shape
        self.out_shape = [out_shape] if isinstance(out_shape, int) else out_shape

        in_dim = prod(self.in_shape)
        out_dim = prod(self.out_shape)
        super().__init__(in_dim=in_dim, out_dim=out_dim, **kwargs)

        self.reset_parameters()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # flattens in_shape
        X = X.flatten(start_dim=X.ndim - len(self.in_shape))
        X = super().forward(X)
        # unflattens out_shape
        X = X.unflatten(dim=-1, sizes=self.out_shape)
        return X

    def reset_parameters(self):
        weights_init(self)


class OurProjectionHead(nn.Module):
    """Multi Layer Perceptron head with skip connection. Will give an MLP with
    `in_dim - bottleneck_size -  (hid_dim - ) * n_hid_layers - bottleneck_size - out_dim` with hiddens skipped.

    Parameters
    ----------
    in_dim : int

    out_dim : int

    hid_dim : int, optional
        Number of hidden neurones of the main layers.

    n_hid_layers : int, optional
        Number of hidden main layers (those that will be skipped over).

    is_force_pre_post_layer : bool, optional
        Whether to use pre and post layers even if the input and / our outputs have the right shape.

    bottleneck_size : int, optional
        What bottleneck size to use (start and end of skip). If `None` uses `out_dim`.
    """

    def __init__(
            self,
            in_shape: Sequence[int],
            out_shape: Sequence[int],
            n_hid_layers: int = 2,
            hid_dim: int = 2048,
            is_force_pre_post_layer : bool = False,
            bottleneck_size : int = 512,
            is_JL_init: bool = True
    ) -> None:
        super().__init__()

        self.in_shape = [in_shape] if isinstance(in_shape, int) else in_shape
        self.out_shape = [out_shape] if isinstance(out_shape, int) else out_shape
        self.in_dim = prod(self.in_shape)
        self.out_dim = prod(self.out_shape)
        self.n_hid_layers = n_hid_layers
        self.hid_dim = hid_dim
        self.is_force_pre_post_layer = is_force_pre_post_layer
        self.bottleneck_size = bottleneck_size
        self.is_JL_init = is_JL_init

        if self.in_dim != self.bottleneck_size or self.is_force_pre_post_layer:
            self.pre_block = nn.Linear(self.in_dim, self.bottleneck_size, bias=False)
        else:
            self.pre_block = nn.Identity()

        layers = []
        input_dim = self.bottleneck_size
        output_dim = self.hid_dim
        # bottleneck not counted as hidden layer
        for i in range(self.n_hid_layers+1):
            if i == self.n_hid_layers:
                output_dim = self.bottleneck_size
            if i > 0:
                input_dim = self.hid_dim

            layers += [
                nn.BatchNorm1d(input_dim),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim, output_dim, bias=False),
            ]
        self.hidden_block = nn.Sequential(*layers)

        if self.out_dim != self.bottleneck_size or self.is_force_pre_post_layer:
            self.post_block = nn.Sequential(nn.BatchNorm1d(self.bottleneck_size),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.bottleneck_size, self.out_dim, bias=False),)
        else:
            self.post_block = nn.Identity()

        self.reset_parameters()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.flatten(start_dim=X.ndim - len(self.in_shape))
        X = self.pre_block(X)
        X = self.hidden_block(X) + X
        X = self.post_block(X)
        X = X.unflatten(dim=-1, sizes=self.out_shape)
        return X

    def reset_parameters(self):
        weights_init(self, is_JL_init=self.is_JL_init)


class FlattenLinear(nn.Module):
    """
    Linear that can take a multi dimensional array as input and output . E.g. for predicting an image use
    `out_shape=(32,32,3)` and this will predict 32*32*3 and then reshape.

    Parameters
    ----------
    in_shape : tuple or int

    out_shape : tuple or int

    is_batchnorm_pre : bool, optional
        Whether to use a batchnorm layer at input. Note that this is simply a reparametrization
        of the following layer and is still linear.

    bottleneck_size : int, optional
        Whether to add a bottleneck in the linear layer, this is equivalent to constraining the linear layer to be
        low rank. The result will still be linear (no non linearity) but more efficient if input and
        output is very large.

    is_batchnorm_bottleneck : bool, optional
        Whether to add a batchnorm between the bottleneck (if there is one) and the last linear layer
        to improve training of two linear layers in a row (still linear).

    kwargs :
        Additional arguments to `torch.nn.Linear`.
    """

    def __init__(
        self, in_shape: Sequence[int], out_shape: Sequence[int], is_batchnorm_pre: bool=False,
        bottleneck_size : Optional[int]=None, is_batchnorm_bottleneck: bool =True,
        batchnorm_kwargs : dict = {},  is_JL_init: bool=True,
        **kwargs
    ) -> None:
        super().__init__()

        self.in_shape = [in_shape] if isinstance(in_shape, int) else in_shape
        self.out_shape = [out_shape] if isinstance(out_shape, int) else out_shape
        self.bottleneck_size = bottleneck_size
        self.is_batchnorm_pre = is_batchnorm_pre
        self.is_batchnorm_bottleneck = is_batchnorm_bottleneck
        self.is_JL_init = is_JL_init

        in_dim = prod(self.in_shape)
        out_dim = prod(self.out_shape)

        # TODO rm next line. this is for backward compatibility
        kwargs = {k: v for k, v in kwargs.items() if k != "is_batchnorm"}

        if self.is_batchnorm_pre:
            self.normalizer_pre = BatchNorm1d(in_dim, **batchnorm_kwargs)

        if self.bottleneck_size is not None:
            self.bottleneck = nn.Linear(in_dim, self.bottleneck_size, bias=False)
            in_dim = self.bottleneck_size

            if self.is_batchnorm_bottleneck:
                # and remove batchnorm_kwargs
                self.normalizer = BatchNorm1d(self.bottleneck_size, **batchnorm_kwargs)

        self.linear = nn.Linear(in_dim, out_dim, **kwargs)

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)
        if self.bottleneck_size is not None:
            if self.is_JL_init:
                johnson_lindenstrauss_init_(self.bottleneck)


    def forward_flatten(self, X: torch.Tensor) -> torch.Tensor:

        if self.is_batchnorm_pre:
            X = self.normalizer_pre(X)

        if self.bottleneck_size is not None:
            X = self.bottleneck(X)

            if self.is_batchnorm_bottleneck:
                X = self.normalizer(X)

        out = self.linear(X)

        return out


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # flattens in_shape
        X = X.flatten(start_dim=X.ndim - len(self.in_shape))

        out = self.forward_flatten(X)

        # unflattened out_shape
        out = out.unflatten(dim=-1, sizes=self.out_shape)
        return out

class FlattenCosine(FlattenLinear):
    """Cosine similarity between inputs and weights."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, bias=False, **kwargs)
        self.linear = nn.utils.weight_norm(self.linear)
        self.linear.weight_g.data.fill_(1)  # unit norm
        self.linear.weight_g.requires_grad = False  # don't optimize norm

    def reset_parameters(self):
        weights_init(self)

        try:
            self.linear.weight_g.data.fill_(1)
        except AttributeError:
            pass  # make sure ok  if call reset_param before weight_norm

    def forward_flatten(self, X: torch.Tensor) -> torch.Tensor:
        unit_X = F.normalize(X, dim=-1, p=2)
        return super().forward_flatten(unit_X)


class Flatten(nn.Flatten):
    """Flatten a representation."""

    def __init__(self, **kwargs) -> None:
        super().__init__(start_dim=1, end_dim=-1)


class Resizer(nn.Module):
    """Resize with a linear layer only if needed.

    Parameters
    ----------
    in_shape : tuple or int

    out_shape : tuple or int

    curr_out_dim : int

    kwargs :
        Additional arguments to `torch.nn.Linear`.
    """

    def __init__(self, curr_out_dim: int, out_shape: Sequence[int], **kwargs) -> None:
        super().__init__()
        self.out_dim = prod(out_shape)

        if curr_out_dim == self.out_dim:
            self.resizer = nn.Identity()
        else:
            self.resizer = nn.Linear(curr_out_dim, self.out_dim, **kwargs)

        self.reset_parameters()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.resizer(X)
        X = X.unflatten(dim=-1, sizes=self.out_shape)
        return X

    def reset_parameters(self):
        weights_init(self)
