from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from issl.architectures.helpers import get_Activation, get_Normalization
from issl.helpers import (batch_flatten, batch_unflatten, prod,
                          weights_init, BatchNorm1d)

# TODO clean you don't need flattens anymore
__all__ = ["MLP", "FlattenLinear", "FlattenCosine"]


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

    is_cosine : bool, optional
        Whether the last layer should be cosine similarity instead of linear.

    kwargs:
        Additional arguments of the last linear layer which is a `FlattenLinear`.
    """

    def __init__(
        self,
        in_shape: int,# temporary should be dim
        out_shape: int, # temporary should be dim
        n_hid_layers: int = 2,
        hid_dim: int = 2048,
        norm_layer: str = "batch",
        activation: str = "ReLU",
        is_cosine: bool= False,
        **kwargs
    ) -> None:
        super().__init__()

        self.in_dim = in_shape
        self.out_dim = out_shape
        self.n_hid_layers = n_hid_layers
        self.hid_dim = hid_dim
        Activation = get_Activation(activation)
        Norm = get_Normalization(norm_layer, dim=1)
        # don't use bias with batch_norm https://twitter.com/karpathy/status/1013245864570073090?l...
        bias_hidden = Norm == nn.Identity
        self.is_cosine = is_cosine

        self.pre_block = nn.Sequential(
            nn.Linear(self.in_dim, hid_dim, bias=bias_hidden),
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
        self.post_block = PostBlock(hid_dim, self.out_dim, **kwargs)

        self.reset_parameters()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # flatten to make for normalizing layer => only 2 dim
        X, shape = batch_flatten(X)
        X = self.pre_block(X)
        X = self.hidden_block(X)
        X = self.post_block(X)
        X = batch_unflatten(X, shape)
        return X

    def reset_parameters(self):
        weights_init(self)



class FlattenLinear(nn.Module):
    """
    Linear that can take a multi dimensional array as input and output . E.g. for predicting an image use
    `out_shape=(32,32,3)` and this will predict 32*32*3 and then reshape.

    Parameters
    ----------
    in_shape : tuple or int

    out_shape : tuple or int

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
        self,
        in_shape: Sequence[int],
        out_shape: Sequence[int],
        bottleneck_size : Optional[int]=None,
        is_batchnorm_bottleneck: bool =True,
        **kwargs
    ) -> None:
        super().__init__()

        self.in_shape = [in_shape] if isinstance(in_shape, int) else in_shape
        self.out_shape = [out_shape] if isinstance(out_shape, int) else out_shape
        self.bottleneck_size = bottleneck_size
        self.is_batchnorm_bottleneck = is_batchnorm_bottleneck

        in_dim = prod(self.in_shape)
        out_dim = prod(self.out_shape)

        if self.bottleneck_size is not None:
            self.bottleneck = nn.Linear(in_dim, self.bottleneck_size, bias=False)
            in_dim = self.bottleneck_size

            if self.is_batchnorm_bottleneck:
                self.normalizer = BatchNorm1d(self.bottleneck_size)

        self.linear = nn.Linear(in_dim, out_dim, **kwargs)

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)


    def forward_flatten(self, X: torch.Tensor) -> torch.Tensor:

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

