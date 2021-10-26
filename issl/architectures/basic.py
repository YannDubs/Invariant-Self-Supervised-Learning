from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from issl.architectures.helpers import get_Activation, get_Normalization
from issl.helpers import batch_flatten, batch_unflatten, prod, weights_init

__all__ = ["FlattenMLP", "FlattenLinear", "Resizer", "Flatten"]


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

    dropout_p : float, optional
        Dropout rate.

    is_skip_hidden : bool, optional
        Whether to skip all the hidden layers with a residual connection.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_hid_layers: int = 1,
        hid_dim: int = 1024,
        norm_layer: str = "identity",
        activation: str = "ReLU",
        dropout_p: float = 0,
        is_skip_hidden: bool = False,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_hid_layers = n_hid_layers
        self.hid_dim = hid_dim
        Activation = get_Activation(activation)
        Dropout = nn.Dropout if dropout_p > 0 else nn.Identity
        Norm = get_Normalization(norm_layer, dim=1)
        # don't use bias with batch_norm https://twitter.com/karpathy/status/1013245864570073090?l...
        is_bias = Norm == nn.Identity
        self.is_skip_hidden = is_skip_hidden

        self.pre_block = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=is_bias),
            Norm(hid_dim),
            Activation(),
            Dropout(p=dropout_p),
        )
        layers = []
        for _ in range(1, n_hid_layers):
            layers += [
                nn.Linear(hid_dim, hid_dim, bias=is_bias),
                Norm(hid_dim),
                Activation(),
                Dropout(p=dropout_p),
            ]
        self.hidden_block = nn.Sequential(*layers)
        self.post_block = nn.Linear(hid_dim, out_dim)

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


class FlattenLinear(nn.Linear):
    """
    Linear that can take a multi dimensional array as input and output . E.g. for predicting an image use
    `out_shape=(32,32,3)` and this will predict 32*32*3 and then reshape.

    Parameters
    ----------
    in_shape : tuple or int

    out_shape : tuple or int

    kwargs :
        Additional arguments to `torch.nn.Linear`.
    """

    def __init__(
        self, in_shape: Sequence[int], out_shape: Sequence[int], **kwargs
    ) -> None:
        self.in_shape = [in_shape] if isinstance(in_shape, int) else in_shape
        self.out_shape = [out_shape] if isinstance(out_shape, int) else out_shape

        in_dim = prod(self.in_shape)
        out_dim = prod(self.out_shape)
        super().__init__(in_features=in_dim, out_features=out_dim, **kwargs)

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
