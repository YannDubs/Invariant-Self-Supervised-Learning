from __future__ import annotations

import math

import torch.nn as nn


def get_Normalization(norm_layer, dim=2):
    """Return the correct normalization layer.

    Parameters
    ----------
    norm_layer : callable or {"batchnorm", "identity"}U{any torch.nn layer}
        Layer to return.

    dim : int, optional
        Number of dimension of the input (e.g. 2 for images).
    """
    Batch_norms = [None, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
    if "batch" in norm_layer:
        Norm = Batch_norms[dim]
    elif norm_layer == "identity":
        Norm = nn.Identity
    elif isinstance(norm_layer, str):
        Norm = getattr(nn, norm_layer)
    else:
        Norm = norm_layer
    return Norm


def get_Activation(activation):
    """Return an uninstantiated activation that takes the number of channels as inputs.

    Parameters
    ----------
    activation : {"gdn"}U{any torch.nn activation}
        Activation to use.
    """
    return getattr(nn, activation)


def is_pow2(n: int) -> bool:
    """Check if a number is a power of 2."""
    return (n != 0) and (n & (n - 1) == 0)


def closest_pow(n: int, base: float = 2) -> float:
    """Return the closest (in log space) power of 2 from a number."""
    return base ** round(math.log(n, base))
