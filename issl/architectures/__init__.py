from __future__ import annotations

from functools import partial
from typing import Any
import torch

from .basic import *
from .cnn import *

__all__ = ["get_Architecture"]


def get_Architecture(architecture: str, **kwargs) -> Any:
    """Return the (uninstantiated) correct architecture.

    Parameters
    ----------
    architecture : {"mlp", "linear", "resnet", "identity", "cosine"}.

    kwargs :
        Additional arguments to the Module.

    Return
    ------
    Architecture : uninstantiated nn.Module
        Architecture that can be instantiated by `Architecture(in_dim, out_shape)`
    """
    if architecture == "mlp":
        return partial(MLP, **kwargs)

    elif architecture == "identity":
        return partial(torch.nn.Identity, **kwargs)

    elif architecture == "linear":
        return partial(FlattenLinear, **kwargs)

    elif architecture == "cosine":
        return partial(FlattenCosine, **kwargs)

    elif architecture == "resnet":
        return partial(ResNet, **kwargs)

    else:
        raise ValueError(f"Unknown architecture={architecture}.")
