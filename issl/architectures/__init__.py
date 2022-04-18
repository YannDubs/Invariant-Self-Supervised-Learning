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
    architecture : {"mlp", "linear", "resnet", "identity", "cnn_unflatten", "vit", "cnn", "clip_vitb16", "clip_vitb32",
                    "clip_rn50", "dino_vitb16", "dino_rn50", "dino_vits16", "simclr_rn50", "swav_rn50"}.

    kwargs :
        Additional arguments to the Module.

    Return
    ------
    Architecture : uninstantiated nn.Module
        Architecture that can be instantiated by `Architecture(in_shape, out_shape)`
    """
    if architecture == "mlp":
        return partial(FlattenMLP, **kwargs)

    if architecture == "mll":
        return partial(FlattenMLL, **kwargs)

    elif architecture == "identity":
        return partial(torch.nn.Identity, **kwargs)

    elif architecture == "flatten":
        return partial(Flatten, **kwargs)

    elif architecture == "linear":
        return partial(FlattenLinear, **kwargs)

    elif architecture == "cosine":
        return partial(FlattenCosine, **kwargs)

    elif architecture == "convnext":
        return partial(ConvNext, **kwargs)

    elif architecture == "resnet":
        return partial(ResNet, **kwargs)

    elif architecture == "resnet_transpose":
        return partial(ResNetTranspose, **kwargs)

    else:
        raise ValueError(f"Unknown architecture={architecture}.")
