from __future__ import annotations

from functools import partial
from typing import Any

from .basic import *
from .cnn import *
from .pretrained_ssl import *
from .vit import *

__all__ = ["get_Architecture"]


def get_Architecture(architecture: Any, **kwargs) -> Any:
    """Return the (uninstantiated) correct architecture.

    Parameters
    ----------
    architecture : {"mlp", "linear", "resnet", "identity", "cnn_unflatten", "vit", "cnn", "clip_vitb16", "clip_vitb32",
                    "clip_rn50", "dino_vitb16", "dino_rn50", "dino_vits16", "simclr_rn50", "swav_rn50"} or callable.
        If callable will return it.

    kwargs :
        Additional arguments to the Module.

    Return
    ------
    Architecture : uninstantiated nn.Module
        Architecture that can be instantiated by `Architecture(in_shape, out_shape)`
    """
    if not isinstance(architecture, str):
        return architecture

    if architecture == "mlp":
        return partial(FlattenMLP, **kwargs)

    elif architecture == "identity":
        return partial(torch.nn.Identity, **kwargs)

    elif architecture == "flatten":
        return partial(Flatten, **kwargs)

    elif architecture == "linear":
        return partial(FlattenLinear, **kwargs)

    elif architecture == "cosine":
        return partial(FlattenCosine, **kwargs)

    elif architecture == "cosine_mlp":
        return partial(FlattenMLP, is_cosine_last=True, **kwargs)

    elif architecture == "resnet":
        return partial(ResNet, **kwargs)

    elif architecture == "resnet_transpose":
        return partial(ResNetTranspose, **kwargs)

    elif architecture == "cnn":
        return partial(CNN, **kwargs)

    elif architecture == "cnn_unflatten":
        return partial(CNNUnflatten, **kwargs)

    elif architecture == "vit":
        return partial(ViT, **kwargs)

    elif ("dino" in architecture) or ("clip" in architecture):
        return partial(PretrainedSSL, model=architecture, **kwargs)

    elif architecture in ["swav_rn50", "swav_rn50"]:
        return partial(PretrainedSSL, model=architecture, **kwargs)

    else:
        raise ValueError(f"Unknown architecture={architecture}.")
