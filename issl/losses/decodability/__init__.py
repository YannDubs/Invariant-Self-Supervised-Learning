from __future__ import annotations

import torch

from .contrastive import *
from .exact import *
from .distillating import *

__all__ = ["get_loss_decodability"]


def get_loss_decodability(mode: str, **kwargs) -> torch.nn.Module:
    if mode == "simclr":
        return SimCLR(**kwargs)
    elif mode == "cissl":
        return CISSL(**kwargs)
    elif mode in ["dissl"]:
        return DISSL(**kwargs)
    elif mode in ["dino"]:
        return DINO(**kwargs)
    else:
        raise ValueError(f"Unknown mode={mode}.")
