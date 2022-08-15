from __future__ import annotations

import torch

from .contrastive import *
from .dissl import *
from .dino import *

__all__ = ["get_loss_decodability"]


def get_loss_decodability(mode: str, encoder=None, **kwargs) -> torch.nn.Module:
    if mode == "simclr":
        return SimCLR(**kwargs)
    elif mode == "cissl":
        return CISSL(**kwargs)
    elif mode in ["dissl"]:
        return DISSL(**kwargs)
    elif mode in ["dino"]:
        return DINO(encoder=encoder, **kwargs)
    else:
        raise ValueError(f"Unknown mode={mode}.")
