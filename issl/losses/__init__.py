from __future__ import annotations

import torch

from .contrastive import SimCLR, CISSL
from .dino import DINO
from .dissl import DISSL


def get_loss_decodability(mode: str, encoder=None, **kwargs) -> torch.nn.Module:
    mode = mode.lower()
    if mode == "simclr":
        return SimCLR(**kwargs)
    elif mode == "cissl":
        return CISSL(**kwargs)
    elif mode == "dino":
        return DINO(encoder=encoder, **kwargs)
    elif mode == "dissl":
        return DISSL(**kwargs)
    else:
        raise ValueError(f"Unknown mode={mode}.")