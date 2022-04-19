from __future__ import annotations

import torch

from .contrastive import *
from .exact import *
from .distillating import *

__all__ = ["get_loss_decodability"]


def get_loss_decodability(mode: str, **kwargs) -> torch.nn.Module:
    if mode == "contrastive":
        return ContrastiveISSL(**kwargs)
    elif mode in ["simsiam_self_distillation", "simsiam"]:
        return SimSiamISSL(**kwargs)
    elif mode in ["assign_self_distillation", "distillating"]:
        return DistillatingISSL(**kwargs)
    elif mode in ["swav_self_distillation", "swav"]:
        return SwavISSL(**kwargs)
    elif mode in ["dino_self_distillation", "dino"]:
        return DinoISSL(**kwargs)
    elif mode == "exact":
        return ExactISSL(**kwargs)
    else:
        raise ValueError(f"Unknown mode={mode}.")
