from __future__ import annotations

import torch

from .contrastive import *
from .exact import *
from .generative import *
from .self_distillation import *

__all__ = ["get_loss_decodability"]


def get_loss_decodability(mode: str, **kwargs) -> torch.nn.Module:
    if mode == "contrastive":
        return ContrastiveISSL(**kwargs)
    elif mode == "generative":
        return GenerativeISSL(**kwargs)
    elif mode == "simsiam_self_distillation":
        return SimSiamSelfDistillationISSL(**kwargs)
    elif mode == "prior_self_distillation":
        return PriorSelfDistillationISSL(**kwargs)
    elif mode == "swav_self_distillation":
        return SwavSelfDistillationISSL(**kwargs)
    elif mode == "dino_self_distillation":
        return DinoSelfDistillationISSL(**kwargs)
    elif mode == "exact":
        return ExactISSL(**kwargs)
    else:
        raise ValueError(f"Unknown mode={mode}.")
