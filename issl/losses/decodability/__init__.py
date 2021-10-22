from __future__ import annotations

import torch

from .contrastive import *
from .generative import *
from .self_distillation import *

__all__ = ["get_loss_decodability"]


def get_loss_decodability(mode: str, **kwargs) -> torch.nn.Module:
    if mode == "contrastive":
        return ContrastiveISSL(**kwargs)
    elif mode == "generative":
        return GenerativeISSL(**kwargs)
    elif mode == "self_distillation":
        return SelfDistillationISSL(**kwargs)
    elif mode == "prior_self_distillation":
        return PriorSelfDistillationISSL(**kwargs)
    elif mode == "cluster_self_distillation":
        return ClusterSelfDistillationISSL(**kwargs)
    else:
        raise ValueError(f"Unknown mode={mode}.")
