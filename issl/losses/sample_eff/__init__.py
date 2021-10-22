import torch

from .entropy import *
from .information import *

__all__ = ["get_loss_sample_eff"]


def get_loss_sample_eff(mode: str, **kwargs) -> torch.nn.Module:
    if mode == "entropy":
        pass
    elif mode == "information":
        pass
    else:
        raise ValueError(f"Unknown mode={mode}.")
