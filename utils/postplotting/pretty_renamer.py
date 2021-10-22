from __future__ import annotations

from utils.helpers import StrFormatter

__all__ = ["PRETTY_RENAMER"]

PRETTY_RENAMER = StrFormatter(
    exact_match={},
    substring_replace={
        # Math stuff
        "beta": r"$\beta$",
        # General
        "_": " ",
        "Resnet": "ResNet",
        "Lr": "Learning Rate",
    },
    to_upper=["Cifar10", "Mnist", "Mlp", "Adam"],
)
