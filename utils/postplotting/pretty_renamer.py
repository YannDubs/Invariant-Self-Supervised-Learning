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
        "Test/Pred/": "",
        # Project specific
        "Std Gen Smallz": "Std. Gen. ISSL",
        "Cntr Stda": "Our Cont. ISSL",
        "Accuracy Score Agg Min": "Worst Acc.",
        "K Labels": "Max. # Labels",
        "N Tasks": "# Tasks",
        "2.0": "Binary",
    },
    to_upper=["Cifar10", "Mnist", "Mlp", "Adam"],
)
