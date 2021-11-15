from __future__ import annotations

from utils.helpers import StrFormatter

__all__ = ["PRETTY_RENAMER"]

PRETTY_RENAMER = StrFormatter(
    exact_match={},
    substring_replace={
        # Math stuff
        "beta": r"$\beta$",
        "calF": r"$\mathcal{F}$",
        " --": r"⁻",
        " ++": r"⁺",
        # General
        "_": " ",
        "Resnet": "ResNet",
        "Lr": "Learning Rate",
        "Test/Pred/": "",
        "Train/Pred/": "Train ",
        # Project specific
        "Accuracy Score": "Acc",
        "Std Gen Smallz": "Std. Gen. ISSL",
        "Cntr Stda": "Our Cont. ISSL",
        "Acc Agg Min": "Worst Acc.",
        # "Acc": "Accuracy",
        "K Labels": "Max. # Labels",
        "N Tasks": "# Tasks",
        "N Samples": "# Samples",
        "2.0": "Binary",
    },
    to_upper=["Cifar10", "Mnist", "Mlp", "Adam"],
)
