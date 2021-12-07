from __future__ import annotations

from utils.helpers import StrFormatter

__all__ = ["PRETTY_RENAMER"]

PRETTY_RENAMER = StrFormatter(
    exact_match={},
    substring_replace={
        # Math stuff
        "beta": r"$\beta$",
        "calFissl": r"$\mathcal{F}_{issl}$",
        "calFpred": r"$\mathcal{F}_{pred}$",
        "calF --": r"$\mathcal{F}^{-}$",
        "calF ++": r"$\mathcal{F}^{+}$",
        "calF": r"$\mathcal{F}$",
        " --": r"⁻",
        " ++": r"⁺",
        # General
        "_": " ",
        "Resnet": "ResNet",
        "Lr": "Learning Rate",
        "Test/Pred/": "",
        "Train/Pred/": "Train ",
        "Zdim": r"$\mathcal{Z}$ dim.",
        "Pred": "Downstream Pred.",
        "Repr": "ISSL",
        # Project specific
        "Acc ": "Acc. ",
        "Accuracy Score": "Acc.",
        "Std Gen Smallz": "Std. Gen. ISSL",
        "Cntr Stda": "Our Cont. ISSL",
        "Acc. Agg Min": "Worst Acc.",
        # "Acc": "Accuracy",
        "K Labels": "Max. # Labels",
        "N Tasks": "# Tasks",
        "N Samples": "# Samples",
        "2.0": "Binary",
    },
    to_upper=["Cifar10", "Mnist", "Mlp", "Adam"],
)
