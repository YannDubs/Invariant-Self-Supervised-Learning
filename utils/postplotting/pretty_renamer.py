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
        "true": "True",
        "false": "False",
        "Resnet": "ResNet",
        "Lr": "Learning Rate",
        "Test/Pred/": "",
        "Pytorch Datarepr/": "",
        "Test/Pred Train/": "Train ",
        "Train/Pred/": "Train ",
        "Test/Repr/": "Test ",
        "Zdim": r"$\mathcal{Z}$ dim.",
        #"Zdim": r"Dimension $d$",
        "Pred": "Downstream Pred.",
        "Repr": "ISSL",
        "N Equiv": "# Equiv.", #r"$|\mathcal{X}/{\sim}|$",
        # Project specific
        "Acc ": "Acc. ",
        "Accuracy Score": "Acc.",
        "Std Gen Smallz": "Std. Gen. ISSL",
        "Cntr Stda": "Our Cont. ISSL",
        "Acc. Agg Min": "Worst Acc.",
        "Acc.": "Accuracy", # if want all long
        "Acc": "Accuracy", #might cause issues
        "K Labels": "Max. # Labels",
        "N Tasks": "# Tasks",
        "N Samples": "# Samples",
        "N Downstream Samples": "# Downstream Samples",
        "2.0": "Binary",
        "Dstl": "DISSL",
        "Cntr": "CISSL",
        "Decodability": "ISSL Loss",
        "Augmentation": "Aug.",
    },
    to_upper=["Cifar10", "Mnist", "Mlp", "Adam", "Dissl", "Cntr"],
)
