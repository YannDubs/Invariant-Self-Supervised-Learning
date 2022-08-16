from __future__ import annotations

from .img import *


def get_Datamodule(datamodule: str) -> type:
    """Return the correct uninstantiated datamodule."""
    datamodule = datamodule.lower()
    if datamodule == "cifar10":
        return Cifar10DataModule
    elif datamodule == "tinyimagenet":
        return TinyImagenetDataModule
    else:
        raise ValueError(f"Unknown datamodule: {datamodule}")
