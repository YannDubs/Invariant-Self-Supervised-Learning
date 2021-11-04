from __future__ import annotations

from .img import *


def get_Datamodule(datamodule: str) -> type:
    """Return the correct uninstantiated datamodule."""
    datamodule = datamodule.lower()
    if datamodule == "cifar10":
        return Cifar10DataModule
    if datamodule == "cifar100":
        return Cifar100DataModule
    elif datamodule == "mnist":
        return MnistDataModule
    elif datamodule == "imagenet":
        return ImagenetDataModule
    elif datamodule == "tinyimagenet":
        return TinyImagenetDataModule
    elif datamodule == "stl10":
        return STL10DataModule
    elif datamodule == "stl10_unlabeled":
        return STL10UnlabeledDataModule
    elif datamodule == "coco":
        return CocoClipDataModule
    elif datamodule == "food101":
        return Food101DataModule
    elif datamodule == "cars196":
        return Cars196DataModule
    elif datamodule == "pets37":
        return Pets37DataModule
    elif datamodule == "caltech101":  # might drop because next version
        return Caltech101DataModule
    elif datamodule == "pcam":
        return PCamDataModule
    else:
        raise ValueError(f"Unknown datamodule: {datamodule}")
