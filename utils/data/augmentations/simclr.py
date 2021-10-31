from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torchvision import transforms

try:
    import cv2
except ImportError:
    pass

__all__ = ["get_simclr_augmentations", "get_finetune_augmentations"]


# taken from pl_bolts.models.self_supervised.simclr.transforms
def get_simclr_augmentations(dataset: str, input_height: int) -> Callable[..., Any]:
    if dataset == "imagenet":
        jitter_strength = 1.0
        gaussian_blur = True
    elif dataset == "cifar10":
        jitter_strength = 0.5
        gaussian_blur = False
    else:
        raise ValueError(f"Unknown dataset={dataset} for simclr augmentations.")

    color_jitter = transforms.ColorJitter(
        0.8 * jitter_strength,
        0.8 * jitter_strength,
        0.8 * jitter_strength,
        0.2 * jitter_strength,
    )

    data_transforms = [
        transforms.RandomResizedCrop(size=input_height),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ]

    if gaussian_blur:
        kernel_size = int(0.1 * input_height)
        if kernel_size % 2 == 0:
            kernel_size += 1

        data_transforms.append(
            transforms.RandomApply([transforms.GaussianBlur(kernel_size)], p=0.5)
        )

    data_transforms = transforms.Compose(data_transforms)

    return data_transforms


# taken from pl_bolts.models.self_supervised.simclr.transforms
def get_finetune_augmentations(input_height: int) -> Callable[..., Any]:
    jitter_strength = 1.0

    color_jitter = transforms.ColorJitter(
        0.8 * jitter_strength,
        0.8 * jitter_strength,
        0.8 * jitter_strength,
        0.2 * jitter_strength,
    )

    data_transforms = [
        transforms.Resize(int(input_height * 1.1)),
        transforms.CenterCrop(input_height),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ]

    data_transforms = transforms.Compose(data_transforms)

    return data_transforms
