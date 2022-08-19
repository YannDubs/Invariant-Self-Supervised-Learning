from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

from torchvision import transforms


__all__ = ["get_simclr_augmentations"]


# taken from pl_bolts.models.self_supervised.simclr.transforms and
# https://github.com/htdt/self-supervised/blob/d24f3c722ac4e945161a9cd8c830bf912403a8d7/cfg.py
from torchvision.transforms import InterpolationMode


def get_simclr_augmentations(
    input_height: int,
    dataset: Optional[str] = None,
    strength : float =1.0,  # multiply all numeric values
    is_force_blur : bool=False
) -> Callable[..., Any]:
    dataset = dataset.lower()

    if dataset in ["stl10", "tiny-imagenet-200"]:
        gaussian_blur = False
        col_s1 = 0.4 * strength
        col_s2 = 0.1 * strength
        crop_s = 0.2 / strength
        p_gray = 0.1 * strength
    elif "cifar10" in dataset:
        gaussian_blur = False
        col_s1 = 0.2 * strength
        col_s2 = 0.05 * strength
        crop_s = 0.2 / strength
        p_gray = 0.1 * strength
    elif dataset == "imagenet":
        gaussian_blur = True
        col_s1 = 0.8 * strength
        col_s2 = 0.2 * strength
        crop_s = 0.08 / strength
        p_gray = 0.2 * strength
    else:
        raise ValueError(f"Unknown dataset={dataset} for simclr augmentations.")

    gaussian_blur = gaussian_blur or is_force_blur

    color_jitter = transforms.ColorJitter(col_s1, col_s1, col_s1, col_s2)
    data_transforms = [
        transforms.RandomResizedCrop(
            size=input_height,
            scale=(crop_s, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=p_gray),
    ]

    if gaussian_blur:
        # need to check if correct implementation because most people used different gaussian blur
        kernel_size = int(0.1 * input_height)
        if kernel_size % 2 == 0:
            kernel_size += 1

        data_transforms.append(
            transforms.RandomApply([transforms.GaussianBlur(kernel_size)], p=0.5)
        )

    data_transforms = transforms.Compose(data_transforms)

    return data_transforms
