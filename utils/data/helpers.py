from __future__ import annotations

import logging
import shutil
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Optional, Union

from sklearn.model_selection import train_test_split
from torch import randperm
from torch.utils.data import Dataset, Subset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose
from tqdm import tqdm
from torch._utils import _accumulate

from issl.helpers import to_numpy

logger = logging.getLogger(__name__)

def random_split_cache(dataset: Dataset, lengths: Sequence[int], generator: Any) -> list[CachedSubset]:
    """Like `random_split` but returns CachedSubset instead of Subset

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()
    return [CachedSubset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]


def subset2dataset(subset):
    """Return the underlying dataset. Contains val when val is subset of train."""
    dataset = subset
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset

class CachedSubset(Subset):
    """Subset a dataset on idcs with optional possibility of caching the data in memory.

    Parameters
    ----------
    dataset : Dataset
        Dataset to subset.

    idcs : sequence of int
        Indices in the whole set selected for subset.

    is_cache_data : bool, optional
        Whether to cache the data in memory.
    """
    def __init__(self, dataset: Dataset, indices: Sequence[int], is_cache_data: bool=False):
        super().__init__(dataset, indices)

        if hasattr(dataset, "cached_targets"):
            self.cached_targets = self.dataset.cached_targets[indices]

        if is_cache_data:
            self.cache_data_()

    def cache_data_(self, idcs=None, **kwargs):
        all_idcs = to_numpy(self.indices)
        if idcs is not None:
            all_idcs = all_idcs[idcs]

        self.dataset.cache_data_(idcs=all_idcs, **kwargs)

class BalancedSubset(CachedSubset):
    """Split the dataset into a subset with possibility of stratifying.

    Parameters
    ----------
    dataset : Dataset
        Dataset to subset.

    size : float or int, optional
            If float, should be between 0.0 and 1.0 and represent the proportion of
            the dataset to retain. If int, represents the absolute number or examples.

    is_stratify : bool, optional
        Whether to stratify splits based on class label. Only works if dataset has
        a `targets` attribute are loaded.

    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        dataset: Dataset,
        size: float = 0.1,
        stratify: Any = None,
        seed: Optional[int] = 123,
    ):
        _, subset_idcs = train_test_split(
            range(len(dataset)), stratify=stratify, test_size=size, random_state=seed
        )
        super().__init__(dataset, subset_idcs)


class ImgAugmentor:
    def __init__(
        self,
        base_transform: Callable,
        augmentations: Sequence[str],
        choices_PIL: dict,
        choices_tens: dict,
    ):
        PIL_augment, tensor_augment = [], []
        for aug in augmentations:
            if aug in choices_PIL:
                PIL_augment += [choices_PIL[aug]]
            elif aug in choices_tens:
                tensor_augment += [choices_tens[aug]]
            else:
                raise ValueError(f"Unknown `augmentation={aug}`.")

        self.PIL_aug = Compose(PIL_augment)
        self.base_transform = base_transform
        self.tensor_aug = Compose(tensor_augment)

    def __call__(self, img):
        img = self.PIL_aug(img)
        img = self.base_transform(img)
        img = self.tensor_aug(img)
        return img


def image_loader(path):
    """Load image and returns PIL."""
    if isinstance(path, Path):
        path = str(path.resolve())
    return default_loader(path)


class DownloadProgressBar(tqdm):
    """Progress bar for downloading files."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


# Modified from https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
def download_url(url, save_dir):
    """Download a url to `save_dir`."""
    filename = url.split("/")[-1]
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:

        urllib.request.urlretrieve(
            url, filename=save_dir / filename, reporthook=t.update_to
        )


def remove_rf(path: Union[str, Path], not_exist_ok: bool = False) -> None:
    """Remove a file or a folder"""
    path = Path(path)

    if not path.exists() and not_exist_ok:
        return

    if path.is_file():
        path.unlink()
    elif path.is_dir:
        shutil.rmtree(path)

def unzip(filename: Union[str, Path], is_rm: bool = True) -> None:
    """Unzip file and optionally removes it."""
    filename = Path(filename)
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall(filename.parent)
        if is_rm:
            filename.unlink()


def _get_img_pool(i, loader):
    return loader(i)[0] if i is not None else None
