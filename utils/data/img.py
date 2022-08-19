from __future__ import annotations

import abc
import copy
import logging
from collections.abc import Callable, Sequence
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
from torchvision.transforms import InterpolationMode
import torch
from torchvision.datasets import CIFAR10,ImageFolder
from torchvision.transforms.functional_pil import _is_pil_image
from torchvision.transforms import (
    ColorJitter,
    Compose,
    RandomAffine,
    RandomApply,
    RandomErasing,
    RandomGrayscale,
    RandomResizedCrop,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)

from issl.helpers import Normalizer, check_import, tmp_seed, to_numpy, int_or_ratio, file_cache
from .augmentations import get_simclr_augmentations
from .base import ISSLDataModule, ISSLDataset
from .helpers import (
    ImgAugmentor,
    _get_img_pool, download_url,
    unzip,
    random_split_cache
)

import utils.helpers  # avoids circular imports



EXIST_DATA = "data_exist.txt"
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".JPEG",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)
logger = logging.getLogger(__name__)

__all__ = [
    "Cifar10DataModule",
    "TinyImagenetDataModule",
]


### Base Classes ###


class ISSLImgDataset(ISSLDataset):
    """Base class for image datasets used for lossy compression but lossless predictions.

    Parameters
    -----------
    a_augmentations : set of str, optional
        Augmentations that should be used to construct the axillary target, i.e., p(A|img). I.e. this should define the
        coarsest possible equivalence relation with respect to which to be invariant. It can be a set of augmentations
        (see self.augmentations) and/or "label" or "perm_label" or "data-standard" or "coarser". In the "label" case
        it samples randomly an image with the same label (slow if too many labels). "perm_label" then similar but it
        randomly samples an image with a different (but fixed => permutation) label. "data-standard" appends the standard
        augmentation for that dataset. "coarser" use a coarser grain equivalence class than the actual one, specifically,
        targets modulo 2.

    train_x_augmentations : set of str or "a_augmentations" , optional
        Augmentations to use for the source input, i.e., p(X|img). I.e. standard augmentations that are
        used to essentially increase the dataset size. This is different from p(A|img). Can be a set of string as in
        `a_augmentations`. In the latter case, will use the same as `"a_augmentations"` which is
        standard in ISSL but not theoretically necessary. Note that this cannot be "label" or "perm_label".

    val_x_augmentations : set of str or "train_x_augmentations" or "a_augmentations", optional
        list of augmentation to use during evaluation.

    is_normalize : bool, optional
        Whether to normalize all input images. Only for colored images. If True, you should ensure
        that `MEAN` and `STD` and `get_normalization` and `undo_normalization` in `issl.helpers`
        can normalize your data.

    normalization : str, optional
        What dataset to use for the normalization. If `None`, uses the default from the dataset.
        Only used if `is_normalize`.

    base_resize : {"resize", None}, optional
        What resizing to apply. If "resize" uses the same standard resizing during train and test.
        If "scale_crop_eval" then (ONLY) during test first up scale to 1.1*size and then center crop (this
        is used by SimCLR).

    curr_split : str, optional
        Which data split you are considering.

    is_shuffle_targets : bool, optional
        Shuffle targets which is useful if want labels to be uncorrelated with inputs.
        
    is_shuffle_Mx : bool, optional
        Shuffle the maximal invariants.

    n_sub_label : int, optional
        Number of sub equivalence classes to augment to when using `a_augmentations=label` or `a_augmentations=permlabel`.
        Specifically instead of augmenting uniformly to images inside the label class, will do so in a subpartition of the
        label class of size. There will be n_sub_label subpartitions. This is useful to know exactly the number of
        equivalence classes you are learning, while being somewhat realistic.

    simclr_aug_strength : float, optional
        Strength of standard simCLR augmentations.

    kwargs:
        Additional arguments to `ISSLDataset`.
    """

    def __init__(
        self,
        *args,
        a_augmentations: Sequence[str] = [],
        train_x_augmentations: Union[Sequence[str], str] = [],
        val_x_augmentations: Union[Sequence[str], str] = [],
        is_normalize: bool = True,
        normalization: Optional[str] = None,
        base_resize: str = "resize",
        curr_split: str = "train",
        is_shuffle_targets: bool = False,
        is_shuffle_Mx: bool = False,
        n_sub_label: int = 1,
        simclr_aug_strength: float = 1.0,
        **kwargs,
    ):
        self.base_resize = base_resize
        self.n_Mxs = -1  # tmp until gets allocated

        super().__init__(
            *args,
            is_normalize=is_normalize,
            normalization=normalization,
            a_augmentations=a_augmentations,
            **kwargs,
        )

        self.is_Mx_aug = False
        self.is_coarser_Mx = False
        self.perm_Mx = None
        self.n_sub_label = n_sub_label
        self.is_shuffle_targets = is_shuffle_targets
        self.is_shuffle_Mx = is_shuffle_Mx
        self.simclr_aug_strength = simclr_aug_strength

        self.cache_targets_()  # cache only targets

        if self.is_shuffle_targets:
            # need to shuffle before computing Mx
            self.shuffle_targets_()

        if "label" in self.a_augmentations:
            self.is_Mx_aug = True
            self.a_augmentations.remove("label")
        if "perm_label" in self.a_augmentations:
            assert not self.is_Mx_aug, "cannot have `label` and `perm_label`"
            self.is_Mx_aug = True
            # Mx permuter simply increases by 1. Attention: self.n_Mxs must be computed later
            self.perm_Mx = lambda x: (x + 1) % self.n_Mxs
            self.a_augmentations.remove("perm_label")
        if "coarser" in self.a_augmentations:
            self.is_coarser_Mx = True
            self.is_Mx_aug = True
            self.a_augmentations.remove("coarser")

        self.train_x_augmentations = train_x_augmentations
        self.val_x_augmentations = val_x_augmentations

        self.curr_split = curr_split
        self.n_Mxs = len(np.unique(self.Mxs))  # also precompute Mx

    @property
    def curr_split(self) -> str:
        """Return the current split."""
        return self._curr_split

    @curr_split.setter
    def curr_split(self, value: str) -> None:
        """Update the current split. Also reloads correct transformation as they are split dependent."""
        self._curr_split = value

        # when updating the split has to also reload the transformations
        self.base_transform = self.get_base_transform()

        self.sample_p_AlI = self.get_augmentor(self.a_augmentations)
        self.sample_p_XlI_train = self.get_augmentor(self.train_x_augmentations)
        self.sample_p_XlI_valid = self.get_augmentor(self.val_x_augmentations)

        if self.is_train:
            self.sample_p_XlI = self.sample_p_XlI_train
        else:
            self.sample_p_XlI = self.sample_p_XlI_valid

    @property
    def is_train(self) -> bool:
        """Whether considering training split."""
        return self.curr_split == "train"

    def cache_targets_(self):
        """Cache the targets as numpy array."""

        # NOTE : this should be overriden if targets already precomputed, because slow for large datasets
        # eg Imagenet
        logger.info("Caching targets.")
        self.cached_targets = np.stack(
            [to_numpy(self.get_img_target(i)[1]) for i in range(self._tmp_len)]
        )

    def cache_data_(self, idcs : Sequence[int]=None, num_workers : int=0):
        """Cache the images as list of PIL."""
        all_idcs = np.arange(self._tmp_len, dtype=object)
        if idcs is not None:
            mask = np.ones_like(all_idcs, dtype=bool)
            mask[to_numpy(idcs)] = False
            all_idcs[mask] = None

        # pool.map doesn't allow local functions
        get_img = partial(_get_img_pool, loader=self.get_img_target)

        if idcs is not None or len(all_idcs) < 1000:
            logger.info(f"Using only 1 worker to avoid freezing because of len={len(all_idcs)} or idcs given.")
            num_workers = 0

        if num_workers > 0:
            with Pool(num_workers) as pool:
                cached_data = pool.map(get_img, all_idcs)
        else:
            cached_data = [get_img(i) for i in all_idcs ]

        assert _is_pil_image(next(img for img in cached_data if img is not None)) # all PIL
        self.del_cached_data_()  # delete previous cached data before assigning
        self.cached_data = cached_data
        self.is_cache_data = True

    def shuffle_targets_(self):
        with tmp_seed(self.seed):
            np.random.shuffle(self.cached_targets)  # inplace

    @property
    def Mxs(self) -> np.ndarray:
        """Return an np.array of M(X), one for each example."""
        if hasattr(self, "_Mxs"):
            return self._Mxs

        if self.is_Mx_aug:
            targets = self.cached_targets
            if self.n_sub_label > 1:
                assert len(self) % self.n_sub_label == 0
                with tmp_seed(self.seed):
                    self._Mxs = targets * self.n_sub_label
                    for t in np.unique(targets):
                        idcs_t = targets == t
                        sub_labels_t = np.random.randint(
                            self.n_sub_label, size=idcs_t.sum()
                        )
                        if self.is_train:
                            # ensure that at least one ex per sub label during training
                            assert len(sub_labels_t) > self.n_sub_label
                            sub_labels_t[: self.n_sub_label] = np.arange(
                                self.n_sub_label
                            )
                            np.random.shuffle(sub_labels_t)

                        self._Mxs[idcs_t] += sub_labels_t
                    # the remainder (i.e. Mx % n_sub_label) will give the sub class
                    # the main part (i.e. Mx // n_sub_label) will give the targets
            else:
                self._Mxs = targets
        else:
            self._Mxs = np.array(range(len(self)))

        if self.is_coarser_Mx:
            targets = self.cached_targets
            self._Mxs = targets % 2

        if self.is_shuffle_Mx:
            with tmp_seed(self.seed):
                np.random.shuffle(self._Mxs)

        return self._Mxs

    def set_eval_(self):
        self.curr_split = "test"


    def get_img_target_cached(self, index: int) -> tuple[Any, npt.ArrayLike]:
        """Return the unaugmented image (in PIL format) and target. With possibility of caching."""
        if self.is_cache_data:
            return self.cached_data[index], self.cached_targets[index]
        else:
            return self.get_img_target(index)


    @abc.abstractmethod
    def get_img_target(self, index: int) -> tuple[Any, npt.ArrayLike]:
        """Return the unaugmented image (in PIL format) and target."""
        ...

    @property
    @abc.abstractmethod
    def standard_augmentations(self) -> list[str]:
        """Return the standard augmentations for the dataset."""
        ...

    @classmethod  # class method property does not work before python 3.9
    def get_available_splits(cls) -> list[str]:
        return ["train", "test"]

    def get_augmentor(self, augmentations: Union[Sequence[str], str]) -> Callable:
        """Return an augmentor that can be called with augmentor(X)."""

        if isinstance(augmentations, str):
            # assumes that the other augmenters have already been initialized
            if augmentations == "a_augmentations":
                return self.sample_p_AlI
            elif augmentations == "train_x_augmentations":
                return self.sample_p_XlI_train
            else:
                raise ValueError(f"Unknown str augmentor={augmentations}.")

        if "data-standard" in augmentations:
            # appends the data standard augmentations
            augmentations = list(augmentations)
            augmentations.remove("data-standard")
            augmentations += self.standard_augmentations

        choices_PIL, choices_tens = (
            self.augmentations["PIL"],
            self.augmentations["tensor"],
        )

        augmentor = ImgAugmentor(
            self.base_transform, augmentations, choices_PIL, choices_tens
        )
        return augmentor

    def get_x_target_Mx(self, index: int) -> tuple[Any, Any, Any]:
        """Return the correct example, target, and maximal invariant."""
        img, target = self.get_img_target_cached(index)

        x = self.sample_p_XlI(img)

        Mx = self.Mxs[index]

        return x, target, Mx

    @property
    def augmentations(self) -> dict[str, dict[str, Callable[..., Any]]]:
        """
        Return a dictionary of dictionaries containing all possible augmentations of interest.
        first dictionary say which kind of data they act on.
        """
        shape = self.shapes["input"]

        augmentations = dict(
            PIL={
                "y-translation": RandomAffine(0, translate=(0, 0.25)),
                "x-translation": RandomAffine(0, translate=(0.25, 0)),
                "scale": RandomAffine(0, scale=(0.6, 1.4)),
                "color": RandomApply(
                    [ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)], p=0.8
                ),
                "gray": RandomGrayscale(p=0.2),
                "resize-crop": RandomResizedCrop(
                    size=(shape[1], shape[2]),
                    scale=(0.2, 1.0),
                    ratio=(3 / 4, 4 / 3),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                "vflip": RandomVerticalFlip(p=0.5),
                "simclr-imagenet": get_simclr_augmentations(
                    shape[-1], dataset="imagenet", strength=self.simclr_aug_strength
                ),
            },
            tensor={"erasing": RandomErasing(value=0.5),},
        )

        try:
            # might not be possible depending on the dataset
            augmentations["PIL"]["simclr"] = get_simclr_augmentations(
                shape[-1], dataset=self.dataset_name, strength=self.simclr_aug_strength
            )
            augmentations["PIL"]["blured_simclr"] = get_simclr_augmentations(
                shape[-1], dataset=self.dataset_name, strength=self.simclr_aug_strength, is_force_blur=True,
            )
        except ValueError:
            logger.debug(f"Could not load dataset-specific augmentation for {self.dataset_name}.")

        return augmentations

    def get_img_from_Mx(self, Mx: int) -> Any:
        """Sample a random image from the corresponding equivalence class."""

        # TODO not working well if subsetted dataset. E.g. there's small chance that augmenting to validation
        # if underlying augmentation data is splitted. ~Ok as we don't this for prediction

        if self.perm_Mx is not None:
            # modify the target you are looking for, using the permuter
            Mx = self.perm_Mx(Mx)

        Mxs = self.Mxs
        if not isinstance(Mxs, torch.Tensor):
            Mxs = torch.tensor(Mxs)

        choices = (Mxs == Mx).nonzero(as_tuple=True)[0]
        index = choices[torch.randint(len(choices), size=[])]

        img, _ = self.get_img_target_cached(index)

        return img

    def sample_p_Alx(self, _: Any, Mx: Any) -> Any:
        # load raw image
        if self.is_Mx_aug:
            img = self.get_img_from_Mx(Mx)
        else:
            index = Mx
            img, _ = self.get_img_target_cached(index)

        # augment it as desired
        a = self.sample_p_AlI(img)
        return a

    def get_representative(self, Mx: Any) -> Any:
        if self.is_Mx_aug:
            # TODO one issue is that representative will actually be different during test / val /train
            with tmp_seed(self.seed, is_cuda=False):
                # to fix the representative use the same seed. Note that cannot set seed inside
                # dataloader because forked subprocess. In any case we only need non cuda.
                representative = self.get_img_from_Mx(Mx)
        else:
            # representative is simply the non augmented example
            index = Mx
            representative, _ = self.get_img_target_cached(index)

        return self.base_transform(representative)

    def get_base_transform(self) -> Callable[..., Any]:
        """Return the base transform, ie train or test."""
        shape = self.shapes["input"]

        trnsfs = []

        if self.base_resize == "resize":
            trnsfs += [Resize((shape[1], shape[2]))]
        elif self.base_resize is None:
            pass  # no resizing
        else:
            raise ValueError(f"Unknown base_resize={self.base_resize}")

        trnsfs += [ToTensor()]

        if self.is_normalize and self.is_color:
            # only normalize colored images
            # raise if can't normalize because you specifically gave `is_normalize`
            # TODO normalization for clip will not affect plotting => if using VAE with clip will look wrong
            trnsfs += [Normalizer(self.normalization, is_raise=True)]

        return Compose(trnsfs)

    def __len__(self) -> int:
        return len(self.cached_targets)

    @property
    def _tmp_len(self):
        # has to use length before cached targets
        return len(self.data)

    @property
    def is_color(self) -> bool:
        shape = self.shapes["input"]
        return shape[0] == 3

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        # Imp: In each child should assign "input" and "target"
        shapes = dict()

        shapes["Mx"] = (self.n_Mxs,)

        return shapes


class ISSLImgDataModule(ISSLDataModule):
    def get_train_val_dataset(
        self, **dataset_kwargs
    ) -> tuple[ISSLImgDataset, ISSLImgDataset]:
        dataset = self.Dataset(
            self.data_dir, download=False, curr_split="train", **dataset_kwargs,
        )
        n_val = int_or_ratio(self.val_size, len(dataset))
        train, valid = random_split_cache(
            dataset,
            [len(dataset) - n_val, n_val],
            generator=torch.Generator().manual_seed(self.seed),
        )

        # ensure that you can change the validation dataset without impacting train
        valid.dataset = copy.deepcopy(valid.dataset)
        valid.dataset.curr_split = "validation"

        return train, valid

    def get_train_dataset(self, **dataset_kwargs) -> ISSLImgDataset:
        if self.is_force_all_train or "validation" in self.Dataset.get_available_splits():
            train = self.Dataset(
                self.data_dir, curr_split="train", download=False, **dataset_kwargs,
            )
        else:
            # if there is no validation split will compute it on the fly
            train, _ = self.get_train_val_dataset(**dataset_kwargs)

        return train

    def get_val_dataset(self, **dataset_kwargs) -> ISSLImgDataset:
        if "validation" in self.Dataset.get_available_splits():
            valid = self.Dataset(
                self.data_dir,
                curr_split="validation",
                download=False,
                **dataset_kwargs,
            )
        else:
            # if there is no validation split will compute it on the fly
            _, valid = self.get_train_val_dataset(**dataset_kwargs)

        return valid

    def get_test_dataset(self, **dataset_kwargs) -> ISSLImgDataset:
        test = self.Dataset(
            self.data_dir, curr_split="test", download=False, **dataset_kwargs,
        )

        return test

    def prepare_data(self) -> None:
        for split in self.Dataset.get_available_splits():
            logger.info(f"Preparing data {split}.")
            self.Dataset(
                self.data_dir, curr_split=split, download=True, **self.dataset_kwargs
            )


### Torchvision Datasets ###


# Cifar10 #
class Cifar10Dataset(ISSLImgDataset, CIFAR10):
    attr_data_memory = "data"

    def __init__(self, *args, curr_split = "train",  **kwargs) -> None:
        is_train = curr_split == "train"
        super().__init__(*args, curr_split=curr_split, train=is_train,  **kwargs)

    @property
    def shapes(self) -> dict[Optional[str], tuple[int, ...]]:
        shapes = super(Cifar10Dataset, self).shapes
        shapes["input"] = shapes.get("input", (3, 32, 32))
        shapes["target"] = (10,)
        return shapes

    def get_img_target(self, index: int) -> tuple[Any, int]:
        img, target = CIFAR10.__getitem__(self, index)
        return img, target

    def cache_targets_(self):
        self.cached_targets = to_numpy(self.targets).copy()

    @classmethod
    @property
    def dataset_name(cls) -> str:
        return "CIFAR10"

    @property
    def standard_augmentations(self) -> list[str]:
        return ["simclr"]


class Cifar10DataModule(ISSLImgDataModule):
    def __init__(self, *args, is_data_in_memory=True, **kwargs):
        super().__init__(*args, is_data_in_memory=is_data_in_memory, **kwargs)

    @classmethod
    @property
    def Dataset(cls) -> Any:
        return Cifar10Dataset


### Other Datasets ###
class CachedImageFolder(ImageFolder):
    @file_cache(filename="cached_classes.json")
    def find_classes(self, directory, *args, **kwargs):
        classes = super().find_classes(directory, *args, **kwargs)
        return classes

    @file_cache(filename="cached_structure.json")
    def make_dataset(self, directory, *args, **kwargs):
        dataset = super().make_dataset(directory, *args, **kwargs)
        return dataset

class ExternalImgDataset(ISSLImgDataset):
    """Base class for external datasets that are neither torchvision nor tensorflow. Images will be
    saved as jpeg.

    Parameters
    ----------
    root : str or Path
        Base path to directory for saving data.

    download : bool, optional
        Whether to download the data if it does not exist.

    kwargs :
        Additional arguments to `ISSLImgDataset`.

    class attributes
    ----------------
    min_size : int, optional
        Resizing of the smaller size of an edge to a certain value. If `None` does not resize.
        Recommended for images that will be always rescaled to a smaller version (for memory gains).
        Only used when downloading.
    """

    min_size = 256
    required_packages = []

    def __init__(
        self,
        root: Union[str, Path],
        *args,
        download: bool = True,
        curr_split: str = "train",
        **kwargs,
    ) -> None:
        for p in self.required_packages:
            check_import(p, type(self).__name__)

        self.root = Path(root)
        if download and not self.is_exist_data:
            self.download_extract()
            self.preprocess()

        self.load_data_(curr_split)
        self.length = 0
        for ext in IMG_EXTENSIONS:
            self.length += len(list(self.get_dir(curr_split).glob(f"**/*{ext}")))

        super().__init__(*args, curr_split=curr_split, **kwargs)

    @property
    def _tmp_len(self) -> int:
        return self.length

    def get_dir(self, split: Optional[str] = None) -> Path:
        """Return the main directory or the one for a split."""
        if split == "validation" and split != self.get_available_splits():
            split = "train"  # validation split comes from train because there's no explicit validation

        main_dir = Path(self.root) / self.dataset_name
        if split is None:
            return main_dir
        else:
            return main_dir / split

    @property
    def is_exist_data(self) -> bool:
        """Whether the data is available."""
        is_exist = True
        for split in self.get_available_splits():
            check_file = self.get_dir(split) / EXIST_DATA
            is_exist &= check_file.is_file()
        return is_exist

    def download_extract(self) -> None:
        """Download the dataset and extract it."""
        logger.info(f"Downloading {self.dataset_name} ...")

        data_dir = self.get_dir()
        utils.helpers.remove_rf(data_dir, not_exist_ok=True)
        data_dir.mkdir(parents=True)

        self.download(data_dir)

        logger.info(f"Extracting {self.dataset_name} ...")

        # extract all zips, and do so recursively if needed
        zips = list(data_dir.glob("*.zip"))
        while len(zips) > 0:
            for filename in zips:
                logger.info(f"Unzipping {filename}")
                unzip(filename)
            zips = list(data_dir.glob("*.zip"))

        logger.info(f"{self.dataset_name} successfully extracted.")

    def preprocess(self) -> None:
        """Preprocesses all the extracted and downloaded data."""
        for split in self.get_available_splits():
            logger.info(f"Preprocessing {self.dataset_name} split={split}.")
            split_path = self.get_dir(split)

            self.move_split(split)
            utils.helpers.remove_rf(split_path, not_exist_ok=True)
            split_path.mkdir()

            to_rm = self.preprocess_split(split)

            check_file = split_path / EXIST_DATA
            check_file.touch()

            for f in to_rm:
                # remove all files and directories that are not needed
                utils.helpers.remove_rf(f)

        logger.info(f"{self.dataset_name} successfully pre-processed.")

    @classmethod
    def get_available_splits(cls) -> list[str]:
        return ["test", "train"]

    def move_split(self, split: str) -> None:
        """Moves folders for a specific split before starting the processing."""
        pass

    @property
    def preprocessing_resizer(self) -> Callable[..., Any]:
        """Resizing function for preprocessing step."""
        if self.min_size is None:
            return Compose([])
        else:
            return Resize(self.min_size)

    @abc.abstractmethod
    def download(self, data_dir: Path) -> None:
        """Actual downloading of the dataset to `data_dir`."""
        ...

    @abc.abstractmethod
    def preprocess_split(self, split: str) -> Sequence[Union[str, Path]]:
        """Preprocesses the current split, and return all the files that can be removed fpr that split."""
        ...

    @abc.abstractmethod
    def load_data_(self, split: str) -> None:
        """Loads data if needed."""
        ...



class TinyImagenetDataset(ExternalImgDataset):
    """Tiny Imagenet."""

    split_to_tmp = dict(test="tmp_test", train="tmp_train")
    split_to_old = dict(test="val", train="train")
    required_packages = []
    min_size = None
    urls = [
        "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    ]

    def __init__(self, *args, normalization="ImageNet", **kwargs):
        super().__init__(
            *args, normalization=normalization, **kwargs,
        )

    def download(self, data_dir: Path) -> None:
        for url in self.urls:
            logger.info(f"Downloading {url}")
            download_url(url, data_dir)

    def preprocess_split(self, split: str) -> Sequence[Union[str, Path]]:
        tmp_path = self.get_dir() / self.split_to_tmp[split]
        split_path = self.get_dir(split)

        if split == "test":
            img_path = tmp_path / "images"
            with open(tmp_path / "val_annotations.txt", "r") as f:
                for line in f.readlines():
                    split_line = line.split("\t")
                    name = split_line[0]
                    label = split_line[1]
                    label_path = split_path / label
                    label_path.mkdir(parents=True, exist_ok=True)
                    (img_path / name).rename(label_path / name)

        elif split == "train":
            for img_path in tmp_path.glob("**/images"):
                label = img_path.parent.stem
                label_path = split_path / label
                label_path.mkdir(parents=True, exist_ok=True)
                img_path.rename(label_path)

        else:
            raise ValueError(f"Unknown split={split}.")

        files_to_rm = [tmp_path]
        return files_to_rm

    def move_split(self, split: str) -> None:
        """Moves folders for a specific split before starting the processing."""
        old_split = self.get_dir() / self.split_to_old[split]
        tmp_split = self.get_dir() / self.split_to_tmp[split]
        old_split.rename(tmp_split)

    @property
    def shapes(self) -> dict[Optional[str], tuple[int, ...]]:
        shapes = super().shapes
        shapes["input"] = shapes.get("input", (3, 64, 64))
        shapes["target"] = (200,)
        return shapes

    @property
    def _tmp_len(self) -> int:
        return len(self.loader)

    def get_img_target(self, index: int) -> tuple[Any, int]:
        img, target = self.loader[index]
        return img, target

    def load_data_(self, curr_split: str) -> None:
        self.loader = CachedImageFolder(self.get_dir(curr_split))

    def cache_targets_(self):
        self.cached_targets = np.array(self.loader.targets).copy()

    def download_extract(self) -> None:
        super().download_extract()
        # data is extracted at tiny-imagenet-200/tiny-imagenet-200/ so move it one folder up
        curr_dir = self.get_dir() / self.dataset_name
        tmp_dir = Path(self.root) / f"tmp_{self.dataset_name}"

        curr_dir.rename(tmp_dir)
        utils.helpers.remove_rf(curr_dir.parent)
        tmp_dir.rename(curr_dir.parent)

    @classmethod
    @property
    def dataset_name(cls) -> str:
        return "tiny-imagenet-200"

    @property
    def standard_augmentations(self) -> list[str]:
        return ["simclr"]


class TinyImagenetDataModule(ISSLImgDataModule):
    def __init__(self, *args, is_data_in_memory=True, **kwargs):
        super().__init__(*args, is_data_in_memory=is_data_in_memory, **kwargs)

    @classmethod
    @property
    def Dataset(cls) -> Any:
        return TinyImagenetDataset


