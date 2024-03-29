"""
Base dataset.
Most code is reused from: https://github.com/YannDubs/lossyless/tree/main/utils/data
"""
from __future__ import annotations

import abc
import logging
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from utils.data.helpers import BalancedSubset, subset2dataset

DIR = Path(__file__).parents[2].joinpath("data")
logger = logging.getLogger(__name__)

__all__ = ["ISSLDataset", "ISSLDataModule"]


### Base Dataset ###
class ISSLDataset(abc.ABC):
    """Base class for Invariant SSL.

    Parameters
    -----------
    aux_target : {"augmentation",  None}, optional
        Auxiliary target to append to the target. This will be used to minimize R[aux_target|Z].
        example X,  `"sample_p_Alx"` is some augmented source A(x). `None` appends nothing.

    a_augmentations : set of str, optional
        Augmentations that should be used to construct the axillary target, i.e., p(A|x). I.e. this should define the
        coarsest possible equivalence relation with respect to which to be invariant. Depends on the dataset.

    is_normalize : bool, optional
        Whether to normalize the data.

    normalization : str, optional
        Name of the normalization. If `None`, uses the default from the dataset. Only used if
        `is_normalize`.

    seed : int, optional
        Pseudo random seed.
    """
    attr_data_memory = ""  # set following if data already precomputed to avoid duplication when caching

    def __init__(
        self,
        *args,
        aux_target: Optional[str] = "augmentation",
        a_augmentations: Sequence[str] = {},
        is_normalize: bool = False,
        normalization: Optional[str] = None,
        seed: int = 123,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.aux_target = aux_target
        self.a_augmentations = deepcopy(a_augmentations)
        self.seed = seed
        self.is_normalize = is_normalize
        self.is_cache_data = False

        self.normalization = (
            self.dataset_name if normalization is None else normalization
        )


    @classmethod
    @property
    @abc.abstractmethod
    def dataset_name(cls) -> str:
        """Name of the dataset."""
        ...

    @abc.abstractmethod
    def get_x_target_Mx(self, index: int) -> tuple[Any, Any, Any]:
        """Return the correct example, target, and maximal invariant."""
        ...

    @abc.abstractmethod
    def sample_p_Alx(self, x: Any, Mx: Any) -> Any:
        """Return some augmentation A of X sampled from p(A|X)."""
        ...

    @abc.abstractmethod
    def set_eval_(self):
        """Set the dataset into evaluation mode."""
        ...

    @property
    @abc.abstractmethod
    def shapes(self) -> dict[Optional[str], tuple[int, ...]]:
        """Return dictionary giving the shape `input`, `target`."""
        ...

    def __getitem__(self, index: int) -> tuple[Any, Any]:

        x, target, Mx = self.get_x_target_Mx(index)

        if self.aux_target is not None:
            aux_target = self.get_aux_target(x, target, Mx)
            targets = [target, aux_target]
        else:
            targets = target

        return x, targets

    def get_aux_target(self, x: Any, target: Any, Mx: Any) -> Any:
        """Appends an additional target."""

        if self.aux_target == "augmentation":
            # augmented example
            to_add = self.sample_p_Alx(x, Mx)
        else:
            raise ValueError(f"Unknown aux_target={self.aux_target}")

        return to_add

    def get_shapes(self,) -> tuple[int, Optional[tuple[int, ...]]]:
        """Return `shapes` for the target, aux_target."""
        shapes = self.shapes
        shapes["augmentation"] = shapes["input"]
        shapes[None] = None

        return shapes["target"], shapes[self.aux_target]

    @abc.abstractmethod
    def cache_data_(self, idcs : Sequence[int]=None, num_workers : int=0):
        """Caches the data for given idcs. If `None` caches all."""
        ...

    def del_cached_data_(self):
        """Clear all data in memory. Useful after caching to make sure avoiding duplicates."""
        if self.is_cache_data and hasattr(self, "cached_data"):
            delattr(self, "cached_data")
            self.is_cache_data = False

        elif self.attr_data_memory != "" and hasattr(self, self.attr_data_memory):
            delattr(self, self.attr_data_memory)


### Base Datamodule ###

# cannot use abc because inheriting from lightning :(
class ISSLDataModule(LightningDataModule):
    """Base class for data module for ISSL.

    Notes
    -----
    - similar to pl_bolts.datamodule.CIFAR10DataModule but more easily modifiable.

    Parameters
    -----------
    data_dir : str, optional
        Directory for saving/loading the dataset.

    val_size : int or float, optional
        How many examples to use for validation. This will generate new examples if possible, or
        split from the training set. If float this is in ratio of training size, eg 0.1 is 10%.

    test_size : int, optional
        How many examples to use for test. `None` means all.

    num_workers : int, optional
        How many workers to use for loading data

    batch_size : int, optional
        Number of example per batch for training.

    val_batch_size : int or None, optional
        Number of example per batch during eval and test. If None uses `batch_size`.

    seed : int, optional
        Pseudo random seed.

    subset_train_size : float or int, optional
        Will subset the training data in a balanced fashion. If float, should be
        between 0.0 and 1.0 and represent the proportion of the dataset to retain.
        If int, represents the absolute number or examples. If `None` does not
        subset the data.

    is_data_in_memory : bool, optional
        Whether to pre-load all the data in memory.

    is_val_on_test : bool, optional
        Whether to validate on test. This can be used for very long runs to get results even
        if you get preempted around the end. Should only use once you performed all validation / tuning.

    is_force_all_train : bool, optional
        Whether to force using all the training set, even if no validation is present.
        I.e. no splitting and use part of train as valid.

    dataset_kwargs : dict, optional
        Additional arguments for the dataset.
    """

    def __init__(
        self,
        data_dir: Union[Path, str] = DIR,
        val_size: float = 0.1,
        test_size: int = None,
        num_workers: int = 8,
        batch_size: int = 512,
        val_batch_size: Optional[int] = None,
        seed: int = 123,
        subset_train_size: Optional[float] = None,
        dataset_kwargs: dict = {},
        is_shuffle_train: bool = True,
        is_data_in_memory: bool=False,
        is_val_on_test: bool = False,
        is_force_all_train: bool=False
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.val_size = val_size
        self.test_size = test_size
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_batch_size = batch_size if val_batch_size is None else val_batch_size
        self.seed = seed
        self.subset_train_size = subset_train_size
        self.dataset_kwargs = dataset_kwargs
        self.is_shuffle_train = is_shuffle_train
        self.is_data_in_memory = is_data_in_memory
        self.is_val_on_test = is_val_on_test
        self.is_force_all_train = is_force_all_train

        self.is_already_called = dict()
        for stage in ("fit", "validate", "test", "predict"):
            self.is_already_called[stage] = False

    @classmethod
    @property
    def Dataset(cls) -> Any:
        """Return the correct dataset."""
        raise NotImplementedError()

    def get_train_dataset(self, **dataset_kwargs) -> ISSLDataset:
        """Return the training dataset."""
        raise NotImplementedError()

    def get_val_dataset(self, **dataset_kwargs) -> ISSLDataset:
        """Return the validation dataset."""
        raise NotImplementedError()

    def get_test_dataset(self, **dataset_kwargs) -> ISSLDataset:
        """Return the test dataset."""
        raise NotImplementedError()

    def prepare_data(self) -> None:
        """Download and save data on file if needed."""
        raise NotImplementedError()

    def get_train_dataset_subset(self, **dataset_kwargs):
        train_dataset = self.get_train_dataset(**dataset_kwargs)
        if self.subset_train_size is not None:
            logger.info(f"Subsetting {self.subset_train_size} examples.")
            train_dataset = BalancedSubset(
                train_dataset, self.subset_train_size, stratify=train_dataset.cached_targets, seed=self.seed
            )
        return train_dataset

    def get_test_dataset_proc(self, **dataset_kwargs):
        return self.get_test_dataset(**dataset_kwargs)

    @property
    def dataset(self) -> ISSLDataset:
        """Return the underlying (train) dataset. Contains val when val is subset of train."""
        return subset2dataset(self.train_dataset)

    def set_info_(self) -> None:
        """Sets some information from the dataset."""
        dataset = self.dataset
        self.target_dim, self.aux_shape = dataset.get_shapes()
        self.shape = dataset.shapes["input"]
        self.aux_target = dataset.aux_target
        self.normalized = dataset.normalization if dataset.is_normalize else None


    def setup(self, stage: Optional[str] = None) -> None:
        """Prepare the datasets for the current stage."""
        logger.info("Setting up data")


        if (stage == "fit" or stage is None) and not self.is_already_called["fit"]:
            self.train_dataset = self.get_train_dataset_subset(**self.dataset_kwargs)

            self.set_info_()

            if self.is_val_on_test:
                logger.info("Validate on the test set.")
                self.val_dataset = self.get_test_dataset_proc(**self.dataset_kwargs)
            else:
                self.val_dataset = self.get_val_dataset(**self.dataset_kwargs)

            if self.is_data_in_memory:
                logger.info(f"Caching the data for split=val.")
                self.val_dataset.cache_data_(num_workers=self.num_workers)

                logger.info(f"Caching the data for split=train.")
                self.train_dataset.cache_data_(num_workers=self.num_workers)

            self.is_already_called["fit"] = True



        if (stage == "test" or stage is None) and not self.is_already_called["test"]:
            self.test_dataset = self.get_test_dataset_proc(**self.dataset_kwargs)

            if self.is_data_in_memory:
                logger.info(f"Caching the data for split=test.")
                self.test_dataset.cache_data_(num_workers=self.num_workers)

            self.is_already_called["test"] = True

        logger.info(f"Train size = {len(self.train_dataset)}, val size = {len(self.val_dataset)}, test size = {len(self.test_dataset)}.")


    def train_dataloader(
        self,
        batch_size: Optional[int] = None,
        train_dataset: Optional[ISSLDataset] = None,
        **kwargs,
    ) -> DataLoader:
        """Return the training dataloader while possibly modifying dataset kwargs."""

        if train_dataset is None:
            train_dataset = self.train_dataset

        if batch_size is None:
            batch_size = self.batch_size

        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=self.is_shuffle_train,
            num_workers=self.num_workers,
            pin_memory=True,
            **kwargs,
        )

    def val_dataloader(self, batch_size: Optional[int] = None, **kwargs) -> DataLoader:
        """Return the validation dataloader while possibly modifying dataset kwargs."""

        if batch_size is None:
            batch_size = self.val_batch_size

        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            **kwargs,
        )

    def test_dataloader(self,
                        batch_size: Optional[int] = None,
                        test_dataset: Optional[ISSLDataset] = None,
                        **kwargs) -> DataLoader:
        """Return the test dataloader while possibly modifying dataset kwargs."""

        if test_dataset is None:
            test_dataset = self.test_dataset

        if batch_size is None:
            batch_size = self.val_batch_size

        return DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            **kwargs,
        )

    def eval_dataloader(self, is_eval_on_test, **kwargs):
        """Return the evaluation dataloader (test or val)."""
        if is_eval_on_test:
            return self.test_dataloader(**kwargs)
        else:
            return self.val_dataloader(**kwargs)
