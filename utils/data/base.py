"""
Base dataset.
Most code is reused from: https://github.com/YannDubs/lossyless/tree/main/utils/data
"""
from __future__ import annotations

import abc
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Union

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

DIR = Path(__file__).parents[2].joinpath("data")

__all__ = ["ISSLDataset", "ISSLDataModule"]


### Base Dataset ###
class ISSLDataset(abc.ABC):
    """Base class for Invariant SSL.

    Parameters
    -----------
    aux_target : {"input", "representative", "augmentation", "target", None}, optional
        Auxiliary target to append to the target. This will be used to minimize R[aux_target|Z]. `"input"` is the input
        example X, "representative" is a representative of the equivalence class, `"augmentation"` is some
        augmented source A(x). "target" is the target. `None` appends nothing.

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

    is_aux_already_represented = False

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
        self.a_augmentations = a_augmentations
        self.seed = seed
        self.is_normalize = is_normalize

        self.normalization = (
            self.dataset_name if normalization is None else normalization
        )

    @property
    @abc.abstractmethod
    def dataset_name(self) -> str:
        """Name of the dataset."""
        ...

    @abc.abstractmethod
    def get_x_target_Mx(self, index: int) -> tuple[Any, Any, Any]:
        """Return the correct example, target, and maximal invariant."""
        ...

    @abc.abstractmethod
    def get_representative(self, Mx: Any) -> Any:
        """Return a representative element for current Mx."""
        ...

    @abc.abstractmethod
    def sample_p_Alx(self, x: Any, Mx: Any) -> Any:
        """Return some augmentation A of X sampled from p(A|X)."""
        ...

    @property
    @abc.abstractmethod
    def is_clfs(self) -> dict[Optional[str], Any]:
        """Return a dictionary saying whether `input`, `target`, should be classified."""
        ...

    @property
    @abc.abstractmethod
    def shapes(self) -> dict[Optional[str], tuple[int, ...]]:
        """Return dictionary giving the shape `input`, `target`."""
        ...

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        x, target, Mx = self.get_x_target_Mx(index)

        if self.aux_target is None:
            targets = target
        else:
            targets = [target, self.get_aux_target(x, target, Mx)]

        return x, targets

    def get_aux_target(self, x: Any, target: Any, Mx: Any) -> Any:
        """Appends an additional target."""

        if self.aux_target == "input":
            # the input
            to_add = x
        elif self.aux_target == "representative":
            # representative element from same equivalence class
            to_add = self.get_representative(Mx)
        elif self.aux_target == "augmentation":
            # augmented example
            to_add = self.sample_p_Alx(x, Mx)
        elif self.aux_target == "target":
            # duplicate but makes code simpler
            to_add = target
        else:
            raise ValueError(f"Unknown aux_target={self.aux_target}")

        return to_add

    def get_is_clf(self) -> tuple[bool, bool]:
        """Return `is_clf` for the target and aux_target."""
        is_clf = self.is_clfs
        is_clf["representative"] = is_clf["input"]
        is_clf["augmentation"] = is_clf["input"]
        is_clf[None] = None

        return is_clf["target"], is_clf[self.aux_target]

    def get_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return `shapes` for the target and aux_target."""
        shapes = self.shapes
        shapes["representative"] = shapes["input"]
        shapes["augmentation"] = shapes["input"]
        shapes[None] = None

        return shapes["target"], shapes[self.aux_target]


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

    reload_dataloaders_every_n_epochs : bool, optional
        Whether to reload (all) dataloaders at each epoch.

    dataset_kwargs : dict, optional
        Additional arguments for the dataset.
    """

    def __init__(
        self,
        data_dir: Union[Path, str] = DIR,
        val_size: float = 0.1,
        test_size: int = None,
        num_workers: int = 16,
        batch_size: int = 128,
        val_batch_size: Optional[int] = None,
        seed: int = 123,
        reload_dataloaders_every_n_epochs: bool = False,
        dataset_kwargs: dict = {},
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.val_size = val_size
        self.test_size = test_size
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_batch_size = batch_size if val_batch_size is None else val_batch_size
        self.seed = seed
        self.dataset_kwargs = dataset_kwargs
        self.reload_dataloaders_every_n_epochs = reload_dataloaders_every_n_epochs

    @property
    def Dataset(self) -> Any:
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

    @property
    def mode(self) -> str:
        """Says what is the mode/type of data. E.g. images, distributions, ...."""
        raise NotImplementedError()

    @property
    def dataset(self) -> ISSLDataset:
        """Return the underlying (train) dataset. Contains val when val is subset of train."""
        dataset = self.train_dataset
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        return dataset

    def set_info_(self) -> None:
        """Sets some information from the dataset."""
        dataset = self.dataset
        self.target_is_clf, self.aux_is_clf = dataset.get_is_clf()
        self.target_shape, self.aux_shape = dataset.get_shapes()
        self.shape = dataset.shapes["input"]
        self.aux_target = dataset.aux_target
        self.normalized = dataset.normalization if dataset.is_normalize else None
        self.is_aux_already_represented = dataset.is_aux_already_represented

    @property
    def balancing_weights(self) -> dict[str, float]:
        """dictionary mapping every target to a weight that examples from this class should carry."""
        return dict()

    def setup(self, stage: Optional[str] = None) -> None:
        """Prepare the datasets for the current stage."""
        if stage == "fit" or stage is None:
            self.train_dataset = self.get_train_dataset(**self.dataset_kwargs)
            self.set_info_()
            self.val_dataset = self.get_val_dataset(**self.dataset_kwargs)

        if stage == "test" or stage is None:
            self.test_dataset = self.get_test_dataset(**self.dataset_kwargs)

    def train_dataloader(
        self,
        batch_size: Optional[int] = None,
        train_dataset: Optional[ISSLDataset] = None,
        **kwargs,
    ) -> DataLoader:
        """Return the training dataloader while possibly modifying dataset kwargs."""
        data_kwargs = kwargs.pop("dataset_kwargs", {})
        if self.reload_dataloaders_every_n_epochs or len(data_kwargs) > 0:
            curr_kwargs = dict(self.dataset_kwargs, **data_kwargs)
            train_dataset = self.get_train_dataset(**curr_kwargs)

        if train_dataset is None:
            train_dataset = self.train_dataset

        if batch_size is None:
            batch_size = self.batch_size

        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            **kwargs,
        )

    def val_dataloader(self, batch_size: Optional[int] = None, **kwargs) -> DataLoader:
        """Return the validation dataloader while possibly modifying dataset kwargs."""
        data_kwargs = kwargs.pop("dataset_kwargs", {})
        if self.reload_dataloaders_every_n_epochs or len(data_kwargs) > 0:
            curr_kwargs = dict(self.dataset_kwargs, **data_kwargs)
            self.val_dataset = self.get_val_dataset(**curr_kwargs)

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

    def test_dataloader(self, batch_size: Optional[int] = None, **kwargs) -> DataLoader:
        """Return the test dataloader while possibly modifying dataset kwargs."""
        data_kwargs = kwargs.pop("dataset_kwargs", {})
        if self.reload_dataloaders_every_n_epochs or len(data_kwargs) > 0:
            curr_kwargs = dict(self.dataset_kwargs, **data_kwargs)
            self.test_dataset = self.get_test_dataset(**curr_kwargs)

        if batch_size is None:
            batch_size = self.val_batch_size

        return DataLoader(
            self.test_dataset,
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
