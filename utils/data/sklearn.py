"""
modified from https://github.com/PyTorchLightning/lightning-bolts/blob/ad771c615284816ecadad11f3172459afdef28e3/pl_bolts/datamodules/sklearn_datamodule.py
"""
import math
from typing import Any, Optional

import numpy as np
import pytorch_lightning as pl
import sklearn
from torch.utils.data import DataLoader, Dataset


class SklearnDataset(Dataset):
    """Mapping between numpy (or sklearn) datasets to PyTorch datasets.

    Parameters
    ----------
    X : np.ndarray

    y : np.ndarray

    X_transform : callable
        Function to transform the inputs.

    y_transform : callable
        Function to transform the outputs.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_transform: Any = None,
        y_transform: Any = None,
    ) -> None:
        super().__init__()
        self.X = X
        self.Y = y
        self.X_transform = X_transform
        self.y_transform = y_transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        x = self.X[idx].astype(np.float32)
        y = self.Y[idx]

        # Do not convert integer to float for classification data
        if not ((y.dtype == np.int32) or (y.dtype == np.int64)):
            y = y.astype(np.float32)

        if self.X_transform:
            x = self.X_transform(x)

        if self.y_transform:
            y = self.y_transform(y)

        return x, y


class SklearnDataModule(pl.LightningDataModule):
    name = "sklearn"

    def __init__(
        self,
        X,
        y,
        x_val=None,
        y_val=None,
        x_test=None,
        y_test=None,
        val_split: float = 0.2,
        test_split: float = 0.1,
        num_workers: int = 0,
        seed: int = 123,
        batch_size: int = 16,
        dataset_kwargs: dict = {},
        *args,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset_kwargs = dataset_kwargs
        self.seed = seed
        X, y = sklearn.utils.shuffle(X, y, random_state=self.seed)

        val_split = 0 if x_val is not None or y_val is not None else val_split
        test_split = 0 if x_test is not None or y_test is not None else test_split

        hold_out_split = val_split + test_split
        if hold_out_split > 0:
            # TODO this should really be done with sklearn test_train_split
            val_split = val_split / hold_out_split
            hold_out_size = math.floor(len(X) * hold_out_split)
            x_holdout, y_holdout = X[:hold_out_size], y[:hold_out_size]
            test_i_start = int(val_split * hold_out_size)
            x_val_hold_out, y_val_holdout = (
                x_holdout[:test_i_start],
                y_holdout[:test_i_start],
            )
            x_test_hold_out, y_test_holdout = (
                x_holdout[test_i_start:],
                y_holdout[test_i_start:],
            )
            X, y = X[hold_out_size:], y[hold_out_size:]

        # if don't have x_val and y_val create split from X
        if x_val is None and y_val is None and val_split > 0:
            x_val, y_val = x_val_hold_out, y_val_holdout

        # if don't have x_test, y_test create split from X
        if x_test is None and y_test is None and test_split > 0:
            x_test, y_test = x_test_hold_out, y_test_holdout

        self._init_datasets(X, y, x_val, y_val, x_test, y_test)

    def train_dataloader(
        self, batch_size: Optional[int] = None, **kwargs
    ) -> DataLoader:

        if batch_size is None:
            batch_size = self.batch_size

        loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            **kwargs,
        )
        return loader

    def val_dataloader(self, batch_size: Optional[int] = None, **kwargs) -> DataLoader:

        if batch_size is None:
            batch_size = self.batch_size

        loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            **kwargs,
        )
        return loader

    def test_dataloader(self, batch_size: Optional[int] = None, **kwargs) -> DataLoader:

        if batch_size is None:
            batch_size = self.batch_size

        loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            **kwargs,
        )
        return loader

    # so that same as ISSLDataModule
    def eval_dataloader(self, is_eval_on_test: bool, **kwargs) -> DataLoader:
        """Return the evaluation dataloader (test or val)."""
        if is_eval_on_test:
            return self.test_dataloader(**kwargs)
        else:
            return self.val_dataloader(**kwargs)

    def _init_datasets(
        self,
        X: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        self.train_dataset = SklearnDataset(X, y, **self.dataset_kwargs)
        self.val_dataset = SklearnDataset(x_val, y_val, **self.dataset_kwargs)
        self.test_dataset = SklearnDataset(x_test, y_test, **self.dataset_kwargs)
