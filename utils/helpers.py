from __future__ import annotations

import glob
import logging
import os
import shutil
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np
import pl_bolts
import pytorch_lightning as pl
import sklearn
import torch
from joblib import dump, load
from omegaconf import Container, OmegaConf
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, Subset

from issl.helpers import NamespaceMap, namespace2dict

logger = logging.getLogger(__name__)


def format_resolver(x: Any, pattern: str) -> str:
    return f"{x:{pattern}}"


def list2str_resolver(l: list) -> list:
    if len(l) > 0:
        return "_".join(str(el) for el in sorted(l))
    else:
        return "none"


def cfg_save(
    cfg: Union[NamespaceMap, dict, Container], filename: Union[str, Path]
) -> None:
    """Save a config as a yaml file."""
    if isinstance(cfg, NamespaceMap):
        cfg = OmegaConf.create(namespace2dict(cfg))
    elif isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    elif OmegaConf.is_config(cfg):
        pass
    else:
        raise ValueError(f"Unknown type(cfg)={type(cfg)}.")
    OmegaConf.save(cfg, filename)


def cfg_load(filename):
    """Load a config yaml file."""
    return omegaconf2namespace(OmegaConf.load(filename))


def omegaconf2namespace(cfg, is_allow_missing=False):
    """Converts omegaconf to namespace so that can use primitive types."""
    cfg = OmegaConf.to_container(cfg, resolve=True)  # primitive types
    return dict2namespace(cfg, is_allow_missing=is_allow_missing)


def dict2namespace(d, is_allow_missing=False, all_keys=""):
    """
    Converts recursively dictionary to namespace. Does not work if there is a dict whose
    parent is not a dict.
    """
    namespace = NamespaceMap(d)

    for k, v in d.items():
        if v == "???" and not is_allow_missing:
            raise ValueError(f"Missing value for {all_keys}.{k}.")
        elif isinstance(v, dict):
            namespace[k] = dict2namespace(v, f"{all_keys}.{k}")
    return namespace


def set_debug(cfg):
    """Enter debug mode."""
    logger.info(OmegaConf.to_yaml(cfg))
    torch.autograd.set_detect_anomaly(True)
    os.environ["HYDRA_FULL_ERROR"] = "1"


def get_latest_match(pattern: Union[Path, str]) -> Path:
    """
    Return the file that matches the pattern which was modified the latest.
    """
    all_matches = (Path(p) for p in glob.glob(str(pattern), recursive=True))
    latest_match = max(all_matches, key=lambda x: x.stat().st_mtime)
    return latest_match


def update_prepending(to_update, new):
    """Update a dictionary with another. the difference with .update, is that it puts the new keys
    before the old ones (prepending)."""
    # makes sure don't update arguments
    to_update = to_update.copy()
    new = new.copy()

    # updated with the new values appended
    to_update.update(new)

    # remove all the new values => just updated old values
    to_update = {k: v for k, v in to_update.items() if k not in new}

    # keep only values that ought to be prepended
    new = {k: v for k, v in new.items() if k not in to_update}

    # update the new dict with old one => new values are at the beginning (prepended)
    new.update(to_update)

    return new


class StrFormatter:
    """String formatter that acts like some default dictionary `"formatted" == StrFormatter()["to_format"]`.

    Parameters
    ----------
    exact_match : dict, optional
        dictionary of strings that will be replaced by exact match.

    substring_replace : dict, optional
        dictionary of substring that will be replaced if no exact_match. Order matters.
        Everything is title case at this point.

    to_upper : list, optional
        Words that should be upper cased.
    """

    def __init__(self, exact_match={}, substring_replace={}, to_upper=[]):
        self.exact_match = exact_match
        self.substring_replace = substring_replace
        self.to_upper = to_upper

    def __getitem__(self, key):
        if not isinstance(key, str):
            return key

        if key in self.exact_match:
            return self.exact_match[key]

        key = key.title()

        for match, replace in self.substring_replace.items():
            key = key.replace(match, replace)

        for w in self.to_upper:
            key = key.replace(w, w.upper())

        return key

    def __call__(self, x):
        return self[x]

    def update(self, new_dict):
        """Update the substring replacer dictionary with a new one (missing keys will be prepended)."""
        self.substring_replace = update_prepending(self.substring_replace, new_dict)


def getattr_from_oneof(list_of_obj: list, name: str) -> pl.callbacks.Callback:
    """
    Equivalent to `getattr` but on a list of objects and will return the attribute from the first
    object that has it.
    """
    if len(list_of_obj) == 0:
        # base case
        raise AttributeError(f"{name} was not found.")

    obj = list_of_obj[0]

    try:
        return getattr(obj, name)
    except AttributeError:
        try:
            return getattr_from_oneof(list_of_obj[1:], name)
        except AttributeError:
            pass

    raise AttributeError(f"{name} was not found in {list_of_obj}.")


def replace_keys(d: dict[str, ...], old: str, new: str) -> dict[str, ...]:
    """replace keys in a dict."""
    return {k.replace(old, new): v for k, v in d.items()}


# credits : https://gist.github.com/simon-weber/7853144
@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.

    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL
      is defined.
    """
    # two kind-of hacks here:
    #    * can't get the highest logging level in effect => delegate to the user
    #    * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        with warnings.catch_warnings():
            yield
    finally:
        logging.disable(previous_level)


# noinspection PyBroadException
def log_dict(trainer: pl.Trainer, to_log: dict, is_param: bool) -> None:
    """Safe logging of param or metrics."""
    try:
        if is_param:
            trainer.logger.log_hyperparams(to_log)
        else:
            trainer.logger.log_metrics(to_log)
    except:
        pass


class SklearnDataModule(pl_bolts.datamodules.SklearnDataModule):
    # so that same as ISSLDataModule
    def eval_dataloader(self, is_eval_on_test: bool, **kwargs) -> DataLoader:
        """Return the evaluation dataloader (test or val)."""
        if is_eval_on_test:
            return self.test_dataloader(**kwargs)
        else:
            return self.val_dataloader(**kwargs)


class SklearnTrainer:
    """Wrapper around sklearn that mimics pytorch lightning trainer."""

    def __init__(self, scores: Union[str, Sequence[str]]):
        self.model = None
        self.stage = None
        if isinstance(scores, str):
            scores = [scores]
        self.scores = [getattr(sklearn.metrics, s) for s in scores]

    def fit(self, model: Pipeline, datamodule: SklearnDataModule):
        data = datamodule.train_dataset
        model.fit(data.X, data.Y)
        self.model = model

    def save_checkpoint(self, ckpt_path: Union[str, Path], weights_only: bool = False):
        dump(self.model, ckpt_path)

    def test(
        self, dataloaders: DataLoader, ckpt_path: Union[str, Path]
    ) -> list[dict[str, float]]:
        data = dataloaders.dataset
        if ckpt_path is not None and ckpt_path != "best":
            self.model = load(ckpt_path)
        y_hat = self.model.predict(data.X)

        # return a list of dict just like pl trainer (where usually the list is an element for each data loader)
        # here only works with one dataloader
        return [
            {
                f"test/{self.stage}/{score.__name__}": score(data.Y, y_hat)
                for score in self.scores
            }
        ]


def apply_representor(
    datamodule: pl.LightningDataModule,
    representor: pl.LightningModule,
    is_eval_on_test: bool = True,
    **kwargs,
) -> pl.LightningDataModule:
    """Apply a representor on every example (precomputed) of a datamodule and return a new datamodule."""
    train_dataset = datamodule.train_dataset
    # ensure that you will not be augmenting
    if isinstance(train_dataset, Subset):
        train_dataset.dataset.curr_split = "validation"
    else:
        train_dataset.curr_split = "validation"

    out_train = representor.predict(
        ckpt_path=None,  # use current model
        dataloaders=[
            datamodule.train_dataloader(batch_size=64, train_dataset=train_dataset)
        ],
    )
    out_val = representor.predict(
        ckpt_path=None, dataloaders=[datamodule.val_dataloader(batch_size=64)]
    )
    out_test = representor.predict(
        ckpt_path=None,
        dataloaders=[datamodule.eval_dataloader(is_eval_on_test, batch_size=64)],
    )

    X_train, Y_train = zip(*out_train)
    X_val, Y_val = zip(*out_val)
    X_test, Y_test = zip(*out_test)

    # only select kwargs that can be given to sklearn
    sklearn_kwargs = dict()
    sklearn_kwargs["batch_size"] = kwargs.get("batch_size", 128)
    sklearn_kwargs["num_workers"] = kwargs.get("num_workers", 4)

    # make a datamodule from features that are precomputed
    datamodule = SklearnDataModule(
        np.concatenate(X_train, axis=0),
        np.concatenate(Y_train, axis=0),
        x_val=np.concatenate(X_val, axis=0),
        y_val=np.concatenate(Y_val, axis=0),
        x_test=np.concatenate(X_test, axis=0),
        y_test=np.concatenate(Y_test, axis=0),
        shuffle=True,
        pin_memory=True,
        **sklearn_kwargs,
    )

    return datamodule


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def on_load_checkpoint(self, *args, **kwargs):
        super().on_load_checkpoint(*args, **kwargs)

        # trick to keep only one model because pytorch lightning by default doesn't save
        # best k_models, so when preempting they stack up. Open issue. This is only correct for k=1
        self.best_k_models = {}
        self.best_k_models[self.best_model_path] = self.best_model_score
        self.kth_best_model_path = self.best_model_path


def remove_rf(path: Union[str, Path], not_exist_ok: bool = False) -> None:
    """Remove a file or a folder"""
    path = Path(path)

    if not path.exists() and not_exist_ok:
        return

    if path.is_file():
        path.unlink()
    elif path.is_dir:
        shutil.rmtree(path)
