from __future__ import annotations

import inspect
import os
from collections import Callable
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import torch
from issl.helpers import to_numpy


def get_default_args(func: Callable) -> dict:
    """Return the default arguments of a function.
    credit : https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
    """
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def assert_sns_vary_only_param(
    data: pd.DataFrame, sns_kwargs: dict, param_vary_only: Optional[list]
) -> None:
    """
    Make sure that the only multi indices that have not been conditioned over for plotting and has non
    unique values are in `param_vary_only`.
    """
    if param_vary_only is not None:
        multi_idcs = data.index
        issues = []
        for idx in multi_idcs.levels:
            is_varying = len(idx.values) != 1
            is_conditioned = idx.name in sns_kwargs.values()
            is_can_vary = idx.name in param_vary_only
            if is_varying and not is_conditioned and not is_can_vary:
                issues.append(idx.name)

        if len(issues) > 0:
            raise ValueError(
                f"Not only varying {param_vary_only}. Also varying {issues}."
            )


def aggregate(
    table: Union[pd.DataFrame, pd.Series],
    cols_to_agg: list[str] = [],
    aggregates: list[str] = ["mean", "sem"],
    is_rename_cols: bool = True
) -> Union[pd.DataFrame, pd.Series]:
    """Aggregate values of pandas dataframe over some columns.

    Parameters
    ----------
    table : pd.DataFrame or pd.Series
        Table to aggregate.

    cols_to_agg : list of str
        list of columns over which to aggregate. E.g. `["seed"]`.

    aggregates : list of str
        list of functions to use for aggregation. The aggregated columns will be called `{col}_{aggregate}`.

    is_rename_cols : bool
        Whether to add the aggregation name to the column.
    """
    if len(cols_to_agg) == 0:
        return table

    if isinstance(table, pd.Series):
        table = table.to_frame()

    new_idcs = [c for c in table.index.names if c not in cols_to_agg]
    table_agg = table.reset_index().groupby(by=new_idcs, dropna=False).agg(aggregates)

    if is_rename_cols:
        table_agg.columns = ["_".join(col).rstrip("_") for col in table_agg.columns.values]
    else:
        table_agg.columns = [col[0] for col in table_agg.columns.values]

    return table_agg


def save_fig(
    fig: Any, filename: Union[str, bytes, os.PathLike], dpi: int, is_tight: bool = True
) -> None:
    """General function for many different types of figures."""

    # order matters ! and don't use elif!
    if isinstance(fig, sns.FacetGrid):
        fig = fig.fig

    if isinstance(fig, torch.Tensor):
        x = fig.permute(1, 2, 0)
        if x.size(2) == 1:
            fig = plt.imshow(to_numpy(x.squeeze()), cmap="gray")
        else:
            fig = plt.imshow(to_numpy(x))
        plt.axis("off")

    if isinstance(fig, plt.Artist):  # any type of axes
        fig = fig.get_figure()

    if isinstance(fig, plt.Figure):

        plt_kwargs = {}
        if is_tight:
            plt_kwargs["bbox_inches"] = "tight"

        fig.savefig(filename, dpi=dpi, **plt_kwargs)
        plt.close(fig)
    else:
        raise ValueError(f"Unknown figure type {type(fig)}")
