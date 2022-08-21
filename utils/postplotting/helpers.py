from __future__ import annotations

import inspect
import os
from collections.abc import Callable
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

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


def kwargs_log_scale(unique_val, mode="equidistant", base=None):
    """Return arguments to set log_scale as one would wish.

    Parameters
    ----------
    unique_val : np.array
        All unique values that will be plotted on the axis that should be put in log scale.

    axis : {"x","y"}
        Axis for which to use log_scales.

    mode : ["smooth","equidistant"], optional
        How to deal with the zero, which cannot be dealt with by default as log would give  -infinity.
        The key is that we will use a region close to zero which is linear instead of log.
        In the case of `equidistant` we use ensure that the large tick at zero is at the same distance
        of other ticks than if there was no linear. The problem is that this will give rise to
        nonexistent kinks when the plot goes from linear to log scale. `Smooth` tries to deal
        with that by smoothly varying between linear and log. For examples see
        https://github.com/matplotlib/matplotlib/issues/7008.

    base : int, optional
        Base to use for the log plot. If `None` automatically tries to find it. If `1` doesn't use
        any log scale.
    """
    unique_val.sort()

    # automatically compute base
    if base is None:
        # take avg multiplier between each consecutive elements as base i.e 2,8,32 would be 4
        # but 0.1,1,10 would be 10
        diffs = unique_val[unique_val > 0][1:] / unique_val[unique_val > 0][:-1]
        base = int(diffs.mean().round())

    # if constant diff don't use logscale
    if base == 1 or np.diff(unique_val).var() == 0:
        return dict(value="linear")

    # only need to use symlog if there are negative values (i.e. need some linear region)
    if (unique_val <= 0).any():
        min_nnz = np.abs(unique_val[unique_val != 0]).min()
        if mode == "smooth":
            linscale = np.log(np.e) / np.log(base) * (1 - (1 / base))
        elif mode == "equidistant":
            linscale = 1 - (1 / base)
        else:
            raise ValueError(f"Unkown mode={mode}")

        return {
            "value": "symlog",
            "linthresh": min_nnz,
            "base": base,
            "subs": list(range(base)),
            "linscale": linscale,
        }
    else:
        return {
            "value": "log",
            "base": base,
            "subs": list(range(base)),
        }
