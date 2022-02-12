"""Entry point to aggregate a series of results obtained using `main.py` in a nice plot / table.

This should be called by `python utils/aggregate.py <conf>` where <conf> sets all configs from the cli, see
the file `config/aggregate.yaml` for details about the configs. or use `python utils/aggregate.py -h`.
"""
from __future__ import annotations

import logging
import os
import sys
from collections.abc import Sequence
from math import isnan
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import minmax_scale

import hydra
import torch
from omegaconf import OmegaConf

try:
    import sklearn.metrics
except:
    pass

try:
    import optuna
except:
    pass

MAIN_DIR = os.path.abspath(str(Path(__file__).parents[1]))
CURR_DIR = os.path.abspath(str(Path(__file__).parents[0]))
sys.path.append(MAIN_DIR)
sys.path.append(CURR_DIR)

from issl.helpers import check_import  # isort:skip
from main import CONFIG_FILE, get_stage_name  # isort:skip
from utils.helpers import (  # isort:skip
    cfg_load,
    cfg_save,
    getattr_from_oneof,
    omegaconf2namespace,
    format_resolver,
    replace_keys,
    remove_rf,
)
from utils.postplotting import (  # isort:skip
    PRETTY_RENAMER,
    PostPlotter,
    data_getter,
    folder_split,
    single_plot,
    table_summarizer,
    filename_format,
)
from utils.postplotting.helpers import aggregate, save_fig  # isort:skip
from utils.visualizations.helpers import kwargs_log_scale  # isort:skip

logger = logging.getLogger(__name__)


@hydra.main(config_path=f"{MAIN_DIR}/config", config_name="aggregate")
def main_cli(cfg):
    # uses main_cli sot that `main` can be called from notebooks.
    return main(cfg)


def main(cfg):

    if cfg.is_force_gpu and not torch.cuda.is_available():
        # force cuda in case you don't want to run on the intitial node
        logger.info(
            "Not aggregating because you asked for GPU which is not available. If that was on purpose, use `is_force_gpu=False`."
        )
        return

    begin(cfg)

    # make sure you are using primitive types from now on because omegaconf does not always work
    cfg = omegaconf2namespace(cfg)

    aggregator = ResultAggregator(pretty_renamer=PRETTY_RENAMER, **cfg.kwargs)

    base_path = aggregator.base_dir / Path(cfg.base_path)
    for job_id in cfg.job_id_to_rm:
        logger.info(f"Removing undesired job={job_id} ..")
        for f in base_path.glob(f"**/jid_*_{job_id}_*"):
            remove_rf(f)

    logger.info(f"Collecting the data ..")
    for name, pattern in cfg.patterns.items():
        if pattern is not None:
            aggregator.collect_data(
                pattern=pattern, table_name=name, **cfg.collect_data
            )

    if len(aggregator.tables) > 1:
        # if multiple tables also add "merged" that contains all
        aggregator.merge_tables(list(aggregator.tables.keys()))

    aggregator.fillna(**cfg.fillna)
    aggregator.cols_to_str(*cfg.cols_to_str)
    aggregator.replace(**cfg.replace)
    aggregator.sort(cfg.sort)
    aggregator.subset(**cfg.col_val_subset)
    aggregator.merge_columns(cfg.merge_cols)

    for f in cfg.agg_mode:

        logger.info(f"Mode {f} ...")

        if f is None:
            continue

        if f in cfg:
            kwargs = cfg[f]
        else:
            kwargs = {}

        getattr(aggregator, f)(**kwargs)

    logger.info("Finished.")


def begin(cfg):
    """Script initialization."""
    OmegaConf.set_struct(cfg, False)  # allow pop
    PRETTY_RENAMER.update(cfg.kwargs.pop("pretty_renamer"))
    OmegaConf.set_struct(cfg, True)

    logger.info(f"Aggregating {cfg.experiment} ...")


# MAIN CLASS
class ResultAggregator(PostPlotter):
    """Aggregates batches of results (multirun)

    Parameters
    ----------
    save_dir : str or Path
        Where to save all results.

    base_dir : str or Path
        Base folder from which all paths start.

    kwargs :
        Additional arguments to `PostPlotter`.
    """

    def __init__(self, save_dir, base_dir=Path(__file__).parent, **kwargs):
        super().__init__(**kwargs)
        self.base_dir = Path(base_dir)
        self.save_dir = self.base_dir / Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.tables = dict()
        self.param_names = dict()
        self.cfgs = dict()
        self.job_ids = set()

    def merge_tables(self, to_merge=["representor", "predictor"]):
        """Add one large table called `"merged"` that concatenates other tables."""
        merged = self.tables[to_merge[0]]
        for table in to_merge[1:]:
            merged = pd.merge(
                merged, self.tables[table], left_index=True, right_index=True
            )
        self.param_names["merged"] = list(merged.index.names)
        self.tables["merged"] = merged

    def collect_data(
        self,
        pattern=f"results/**/results_representor.csv",
        table_name="representor",
        params_to_rm=["jid"],
        params_to_add={},
    ):
        """Collect all the data.

        Notes
        -----
        - Load all the results that are saved in csvs, such that the path name of the form
        `param1_value1/param2_value2/...` and the values in the csv are such that the columns are
        "train", "test", (and possibly other mode), while index shows parameter name.
        - The loaded data are a dataframe where each row is a different run, (multi)indices are the
        parameters and columns contain train_metrics and test_metrics.

        Parameters
        ----------
        pattern : str
            Pattern for globbing data.

        table_name : str, optional
            Name of the table under which to save the loaded data.

        params_to_rm : list of str, optional
            Params to remove.

        params_to_add : dict, optional
            Parameters to add. Those will be added from the `config.yaml` files. The key should be
            the name of the paramter that you weant to add and the value should be the config key
            (using dots). E.g. {"lr": "optimizer.lr"}. The config file should be saved at the same
            place as the results file.
        """
        paths = list(self.base_dir.glob(pattern))
        if len(paths) == 0:
            raise ValueError(f"No files found for your pattern={pattern}")

        results = []
        self.param_names[table_name] = set()
        for path in paths:
            folder = path.parent

            # select everything from "exp_"
            path_clean = "exp_" + str(path.parent.resolve()).split("/exp_")[-1]

            # make dict of params
            params = path_to_params(path_clean)

            try:
                self.job_ids.add(params["jid"].split("_")[1])
            except:
                pass

            for p in params_to_rm:
                params.pop(p)

            try:
                cfg = cfg_load(folder / f"{get_stage_name(table_name)}_{CONFIG_FILE}")
                for name, param_key in params_to_add.items():
                    params[name] = cfg.select(param_key)
                self.cfgs[table_name] = cfg  # will ony save last
            except FileNotFoundError:
                if len(params_to_add) > 0:
                    logger.exception(
                        "Cannot use `params_to_add` as config file was not found:"
                    )
                    raise

            # looks like : DataFrame(param1:...,param2:..., param3:...)
            df_params = pd.DataFrame.from_dict(params, orient="index").T
            # looks like : dict(train={metric1:..., metric2:...}, test={metric1:..., metric2:...})
            dicts = pd.read_csv(path, index_col=0).to_dict()

            # # TODO remove. This is just temporary to work with previous runs.
            # if "datapred" in params.keys() and "_test" in params["datapred"]:
            #     corr = params["datapred"]
            #     curr = corr.replace("test", "")
            #     dicts = {k: replace_keys(v, curr, corr + "/") for k, v in dicts.items()}

            # TODO remove. This is just temporary to work with previous runs.
            dicts = {
                k: {s_k: s_v for s_k, s_v in v.items() if not isnan(s_v)}
                for k, v in dicts.items()
            }
            dicts = {k: replace_keys(v, "_train", "") for k, v in dicts.items()}

            dicts = {
                k: replace_keys(v, f"{cfg.task}/", "") for k, v in dicts.items()
            }

            # flattens dicts and make dataframe :
            # DataFrame(train/metric1:...,train/metric2:..., test/metric1:..., test/metric2:...)
            df_metrics = pd.json_normalize(dicts, sep="/")

            result = pd.concat([df_params, df_metrics], axis=1)

            # to numeric if appropriate
            result = result.apply(pd.to_numeric, errors="ignore")

            results.append(result)

        param_name = list(params.keys())
        self.tables[table_name] = pd.concat(results, axis=0).set_index(param_name)
        self.param_names[table_name] = param_name

        logger.info(f"All job_ids={self.job_ids}.")

    def subset(self, **col_val):
        """Subset all tables by keeping only the given values in given columns.

        Parameters
        ----------
        col_val :
            Keys are the columns to subset and values are a list of values to keep.
        """
        for col, val in col_val.items():
            logger.debug("Keeping only val={val} for col={col}.")
            for k in self.tables.keys():
                table = self.tables[k].reset_index()
                if col not in table.columns:
                    logger.info(f"Skipping subsetting {k} as {col} not there.")
                    continue

                table = table[(table[col]).isin(val)]
                if table.empty:
                    logger.info(f"Empty table after filtering {col}={val}")

                self.tables[k] = table.set_index(self.tables[k].index.names)

    def fillna(self, **values):
        """Fill nans in all tables. Directly call `df.fillna`."""
        for k, df in self.tables.items():
            self.tables[k] = (
                df.reset_index().fillna(value=values).set_index(df.index.names)
            )

    def cols_to_str(self, *cols):
        """Fill nans in all tables. Directly call `df.fillna`."""
        for k, df in self.tables.items():
            table = df.reset_index()
            for col in cols:
                if col in table.columns:
                    table[col] = table[col].astype(str)

            self.tables[k] = table.set_index(df.index.names)

    def replace(self, **values):
        """Fill nans in all tables. Directly call `df.fillna`."""
        if len(values) > 0:
            for k, df in self.tables.items():
                self.tables[k] = (
                    df.reset_index().replace(value=values).set_index(df.index.names)
                )

    def sort(self, col: Optional[Union[Sequence[str], str]]):
        """Sort column."""
        if col is not None:
            for k, df in self.tables.items():
                self.tables[k] = (
                    df.reset_index().sort_values(by=col).set_index(df.index.names)
                )

    def merge_columns(self, merge_cols: dict[str, str]):
        """Merge multiple columns into a single one."""
        is_merged = False
        for k, df in self.tables.items():
            for name, cols in merge_cols.items():
                if cols[0] not in df.columns:
                    continue
                to_merge = df[cols]
                n_not_nan = (~to_merge.isna()).sum(1)
                assert (n_not_nan <= 1).all()
                df[name] = to_merge.iloc[:, 0]
                for _, to_combine in to_merge.iloc[:, 1:].iteritems():
                    df[name] = df[name].combine_first(to_combine)
                self.tables[k][name] = df[name]
                is_merged = True

        if len(merge_cols) > 0:
            assert is_merged

    @data_getter
    @folder_split
    @table_summarizer
    def summarize_metrics(
        self,
        data=None,
        cols_to_agg=["seed"],
        aggregates=["mean", "sem"],
        filename="summarized_metrics_{table}",
        folder_col=None,
    ):
        """Aggregate all the metrics and save them.

        Parameters
        ----------
        data : pd.DataFrame or str, optional
                Dataframe to summarize. If str will use one of self.tables. If `None` uses all data
                in self.tables.

        cols_to_agg : list of str
            List of columns over which to aggregate. E.g. `["seed"]`.

        aggregates : list of str
            List of functions to use for aggregation. The aggregated columns will be called `{col}_{aggregate}`.

        filename : str, optional
                Name of the file for saving the metrics. Can interpolate {table} if from self.tables.

        folder_col : str, optional
            Name of a column that will be used to separate the tables into multiple subfolders.
        """
        return aggregate(data, cols_to_agg, aggregates)

    @filename_format(["cols_to_sweep", "metric", "operator", "threshold"])
    @data_getter
    @table_summarizer
    def summarize_threshold(
        self,
        data: Optional[str] = "predictor",
        cols_to_agg: list[str] = ["seed"],
        cols_to_sweep: list[str] = ["zdim"],
        metric: str = "test/pred/accuracy_score_agg_min_mean",
        operator: str = "geq",
        threshold: float = 0.99,
        filename: str = "summarized_{cols_to_sweep}_{metric}_{operator}_{threshold}_{table}",
    ):
        """Sweep over `col_to_sweep` and store return first value s.t. the metric is larger or smaller
        than a given threshold.

        Parameters
        ----------
        data : pd.DataFrame or str, optional
            Dataframe to summarize. If str will use one of self.tables. If `None` uses all data
            in self.tables.

        cols_to_agg : list of str, optional
            List of columns over which to avg metric before thresholding. E.g. `["seed"]`.
            Note that aggregation is not done after thresholding because usually the gap between
            the sweeps is large => would seem like huge variance.

        cols_to_sweep : list str, optional
            Columns over which to sweep to find value achieving threshold.

        operator : {"leq","geq"}, optional
            Whether to achieve >= (geq) or <= (leq) than threshold.

        threshold : float, optional
            Value that should be achieved.

        filename : str, optional
            Name of the file for saving the metrics. Can interpolate {col_to_sweep} {metric}
            {operator} {threshold} and {table}.
        """
        data_agg = aggregate(data, cols_to_agg, ["mean"])
        data_metric = data_agg[metric]

        if operator == "geq":
            idcs = data_metric >= threshold
            filtered = data_metric[idcs]
            filtered = filtered.reset_index(level=cols_to_sweep)
            filtered = filtered.groupby(filtered.index.names).min()
        elif operator == "leq":
            idcs = data_metric <= threshold
            filtered = data_metric[idcs]
            filtered = filtered.reset_index(level=cols_to_sweep)
            filtered = filtered.groupby(filtered.index.names).max()
        else:
            raise ValueError(f"Unknown operator={operator}.")

        base = data_metric.reset_index(level=cols_to_sweep)
        base = base.groupby(filtered.index.names).max()
        base.iloc[:] = np.nan  # fill with nan

        return base.fillna(filtered)

    @data_getter
    def plot_superpose(
        self,
        x,
        to_superpose,
        value_name,
        data=None,
        filename="{table}_superposed_{value_name}",
        **kwargs,
    ):
        """Plot a single line figure with multiple superposed lineplots.

        Parameters
        ----------
        x : str
            Column name of x axis.

        to_superpose : dictionary
            Dictionary of column values that should be plotted on the figure. The keys
            correspond to the columns to plot and the values correspond to the name they should be given.

        value_name : str
            Name of the yaxis.

        data : pd.DataFrame or str, optional
            Dataframe used for plotting. If str will use one of self.tables. If `None` runs all tables.

        filename : str, optional
            Name of the figure when saving. Can use {value_name} for interpolation.

        kwargs :
            Additional arguments to `plot_scatter_lines`.
        """
        renamer = to_superpose
        key_to_plot = to_superpose.keys()

        data = data.melt(
            ignore_index=False,
            id_vars=[x],
            value_vars=[c for c in key_to_plot],
            value_name=value_name,
            var_name="mode",
        )

        data["mode"] = data["mode"].replace(renamer)
        kwargs["hue"] = "mode"

        return self.plot_scatter_lines(
            data=data,
            x=x,
            y=value_name,
            filename=filename.format(value_name=value_name),
            **kwargs,
        )

    @data_getter
    @folder_split
    @single_plot
    def plot_scatter_lines(
        self,
        x,
        y,
        data=None,
        filename="{table}_lines_{y}_vs_{x}",
        mode="relplot",
        folder_col=None,
        logbase_x=1,
        logbase_y=1,
        sharex=True,
        sharey=False,
        legend_out=True,
        is_no_legend_title=False,
        set_kwargs={},
        x_rotate=0,
        cols_vary_only=None,
        cols_to_agg=[],
        aggregates=["mean", "sem"],
        is_x_errorbar=False,
        is_y_errorbar=False,
        row_title="{row_name}",
        col_title="{col_name}",
        plot_config_kwargs={},
        xlabel="",
        ylabel="",
        hue_order=None,
        style_order=None,
        **kwargs,
    ):
        """Plotting all combinations of scatter and line plots.

        Parameters
        ----------
        x : str
            Column name of x axis.

        y : str
            Column name for the y axis.

        data : pd.DataFrame or str, optional
            Dataframe used for plotting. If str will use one of self.tables. If `None` runs all tables.

        filename : str or Path, optional
            Path to the file to which to save the results to. Will start at `base_dir`.
            Can interpolate {x} and {y}.

        mode : {"relplot","lmplot"}, optional
            Underlying function to use from seaborn. `lmplot` can also plot the estimated regression
            line.

        folder_col : str, optional
            Name of a column that will be used to separate the plot into multiple subfolders.

        logbase_x, logbase_y : int, optional
            Base of the x (resp. y) axis. If 1 no logscale. if `None` will automatically chose.

        sharex,sharey : bool, optional
            Wether to share x (resp. y) axis.

        legend_out : bool, optional
            Whether to put the legend outside of the figure.

        is_no_legend_title : bool, optional
            Whether to remove the legend title. If `legend_out` then will actually duplicate the
            legend :/, the best in that case is to remove the text of the legend column .

        set_kwargs : dict, optional
            Additional arguments to `FacetGrid.set`. E.g.
            dict(xlim=(0,None),xticks=[0,1],xticklabels=["a","b"]).

        x_rotate : int, optional
            By how much to rotate the x labels.

        cols_vary_only : list of str, optional
            Name of the columns that can vary when plotting (e.g. over which to compute bootstrap CI).
            This ensures that you are not you are not taking averages over values that you don't want.
            If `None` does not check. This is especially useful for

        cols_to_agg : list of str
            List of columns over which to aggregate. E.g. `["seed"]`. In case the underlying data
            are given at uniform intervals X, this is probably not needed as seaborn's line plot will
            compute the bootstrap CI for you.

        aggregates : list of str
            List of functions to use for aggregation. The aggregated columns will be called
            `{col}_{aggregate}`.

        is_x_errorbar,is_y_errorbar : bool, optional
            Whether to standard error (over the aggregation of cols_to_agg) as error bar . If `True`,
            `cols_to_agg` should not be empty and `"sem"` should be in `aggregates`.

        row_title,col_title : str, optional
            Template for the titles of the Facetgrid. Can use `{row_name}` and `{col_name}`
            respectively.

        hue_order, style_order : list of str, optional
            Specify the order of processing and plotting for categorical levels of the hue / style semantic.

        plot_config_kwargs : dict, optional
            General config for plotting, e.g. arguments to matplotlib.rc, sns.plotting_context,
            matplotlib.set ...

        kwargs :
            Additional arguments to underlying seaborn plotting function. E.g. `col`, `row`, `hue`,
            `style`, `size` ...
        """

        kwargs["x"] = x
        kwargs["y"] = y

        if hue_order is not None:
            # prettify hue order => can give in not prettified version
            kwargs["hue_order"] = [self.pretty_renamer[h] for h in hue_order]

        if style_order is not None:
            # prettify hue order => can give in not prettified version
            kwargs["style_order"] = [self.pretty_renamer[h] for h in style_order]

        if is_x_errorbar or is_y_errorbar:
            if (len(cols_to_agg) == 0) or ("sem" not in aggregates):
                logger.warning(
                    f"Not plotting errorbars due to empty cols_to_agg={cols_to_agg} or 'sem' not in aggregates={aggregates}."
                )
                is_x_errorbar, is_y_errorbar = False, False

        if mode == "relplot":
            used_kwargs = dict(
                legend="full",
                kind="line",
                markers=True,
                facet_kws={
                    "sharey": sharey,
                    "sharex": sharex,
                    "legend_out": legend_out,
                },
                style=kwargs.get("hue", None),
            )
            used_kwargs.update(kwargs)

            sns_plot = sns.relplot(data=data, **used_kwargs)

        elif mode == "lmplot":
            used_kwargs = dict(
                legend="full", sharey=sharey, sharex=sharex, legend_out=legend_out,
            )
            used_kwargs.update(kwargs)

            sns_plot = sns.lmplot(data=data, **used_kwargs)

        else:
            raise ValueError(f"Unknown mode={mode}.")

        if is_x_errorbar or is_y_errorbar:
            xerr, yerr = None, None
            if is_x_errorbar:
                x_sem = x.rsplit(" ", maxsplit=1)[0] + " Sem"  # _mean -> _sem
                xerr = data[x_sem]

            if is_y_errorbar:
                y_sem = y.rsplit(" ", maxsplit=1)[0] + " Sem"  # _mean -> _sem
                yerr = data[y_sem]

            sns_plot.map_dataframe(add_errorbars, yerr=yerr, xerr=xerr)

        if logbase_x != 1 or logbase_y != 1:
            sns_plot.map_dataframe(
                set_log_scale, basex=logbase_x, basey=logbase_y, **kwargs
            )

        # TODO remove when waiting for https://github.com/mwaskom/seaborn/issues/2456
        if xlabel != "":
            for ax in sns_plot.fig.axes:
                ax.set_xlabel(xlabel)

        if ylabel != "":
            for ax in sns_plot.fig.axes:
                ax.set_ylabel(ylabel)

        sns_plot.tight_layout()

        return sns_plot

    def plot_optuna_hypopt(
        self,
        storage,
        study_name="main",
        filename="hypopt",
        plot_functions_str=[
            "plot_param_importances",
            "plot_parallel_coordinate",
            "plot_optimization_history",
            "plot_pareto_front",
        ],
    ):
        """Plot a summary of Optuna study"""
        check_import("optuna", "plot_optuna_hypopt")
        study = optuna.load_study(study_name, storage)
        cfg = self.cfgs[list(self.cfgs.keys())[-1]]  # which cfg shouldn't matter

        best_trials = study.best_trials
        to_save = {
            "solutions": [{"values": t.values, "params": t.params} for t in best_trials]
        }
        cfg_save(to_save, self.save_dir / f"{self.prfx}{filename}.yaml")

        for i, monitor in enumerate(cfg.monitor_return):
            for plot_f_str in plot_functions_str:
                try:
                    # plotting
                    plt_modules = [optuna.visualization.matplotlib]
                    plot_f = getattr_from_oneof(plt_modules, plot_f_str)
                    out = plot_f(
                        study, target=lambda trial: trial.values[i], target_name=monitor
                    )
                except:
                    logger.exception(f"Could not plot {monitor}. Error:")

                # saving
                nice_monitor = monitor.replace("/", "_")
                filename = self.save_dir / f"optuna_{plot_f_str}_{nice_monitor}"
                save_fig(out, filename, self.dpi)

    @data_getter
    @folder_split
    @single_plot
    def plot_heatmap(
        self,
        x,
        y,
        metric,
        data=None,
        filename: str = "heatmap_{y}_vs_{x}",
        col: Optional[str] = None,
        row: Optional[str] = None,
        normalize: Optional[str] = None,
        cbar_label: Optional[str] = None,
        is_percentage: bool = False,
        folder_col=None,
        logbase_x: Optional[int] = 1,
        logbase_y: Optional[int] = 1,
        sharex: bool = True,
        sharey: bool = True,
        legend_out: bool = True,
        is_no_legend_title: bool = False,
        set_kwargs: set = {},
        x_rotate: float = 0,
        cols_vary_only: Optional[str] = None,
        cols_to_agg: Sequence[str] = [],
        aggregates: Sequence[str] = ["mean", "sem"],
        row_title: str = "{row_name}",
        col_title: str = "{col_name}",
        plot_config_kwargs: dict = {},
        xlabel: str = "",
        ylabel: str = "",
        order_rows: Optional[list] = None,
        order_cols: Optional[list] = None,
        **kwargs,
    ):
        """Plot a heatmap.

        Parameters
        ----------
        x : str
            Column name of x axis of heatmaps.

        y : str
            Column name of y axis of heatmaps.

        metric : str
            Column name of the metric to plot.

        col : str, optional
            Column name in data for plotting multiple heatmaps, one per columns.

        row : str, optional
            Column name in data for plotting multiple heatmaps, one per columns.

        normalize : ["row","col",None], optional
            Whether to normalize the values by row (single 1 per row), by column (single 1 per col)
        or not to.

        cbar_label : str, optional
            Name for the colorbar.

        is_percentage : bool, optional
            Whether to use percentage for the annotation of the heatmap.

        order_rows, order_cols : list of str, optional
            Order of rows and columns.

        row_title, col_title : str, optional
            Name of the columns and row. Can use {col_var} and {col_name}.

        kwargs :
            Additional arguments to `sns.heatmap`.
        """
        data = data.reset_index()
        dflt_kwargs = dict(annot=True, linewidths=0.5, fmt=".2f")

        if is_percentage:
            fmt = lambda x, pos: "{:.0%}".format(x)
            dflt_kwargs.update(dict(fmt=".1%", cbar_kws={"format": FuncFormatter(fmt)}))

        if cbar_label is not None:
            dflt_kwargs["cbar_kws"] = {"label": self.pretty_renamer[cbar_label]}

        if order_rows is not None:
            order_rows = [self.pretty_renamer[r] for r in order_rows]

        if order_cols is not None:
            order_cols = [self.pretty_renamer[c] for c in order_cols]

        dflt_kwargs.update(kwargs)

        sns_plot = sns.FacetGrid(
            data,
            row=row,
            col=col,
            dropna=False,
            sharex=sharex,
            sharey=sharey,
            aspect=1.5,
            height=6,
        )

        sns_plot.map_dataframe(
            draw_heatmap,
            x,
            y,
            metric,
            normalize=normalize,
            order_rows=order_rows,
            order_cols=order_cols,
            **dflt_kwargs,
        )
        sns_plot.fig.tight_layout()

        return sns_plot


# HELPERS


def draw_heatmap(
    x, y, values, normalize=None, order_rows=None, order_cols=None, **kwargs
):
    data = kwargs.pop("data")
    mat = data.pivot(index=y, columns=x, values=values)

    if normalize is None:
        pass
    elif normalize == "row":
        mat = pd.DataFrame(
            minmax_scale(mat.values, axis=1), columns=mat.columns, index=mat.index
        )
    elif normalize == "col":
        mat = pd.DataFrame(
            minmax_scale(mat.values, axis=0), columns=mat.columns, index=mat.index
        )
    else:
        raise ValueError(f"Unknown normalize={normalize}")

    if order_rows:
        mat.index = pd.CategoricalIndex(mat.index, categories=order_rows)
        mat.sort_index(axis=0, inplace=True)

    if order_cols:
        mat.columns = pd.CategoricalIndex(mat.columns, categories=order_cols)
        mat.sort_index(axis=1, inplace=True)

    ax = sns.heatmap(mat, **kwargs)
    ax.invert_yaxis()
    for label in ax.get_yticklabels():
        label.set_rotation(0)


def path_to_params(path):
    """Take a path name of the form `param1_value1/param2_value2/...` and returns a dictionary."""
    params = {}

    for name in path.split("/"):
        if "_" in name:
            k, v = name.split("_", maxsplit=1)
            params[k] = v

    return params


def get_param_in_kwargs(data, **kwargs):
    """
    Return all arguments that are names of the multiindex (i.e. param) of the data. I.e. for plotting
    this means that you most probably conditioned over them.
    """
    return {
        n: col
        for n, col in kwargs.items()
        if (isinstance(col, str) and col in data.index.names)
    }


def add_errorbars(data, yerr, xerr, **kwargs):
    """Add errorbar to each sns.facetplot."""
    datas = [data]
    if xerr is not None:
        datas += [xerr.rename("xerr")]
    if yerr is not None:
        datas += [yerr.rename("yerr")]

    df = pd.concat(datas, axis=1).set_index(["hue", "style"])

    for idx in df.index.unique():
        # error bars will be different for different hue and style
        df_curr = df.loc[idx, :] if len(df.index.unique()) > 1 else df
        errs = dict()
        if xerr is not None:
            errs["xerr"] = df_curr["xerr"]
        if yerr is not None:
            errs["yerr"] = df_curr["yerr"]

        plt.errorbar(
            df_curr["x"].values,
            df_curr["y"].values,
            fmt="none",
            ecolor="lightgray",
            **errs,
        )


def set_log_scale(data, basex, basey, **kwargs):
    """Set the log scales as desired."""
    x_data = data[kwargs["x"]].unique()
    y_data = data[kwargs["y"]].unique()
    plt.xscale(**kwargs_log_scale(x_data, base=basex))
    plt.yscale(**kwargs_log_scale(y_data, base=basey))


if __name__ == "__main__":
    OmegaConf.register_new_resolver("format", format_resolver)
    main_cli()
