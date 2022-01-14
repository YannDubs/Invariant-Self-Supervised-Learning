"""Entry point to train the models and evaluate them.

This should be called by `python main.py <conf>` where <conf> sets all configs from the cli, see
the file `config/main.yaml` for details about the configs. or use `python main.py -h`.
"""


from __future__ import annotations

import copy
import logging
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Optional, Type

import hydra
import matplotlib.pyplot as plt
import omegaconf
import pandas as pd
import pytorch_lightning as pl
import torch
from hydra import compose
from omegaconf import Container, OmegaConf
from pytorch_lightning.trainer.configuration_validator import verify_loop_configurations
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities import parsing

import issl
from issl import ISSLModule, Predictor
from issl.callbacks import (
    LatentDimInterpolator,
    ReconstructImages, ReconstructMx,RepresentationUMAP
)
from issl.helpers import MAWeightUpdate, check_import, prod
from issl.predictors import SklearnPredictor, get_representor_predictor
from utils.data import get_Datamodule
from utils.helpers import (
    ModelCheckpoint,
    NamespaceMap,
    SklearnTrainer,
    apply_representor,
    cfg_save,
    format_resolver,
    get_latest_match,
    getattr_from_oneof,
    list2str_resolver,
    log_dict,
    omegaconf2namespace,
    remove_rf,
    replace_keys,
    set_debug,
)

try:
    # noinspection PyUnresolvedReferences
    import wandb
except ImportError:
    pass

logger = logging.getLogger(__name__)
BEST_CHECKPOINT = "best_{stage}.ckpt"
RESULTS_FILE = "results_{stage}.csv"
LAST_CHECKPOINT = "last.ckpt"
FILE_END = "end.txt"
CONFIG_FILE = "config.yaml"

# noinspection PyBroadException
try:
    GIT_HASH = (
        subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"])
        .decode("utf-8")
        .strip()
    )
except:
    logger.exception("Failed to save git hash with error:")
    GIT_HASH = None


@hydra.main(config_name="main", config_path="config")
def main(cfg):
    logger.info(os.uname().nodename)

    ############## STARTUP ##############
    logger.info("Stage : Startup")
    begin(cfg)
    finalize_kwargs = dict(modules={}, trainers={}, datamodules={}, cfgs={}, results={})

    ############## REPRESENTATION LEARNING ##############
    logger.info("Stage : Representor")
    stage = "representor"
    repr_cfg = set_cfg(cfg, stage)
    repr_datamodule = instantiate_datamodule_(repr_cfg)
    repr_cfg = omegaconf2namespace(repr_cfg)  # ensure real python types

    if repr_cfg.representor.is_train and not is_trained(repr_cfg, stage):
        representor = ISSLModule(hparams=repr_cfg)
        repr_trainer = get_trainer(repr_cfg, representor, dm=repr_datamodule, is_representor=True)
        initialize_representor_(representor, repr_datamodule, repr_trainer, repr_cfg)

        logger.info("Train representor ...")
        fit_(repr_trainer, representor, repr_datamodule, repr_cfg)
        save_pretrained(repr_cfg, repr_trainer, stage)
    else:
        logger.info("Load pretrained representor ...")
        if repr_cfg.representor.is_use_init:
            # for pretrained SSL models simply load the pretrained model (init)
            representor = ISSLModule(hparams=repr_cfg)
        else:
            representor = load_pretrained(repr_cfg, ISSLModule, stage)
        repr_trainer = get_trainer(repr_cfg, representor, dm=repr_datamodule, is_representor=True)
        placeholder_fit(repr_trainer, representor, repr_datamodule)
        repr_cfg.evaluation.representor.ckpt_path = None  # eval loaded model

    if repr_cfg.evaluation.representor.is_evaluate:
        logger.info("Evaluate representor ...")
        is_eval_train = repr_cfg.evaluation.representor.is_eval_train
        repr_res = evaluate(
            repr_trainer, repr_datamodule, repr_cfg, stage, is_eval_train=is_eval_train
        )
    else:
        repr_res = load_results(repr_cfg, stage)

    finalize_stage_(
        stage,
        repr_cfg,
        representor,
        repr_trainer,
        repr_datamodule,
        repr_res,
        finalize_kwargs,
        is_save_best=True,
    )
    if repr_cfg.is_skip_pred:
        return finalize(**finalize_kwargs)

    del repr_datamodule  # not used anymore and can be large

    ############## REPRESENT ##############
    if repr_cfg.representor.is_on_the_fly:
        # this will perform representation on the fly.
        on_fly_representor = representor
        pre_representor = None
    else:
        # this is quicker but means that you cannot augment at test time and requires more RAM
        on_fly_representor = None
        pre_representor = repr_trainer

    ############## DOWNSTREAM PREDICTOR ##############
    for task in cfg.downstream_task.all_tasks:
        logger.info(f"Stage : Predict {task}")
        stage = "predictor"
        pred_cfg = set_downstream_task(cfg, task)
        pred_cfg = set_cfg(pred_cfg, stage)
        pred_datamodule = instantiate_datamodule_(
            pred_cfg, pre_representor=pre_representor
        )
        pred_cfg = omegaconf2namespace(pred_cfg)

        is_sklearn = pred_cfg.predictor.is_sklearn
        if pred_cfg.predictor.is_train and not is_trained(pred_cfg, stage):
            if is_sklearn:
                assert not repr_cfg.representor.is_on_the_fly
                predictor = SklearnPredictor(pred_cfg.predictor)
                pred_trainer = SklearnTrainer(pred_cfg)
            else:
                predictor = Predictor(hparams=pred_cfg, representor=on_fly_representor)
                pred_trainer = get_trainer(pred_cfg, predictor, is_representor=False)

            logger.info(f"Train predictor for {task} ...")
            fit_(pred_trainer, predictor, pred_datamodule, pred_cfg)
            save_pretrained(pred_cfg, pred_trainer, stage, is_sklearn=is_sklearn)

        else:
            logger.info(f"Load pretrained predictor for {task} ...")
            ReprPred = get_representor_predictor(on_fly_representor)
            predictor = load_pretrained(pred_cfg, ReprPred, stage)
            pred_trainer = get_trainer(pred_cfg, predictor, is_representor=False)
            placeholder_fit(pred_trainer, predictor, pred_datamodule)
            pred_cfg.evaluation.predictor.ckpt_path = None  # eval loaded model

        if pred_cfg.evaluation.predictor.is_evaluate:
            logger.info(f"Evaluate predictor for {task} ...")
            is_eval_train = pred_cfg.evaluation.predictor.is_eval_train
            pred_res = evaluate(
                pred_trainer,
                pred_datamodule,
                pred_cfg,
                stage,
                is_eval_train=is_eval_train,
                is_sklearn=is_sklearn,
            )
        else:
            pred_res = load_results(pred_cfg, stage)

        # TODO currently finalize_stage only stores the last predictor
        # so if you return, will only return last result from last loop
        finalize_stage_(
            stage,
            pred_cfg,
            predictor,
            pred_trainer,
            pred_datamodule,
            pred_res,
            finalize_kwargs,
        )

    ############## SHUTDOWN ##############

    return finalize(**finalize_kwargs)


def begin(cfg: Container) -> None:
    """Script initialization."""
    if cfg.other.is_debug:
        set_debug(cfg)

    pl.seed_everything(cfg.seed)

    cfg.paths.work = str(Path.cwd())
    cfg.other.git_hash = GIT_HASH

    logger.info(f"Workdir : {cfg.paths.work}.")


def get_stage_name(stage: str) -> str:
    """Return the correct stage name given the mode (representor, predictor, ...)"""
    return stage[:4]


def set_downstream_task(cfg: Container, task: str):
    """Set the downstream task."""
    cfg = copy.deepcopy(cfg)  # not inplace
    with omegaconf.open_dict(cfg):
        cfg.downstream_task = compose(
            config_name="main", overrides=[f"+downstream_task={task}"]
        ).downstream_task
        data = cfg.downstream_task.data
        pred = cfg.downstream_task.predictor

        # TODO should clean that but not sure how. Currently:
        # 1/ reload hydra config with the current data as dflt config
        dflts = compose(
            config_name="main",
            overrides=[
                f"+data@dflt_data_pred={data}",
                f"+predictor@dflt_predictor={pred}",
            ],
        )
        cfg.dflt_predictor = dflts.dflt_predictor
        cfg.dflt_data_pred = dflts.dflt_data_pred

        # 2/ add any overrides
        cfg.predictor = OmegaConf.merge(cfg.dflt_predictor, cfg.predictor)
        cfg.data_pred = OmegaConf.merge(cfg.dflt_data_pred, cfg.data_pred)

        if cfg.data_pred.is_copy_repr:
            name = cfg.data_repr.name
            if cfg.data_repr.name == "stl10_unlabeled":
                # stl10_unlabeled goes to stl10 at test time
                name = "stl10"
                cfg.data_pred.dataset = "stl10"

            cfg.data_pred.name = cfg.data_pred.name.format(name=name)
            cfg.data_pred = OmegaConf.merge(cfg.data_repr, cfg.data_pred)

    return cfg


def set_cfg(cfg: Container, stage: str) -> Container:
    """Set the configurations for a specific mode."""
    cfg = copy.deepcopy(cfg)  # not inplace

    with omegaconf.open_dict(cfg):
        cfg.stage = get_stage_name(stage)

        cfg.long_name = cfg[f"long_name_{cfg.stage}"]
        if stage == "representor":
            # not yet instantiated because doesn't know the data and predictor yet
            del cfg[f"long_name_pred"]
            del cfg.evaluation[f"predictor"]

        cfg.data = OmegaConf.merge(cfg.data, cfg[f"data_{cfg.stage}"])
        cfg.trainer = OmegaConf.merge(cfg.trainer, cfg[f"update_trainer_{cfg.stage}"])
        cfg.checkpoint = OmegaConf.merge(cfg.checkpoint, cfg[f"checkpoint_{cfg.stage}"])

        logger.info(f"Name : {cfg.long_name}.")

    if not cfg.is_no_save:
        # make sure all paths exist
        for _, path in cfg.paths.items():
            if isinstance(path, str):
                Path(path).mkdir(parents=True, exist_ok=True)

        Path(cfg.paths.pretrained.save).mkdir(parents=True, exist_ok=True)

    file_end = Path(cfg.paths.logs) / f"{cfg.stage}_{FILE_END}"
    if file_end.is_file():
        logger.info(f"Skipping most of {cfg.stage} as {file_end} exists.")

        with omegaconf.open_dict(cfg):
            if stage == "representor":
                cfg.representor.is_train = False
                cfg.evaluation.representor.is_evaluate = False

            elif stage == "predictor":  # improbable
                cfg.predictor.is_train = False
                cfg.evaluation.predictor.is_evaluate = False

            else:
                raise ValueError(f"Unknown stage={stage}.")

    return cfg


def instantiate_datamodule_(
    cfg: Container, pre_representor: Optional[pl.Trainer] = None
) -> pl.LightningDataModule:
    """Instantiate dataset."""

    cfgd = cfg.data
    cfgt = cfg.trainer

    Datamodule = get_Datamodule(cfgd.dataset)
    datamodule = Datamodule(**cfgd.kwargs)
    datamodule.prepare_data()
    datamodule.setup()

    cfgd.aux_is_clf = datamodule.aux_is_clf
    limit_train_batches = cfgt.get("limit_train_batches", 1)
    cfgd.length = int(len(datamodule.train_dataset) * limit_train_batches)
    cfgd.shape = datamodule.shape
    cfgd.target_is_clf = datamodule.target_is_clf
    cfgd.target_shape = datamodule.target_shape
    cfgd.balancing_weights = datamodule.balancing_weights
    cfgd.aux_shape = datamodule.aux_shape
    cfgd.mode = datamodule.mode
    cfgd.aux_target = datamodule.aux_target
    cfgd.normalized = datamodule.normalized
    cfgd.is_aux_already_represented = datamodule.is_aux_already_represented
    if pre_representor is not None:
        datamodule = apply_representor(
            datamodule,
            pre_representor,
            is_eval_on_test=cfg.evaluation.is_eval_on_test,
            is_agg_target=cfg.data.aux_target == "agg_target",
            **cfgd.kwargs,
        )
        datamodule.prepare_data()
        datamodule.setup()

        # changes due to the representations
        cfgd.shape = (datamodule.train_dataset.X.shape[-1],)
        cfgd.mode = "vector"

    n_devices = max(cfgt.gpus * cfgt.num_nodes, 1)
    eff_batch_size = n_devices * cfgd.kwargs.batch_size * cfgt.accumulate_grad_batches
    train_batches = 1 + cfgd.length // eff_batch_size
    cfgd.max_steps = cfgt.max_epochs * train_batches

    return datamodule


def initialize_representor_(
    module: ISSLModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    cfg: NamespaceMap,
) -> None:
    """Additional steps needed for initialization of the compressor + logging."""
    # LOGGING
    # save number of parameters for the main model (not online optimizer but with coder)
    n_param = sum(
        p.numel() for p in module.get_specific_parameters("all") if p.requires_grad
    )
    log_dict(trainer, {"n_param": n_param}, is_param=True)


def get_callbacks(
    cfg: NamespaceMap, is_representor: bool, dm: pl.LightningDataModule=None
) -> list[pl.callbacks.Callback]:
    """Return list of callbacks."""
    callbacks = []

    if is_representor:

        aux_target = cfg.data.kwargs.dataset_kwargs.aux_target
        is_img_aux_target = cfg.data.mode == "image"
        is_img_aux_target &= aux_target in ["representative", "input", "augmentation"]
        is_img_aux_target &= not cfg.data.is_aux_already_represented

        if cfg.logger.is_can_plot_img and cfg.evaluation.representor.is_online_eval:
            # can only plot if you have labels
            callbacks += [RepresentationUMAP(dm)]

        is_reconstruct = cfg.decodability.is_reconstruct
        if cfg.logger.is_can_plot_img and is_img_aux_target and is_reconstruct:
            # TODO will not currently work with latent images
            z_dim = cfg.encoder.z_shape
            if not isinstance(z_dim, int):
                z_dim = prod(z_dim)
            callbacks += [LatentDimInterpolator(z_dim)]

            if cfg.trainer.gpus <= 1:
                # TODO does not work (D)DP because of self.store
                callbacks += [ReconstructImages()]

            if "predecode_n_Mx" in cfg.decodability.kwargs and cfg.decodability.kwargs.predecode_n_Mx is not None:
                callbacks += [ReconstructMx()]


    if hasattr(cfg.decodability.kwargs, "is_ema") and cfg.decodability.kwargs.is_ema:
        callbacks += [MAWeightUpdate()]

    callbacks += [ModelCheckpoint(**cfg.checkpoint.kwargs)]

    if not cfg.callbacks.is_force_no_additional_callback:
        for name, kwargs in cfg.callbacks.items():
            try:
                if kwargs.is_use:
                    callback_kwargs = kwargs.get("kwargs", {})
                    modules = [issl.callbacks, pl.callbacks]
                    Callback = getattr_from_oneof(modules, name)
                    new_callback = Callback(**callback_kwargs)

                    if isinstance(new_callback, BaseFinetuning) and not is_representor:
                        pass  # don't add finetuner during prediction
                    else:
                        callbacks.append(new_callback)

            except AttributeError:
                pass

    return callbacks


def get_logger(
    cfg: NamespaceMap, module: pl.LightningModule, is_representor: bool
) -> pl.loggers.base.LightningLoggerBase:
    """Return correct logger."""

    kwargs = cfg.logger.kwargs
    # useful for different modes (e.g. wandb_kwargs)
    kwargs.update(cfg.logger.get(f"{cfg.logger.name}_kwargs", {}))

    if cfg.logger.name == "csv":
        pl_logger = CSVLogger(**kwargs)

    elif cfg.logger.name == "wandb":
        check_import("wandb", "WandbLogger")

        # noinspection PyBroadException
        try:
            pl_logger = WandbLogger(**kwargs)
        except Exception:
            cfg.logger.kwargs.offline = True
            pl_logger = WandbLogger(**kwargs)

        try:
            # try to save all the current code
            pl_logger.experiment.log_code(cfg.paths.base_dir)
        except Exception:
            pass

    elif cfg.logger.name is None:
        pl_logger = False

    else:
        raise ValueError(f"Unknown logger={cfg.logger.name}.")

    return pl_logger


def get_trainer(
    cfg: NamespaceMap, module: pl.LightningModule, is_representor: bool, dm: pl.LightningDataModule=None,
) -> pl.Trainer:
    """Instantiate trainer."""

    kwargs = dict(**cfg.trainer)

    # PARALLEL PROCESSING
    if kwargs["gpus"] > 1:
        # TODO test
        kwargs["sync_batchnorm"] = True
        parallel_devices = [torch.device(f"cuda:{i}") for i in range(kwargs["gpus"])]
        kwargs["plugins"] = DDPPlugin(
            parallel_devices=parallel_devices, find_unused_parameters=True,
        )

    # TRAINER
    trainer = pl.Trainer(
        logger=get_logger(cfg, module, is_representor),
        callbacks=get_callbacks(cfg, is_representor, dm=dm),
        **kwargs,
    )

    # lightning automatically detects slurm and tries to handle checkpointing but we want outside
    # so simply remove hpc save until  #6204 #5225 #6389
    # TODO change when #6389
    trainer.checkpoint_connector.hpc_save = lambda *args, **kwargs: None

    return trainer


def fit_(
    trainer: pl.Trainer,
    module: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    cfg: NamespaceMap,
):
    """Fit the module."""
    kwargs = dict()

    # Resume training ?
    last_checkpoint = Path(cfg.checkpoint.kwargs.dirpath) / LAST_CHECKPOINT
    if last_checkpoint.exists():
        kwargs["ckpt_path"] = str(last_checkpoint)

    trainer.fit(module, datamodule=datamodule, **kwargs)


def placeholder_fit(
    trainer: pl.Trainer, module: pl.LightningModule, datamodule: pl.LightningDataModule
) -> None:
    """Necessary setup of trainer before testing if you don't fit it."""

    # TODO: clean as it seems that lightning keep changing stuff here which makes it impossible for
    # backward compatibility

    # links data to the trainer
    # TODO check carefully as it seems that lightning will start changing data connectors
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/9778
    trainer._data_connector.attach_data(module, datamodule=datamodule)

    # clean hparams
    if hasattr(module, "hparams"):
        parsing.clean_namespace(module.hparams)

    # check that model is configured correctly
    verify_loop_configurations(trainer, module)

    # attach model log function to callback
    trainer._callback_connector.attach_model_logging_functions(module)

    trainer.model = module


def save_pretrained(
    cfg: NamespaceMap, trainer: pl.Trainer, stage: str, is_sklearn: bool = False
) -> None:
    """Send best checkpoint to main directory."""

    if not is_sklearn:
        # restore best checkpoint
        best = trainer.checkpoint_callback.best_model_path
        trainer.checkpoint_connector.resume_start(best)

    # save
    dest_path = Path(cfg.paths.pretrained.save)
    dest_path.mkdir(parents=True, exist_ok=True)
    filename = BEST_CHECKPOINT.format(stage=stage)
    ckpt_path = dest_path / filename
    trainer.save_checkpoint(ckpt_path, weights_only=True)
    logger.info(f"Saved best checkpoint to {ckpt_path}.")


def is_trained(cfg: NamespaceMap, stage: str) -> bool:
    """Test whether already saved the checkpoint, if yes then you already trained but might have preempted."""
    dest_path = Path(cfg.paths.pretrained.save)
    filename = BEST_CHECKPOINT.format(stage=stage)
    return (dest_path / filename).is_file()


def load_pretrained(
    cfg: NamespaceMap, Module: Type[pl.LightningModule], stage: str, **kwargs
) -> pl.LightningModule:
    """Load the best checkpoint from the latest run that has the same name as current run."""
    save_path = Path(cfg.paths.pretrained.load)
    filename = BEST_CHECKPOINT.format(stage=stage)
    # select the latest checkpoint matching the path
    checkpoint = get_latest_match(save_path / filename)

    loaded_module = Module.load_from_checkpoint(checkpoint, **kwargs)

    return loaded_module


# noinspection PyBroadException
def evaluate(
    trainer: pl.Trainer,
    datamodule: pl.LightningDataModule,
    cfg: NamespaceMap,
    stage: str,
    is_eval_train: bool = False,
    is_sklearn: bool = False,
) -> dict:
    """Evaluate the trainer by logging all the metrics from the test set from the best model."""
    test_res = dict()
    to_save = dict()
    try:
        ckpt_path = cfg.evaluation[stage].ckpt_path

        if is_eval_train:
            # add train so that can see in wandb
            train_stage = f"{cfg.stage}_train"

            # ensure that logging train and test resutls differently
            if is_sklearn:
                trainer.stage = train_stage
            else:
                trainer.lightning_module.stage = train_stage

            # first save the training ones because they will be under "test" in wandb
            try:
                # also evaluate training set
                train_dataloader = datamodule.train_dataloader()
                train_res = trainer.test(
                    dataloaders=train_dataloader, ckpt_path=ckpt_path
                )[0]
                train_res = {
                    k: v for k, v in train_res.items() if f"/{train_stage}/" in k
                }
                train_res = replace_keys(
                    train_res, f"/{train_stage}/", f"/{cfg.stage}/"
                )
                to_save["train"] = replace_keys(train_res, "test/", "", is_prfx=True)
            except:
                logger.exception(
                    "Failed to evaluate training set. Skipping this error:"
                )

        eval_dataloader = datamodule.eval_dataloader(cfg.evaluation.is_eval_on_test)

        # logging correct stage
        if is_sklearn:
            trainer.stage = cfg.stage
        else:
            trainer.lightning_module.stage = cfg.stage

        test_res = trainer.test(dataloaders=eval_dataloader, ckpt_path=ckpt_path)[0]

        # ensure that select only correct stage
        test_res = {k: v for k, v in test_res.items() if f"/{cfg.stage}/" in k}
        log_dict(trainer, test_res, is_param=False)
        to_save["test"] = replace_keys(test_res, "test/", "", is_prfx=True)

        # save results
        results = pd.DataFrame.from_dict(to_save)
        filename = RESULTS_FILE.format(stage=stage)
        path = Path(cfg.paths.results) / filename
        results.to_csv(path, header=True, index=True)
        logger.info(f"Logging results to {path}.")

    except:
        logger.exception("Failed to evaluate. Skipping this error:")

    return test_res


def load_results(cfg: NamespaceMap, stage: str) -> dict:
    """
    Load the results that were previously saved or return empty dict. Useful in case you
    preempted but still need access to the results.
    """
    # noinspection PyBroadException
    try:
        filename = RESULTS_FILE.format(stage=stage)
        path = Path(cfg.paths.results) / filename

        # dict of "test","train" ... where subdicts are keys and results
        results = pd.read_csv(path, index_col=0).to_dict()

        results = {
            f"{mode}/{k}": v
            for mode, sub_dict in results.items()
            for k, v in sub_dict.items()
        }
        return results
    except:
        return dict()


def finalize_stage_(
    stage: str,
    cfg: NamespaceMap,
    module: pl.LightningModule,
    trainer: pl.Trainer,
    datamodule: pl.LightningDataModule,
    results: dict,
    finalize_kwargs: dict,
    is_save_best: bool = False,
) -> None:
    """Finalize the current stage."""
    logger.info(f"Finalizing {stage}.")

    # no checkpoints during representation
    assert (
        cfg.checkpoint.kwargs.dirpath != cfg.paths.pretrained.save
    ), "This will remove desired checkpoints"

    # remove all checkpoints as best is already saved elsewhere
    remove_rf(cfg.checkpoint.kwargs.dirpath, not_exist_ok=True)

    # don't keep the pretrained model
    if not is_save_best and "hub" not in cfg.paths.pretrained.save:
        remove_rf(cfg.paths.pretrained.save, not_exist_ok=True)

    if not cfg.is_no_save:
        # save end file to make sure that you don't retrain if preemption
        file_end = Path(cfg.paths.logs) / f"{cfg.stage}_{FILE_END}"
        file_end.touch(exist_ok=True)

        # save config to results
        cfg_save(cfg, Path(cfg.paths.results) / f"{cfg.stage}_{CONFIG_FILE}")

    finalize_kwargs["results"][stage] = results
    finalize_kwargs["cfgs"][stage] = cfg

    if cfg.is_return:
        # don't store large stuff if unnecessary
        finalize_kwargs["modules"][stage] = module
        finalize_kwargs["trainers"][stage] = trainer
        finalize_kwargs["datamodules"][stage] = datamodule


def finalize(
    modules: dict[str, pl.LightningModule],
    trainers: dict[str, pl.Trainer],
    datamodules: dict[str, pl.LightningDataModule],
    cfgs: dict[str, NamespaceMap],
    results: dict[str, dict],
) -> Any:
    """Finalizes the script."""
    cfg = cfgs["representor"]  # this is always in

    logger.info("Stage : Shutdown")

    plt.close("all")

    if cfg.logger.name == "wandb" and wandb.run is not None:
        wandb.run.finish()  # finish the run if still on

    logger.info("Finished.")
    logging.shutdown()

    all_results = dict()
    for partial_results in results.values():
        all_results.update(partial_results)

    if cfg.is_return:
        return modules, trainers, datamodules, cfgs, all_results
    else:
        return get_hypopt_monitor(cfg, all_results)


def get_hypopt_monitor(cfg: NamespaceMap, all_results: dict) -> Any:
    """Return the correct monitor for hyper parameter tuning."""
    out = []
    for i, result_key in enumerate(cfg.monitor_return):
        res = all_results[result_key]
        try:
            direction = cfg.monitor_direction[i]
            if not math.isfinite(res):
                # make sure that infinite or nan monitor are not selected by hypopt
                if direction == "minimize":
                    res = float("inf")
                else:
                    res = -float("inf")
        except IndexError:
            pass

        out.append(res)

    if len(out) == 1:
        return out[0]  # return single value rather than tuple
    return tuple(out)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("format", format_resolver)
    OmegaConf.register_new_resolver("list2str", list2str_resolver)
    main()
