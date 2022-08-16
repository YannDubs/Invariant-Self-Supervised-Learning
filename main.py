"""Entry point to train the models and evaluate them.

This should be called by `python main.py <conf>` where <conf> sets all configs from the cli, see
the file `config/main.yaml` for details about the configs. or use `python main.py -h`.
"""


from __future__ import annotations

import copy
import logging
import os
import shutil
import traceback
from pathlib import Path
from typing import Any, Optional, Type
import sys

import hydra
import matplotlib.pyplot as plt
import omegaconf
import pandas as pd
import pytorch_lightning as pl
import torch
from hydra import compose
from omegaconf import Container, OmegaConf
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment

from issl import ISSLModule, Predictor
from issl.losses.dino import MAWeightUpdate
from issl.helpers import check_import
from utils.cluster.nlprun import nlp_cluster
from utils.data import get_Datamodule
from utils.helpers import (
    ModelCheckpoint,
    NamespaceMap,
    apply_representor,
    cfg_save,
    format_resolver,
    get_latest_match,
    list2str_resolver,
    log_dict,
    omegaconf2namespace,
    remove_rf,
    replace_keys,
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


@hydra.main(config_name="main", config_path="config")
def main_except(cfg):
    try:
        if cfg.is_nlp_cluster:
            with nlp_cluster(cfg):
                main(cfg)
        else:
            main(cfg)

    except SystemExit:
        logger.exception("Failed this error:")
        # submitit returns sys.exit when SIGTERM. This will be run before exiting.
        smooth_exit(cfg)


def main(cfg):
    breakpoint()
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

    is_force_retrain = repr_cfg.representor.is_force_retrain
    is_train = repr_cfg.representor.is_train or is_force_retrain
    if is_train and not is_trained(repr_cfg, stage, is_force_retrain=is_force_retrain):
        representor = ISSLModule(hparams=repr_cfg)
        repr_trainer = get_trainer(repr_cfg, representor, dm=repr_datamodule, is_representor=True)
        representor = initialize_representor_(representor, repr_datamodule, repr_trainer, repr_cfg)

        logger.info("Train representor ...")
        fit_(repr_trainer, representor, repr_datamodule, repr_cfg)
        save_pretrained(repr_cfg, repr_trainer, stage)
    else:
        logger.info("Load pretrained representor ...")
        if repr_cfg.representor.is_use_init:
            # for online pretrained SSL models simply load the pretrained model (init)
            representor = ISSLModule(hparams=repr_cfg)
        else:
            representor = load_pretrained(repr_cfg, ISSLModule, stage)
        repr_trainer = get_trainer(repr_cfg, representor, is_representor=True)
        placeholder_fit(repr_trainer, representor)
        repr_cfg.evaluation.representor.ckpt_path = None  # eval loaded model

    if repr_cfg.evaluation.representor.is_evaluate:
        logger.info("Evaluate representor ...")
        repr_res = evaluate(
            repr_trainer, repr_datamodule, repr_cfg, stage, model=representor
        )
    else:
        repr_res = load_results(repr_cfg, stage)

    finalize_stage_(
        stage,
        repr_cfg,
        repr_res,
        finalize_kwargs,
        is_save_best=True,
    )

    del repr_datamodule  # not used anymore and can be large

    ############## DOWNSTREAM PREDICTOR ##############
    pred_res_all = dict()

    for task in cfg.downstream_task.all_tasks:
        logger.info(f"Stage : Predict {task}")
        stage = "predictor"
        pred_cfg = set_downstream_task(cfg, task)
        pred_cfg = set_cfg(pred_cfg, stage)
        pred_datamodule = instantiate_datamodule_(
            pred_cfg, pre_representor=repr_trainer
        )
        pred_cfg = omegaconf2namespace(pred_cfg)

        is_force_retrain = pred_cfg.predictor.is_force_retrain
        is_train = pred_cfg.predictor.is_train or is_force_retrain
        if is_train and not is_trained(pred_cfg, stage, is_force_retrain=is_force_retrain):
            predictor = Predictor(hparams=pred_cfg)
            pred_trainer = get_trainer(pred_cfg, predictor, is_representor=False)

            logger.info(f"Train predictor for {task} ...")
            fit_(pred_trainer, predictor, pred_datamodule, pred_cfg)
            save_pretrained(pred_cfg, pred_trainer, stage)

        else:
            logger.info(f"Load pretrained predictor for {task} ...")
            predictor = load_pretrained(pred_cfg, Predictor, stage)
            pred_trainer = get_trainer(pred_cfg, predictor, is_representor=False)
            placeholder_fit(pred_trainer, predictor)

        pred_cfg.evaluation.predictor.ckpt_path = None  # eval loaded model

        if pred_cfg.evaluation.predictor.is_evaluate:
            logger.info(f"Evaluate predictor for {task} ...")
            pred_res = evaluate(
                pred_trainer,
                pred_datamodule,
                pred_cfg,
                stage,
                model=predictor
            )
        else:
            pred_res = load_results(pred_cfg, stage)

        pred_res_all.update(pred_res)
        save_end_file(pred_cfg)


    finalize_stage_(
        stage,
        pred_cfg,
        pred_res_all,
        finalize_kwargs,
    )

    ############## SHUTDOWN ##############

    finalize(**finalize_kwargs)


def begin(cfg: Container) -> None:
    """Script initialization."""

    pl.seed_everything(cfg.seed)

    cfg.paths.work = str(Path.cwd())

    try:
        # if continuing from single job you shouldn't append run to the end
        continue_job = cfg.continue_job  #! used to trigger the try except
        if cfg.is_rm_job_num:
            # in case the original job was actually without a job num
            cfg.job_id = "_".join(str(cfg.job_id).split("_")[:-1])
    except:
        pass


    logger.info(f"Workdir : {cfg.paths.work}.")
    logger.info(f"Job id : {cfg.job_id}.")


def get_stage_name(stage: str) -> str:
    """Return the correct stage name given the mode (representor, predictor, ...)"""
    return stage[:4]


def set_downstream_task(cfg: Container, task: str):
    """Set the downstream task."""


    cfg = copy.deepcopy(cfg)  # not inplace
    with omegaconf.open_dict(cfg):

        cfg.downstream_task = compose(  config_name="main", overrides=[f"+downstream_task={task}"] ).downstream_task
        cfg.update_trainer_pred.max_epochs = int(cfg.update_trainer_pred.max_epochs * cfg.downstream_task.epochs_mult_factor)

        # TODO should clean that but not sure how. Currently:
        # 1/ reload hydra config with the current data as dflt config
        overrides = [f"+data@dflt_data_pred={cfg.downstream_task.data}",f"+predictor@dflt_predictor={cfg.downstream_task.predictor}"]
        if "optimizer" in cfg.downstream_task:
            overrides += [f"optimizer@dflt_optimizer_pred={cfg.downstream_task.optimizer}"] # no + because there is a default
        if "scheduler" in cfg.downstream_task:
            overrides += [f"scheduler@dflt_scheduler_pred={cfg.downstream_task.scheduler}"] # no + because there is a default

        dflts = compose(config_name="main", overrides=overrides)
        cfg.dflt_predictor = dflts.dflt_predictor
        cfg.dflt_data_pred = dflts.dflt_data_pred
        cfg.dflt_optimizer_pred = dflts.dflt_optimizer_pred
        cfg.dflt_scheduler_pred = dflts.dflt_scheduler_pred

        # 2/ add any overrides
        cfg.predictor = OmegaConf.merge(cfg.dflt_predictor, cfg.predictor)
        cfg.data_pred = OmegaConf.merge(cfg.dflt_data_pred, cfg.data_pred)
        cfg.optimizer_pred = OmegaConf.merge(cfg.dflt_optimizer_pred, cfg.optimizer_pred)
        cfg.scheduler_pred = OmegaConf.merge(cfg.dflt_scheduler_pred, cfg.scheduler_pred)

        if "max_epochs" in cfg.downstream_task:
            cfg.update_trainer_pred.max_epochs = cfg.downstream_task.max_epochs

        if "batch_size" in cfg.downstream_task:
            cfg.data_pred.kwargs.batch_size = cfg.downstream_task.batch_size

        if "add_pred" in cfg.downstream_task:
            cfg.other.add_pred = cfg.other.add_pred

        if cfg.data_pred.is_copy_repr:
            name = cfg.data_repr.name
            if cfg.data_repr.name == "stl10_unlabeled":
                # stl10_unlabeled goes to stl10 at test time
                name = "stl10"
                cfg.data_pred.dataset = "stl10"

            cfg.data_pred.name = cfg.data_pred.name.format(name=name)
            cfg.data_pred = OmegaConf.merge(cfg.data_repr, cfg.data_pred)

        cfg.downstream_task.name = task

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

        if stage == "representor":
            cfg.task = cfg.data.name
        elif stage == "predictor":
            cfg.task = cfg.downstream_task.name

        logger.info(f"Name : {cfg.long_name}.")

        # rescaling learning rate depening on batch size
        lr_stage = "issl" if cfg.stage == "repr" else cfg.stage
        batch_size = cfg.data.kwargs.batch_size
        if batch_size != 256:
            new_lr = cfg[f"optimizer_{lr_stage}"].kwargs.lr * batch_size / 256
            logger.info(f"Rescaling lr to {new_lr}.")
            cfg[f"optimizer_{lr_stage}"].kwargs.lr = new_lr

    # make sure all paths exist
    for name, path in cfg.paths.items():
        if isinstance(path, str):
            Path(path).mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoint path is {cfg.paths.checkpoint}.")
    logger.info(f"Results path is {cfg.paths.results}.")

    Path(cfg.paths.pretrained.save).mkdir(parents=True, exist_ok=True)


    file_end_results = Path(cfg.paths.results) / f"{cfg.stage}_{FILE_END}"

    if file_end_results.is_file() and not cfg[stage].is_force_retrain:
        logger.info(f"Skipping most of {cfg.stage} as {file_end_results} exists.")

        with omegaconf.open_dict(cfg):
            if stage == "representor":
                cfg.representor.is_train = False
                cfg.evaluation.representor.is_evaluate = False
                cfg.data_repr.kwargs.is_data_in_memory = False

            elif stage == "predictor":
                cfg.predictor.is_train = False
                cfg.evaluation.predictor.is_evaluate = False
                cfg.data_pred.kwargs.is_data_in_memory = False

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

    limit_train_batches = cfgt.get("limit_train_batches", 1)
    if limit_train_batches > 1:
        # if limit_train_batches is in number of batches
        cfgd.length = cfgd.kwargs.batch_size * limit_train_batches
    else:
        # if limit_train_batches is in percentage
        cfgd.length = int(len(datamodule.train_dataset) * limit_train_batches)
    cfgd.shape = datamodule.shape
    cfgd.target_shape = datamodule.target_shape
    cfgd.aux_shape = datamodule.aux_shape
    cfgd.mode = datamodule.mode
    cfgd.aux_target = datamodule.aux_target
    cfgd.normalized = datamodule.normalized
    if pre_representor is not None:
        datamodule = apply_representor(
            datamodule,
            pre_representor,
            is_eval_on_test=cfg.evaluation.is_eval_on_test,
            **cfgd.kwargs,
        )
        datamodule.prepare_data()
        datamodule.setup()

        # changes due to the representations
        cfgd.shape = (datamodule.train_dataset.X.shape[-1],)
        cfgd.mode = "vector"

    n_devices = max(cfgt.gpus * cfgt.num_nodes, 1)
    eff_batch_size = n_devices * cfgd.kwargs.batch_size * cfgt.accumulate_grad_batches
    cfgd.n_train_batches = 1 + cfgd.length // eff_batch_size
    cfgd.max_steps = cfgt.max_epochs * cfgd.n_train_batches

    return datamodule


def initialize_representor_(
    module: ISSLModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    cfg: NamespaceMap,
) -> None:
    """Additional steps needed for initialization of the compressor + logging."""

    # save number of parameters for the main model (not online optimizer but with coder)
    n_param = sum(
        p.numel() for p in module.get_specific_parameters("all") if p.requires_grad
    )
    log_dict(trainer, {"n_param": n_param}, is_param=True)

    return module


def get_callbacks(
    cfg: NamespaceMap, is_representor: bool, dm: pl.LightningDataModule=None
) -> list[pl.callbacks.Callback]:
    """Return list of callbacks."""
    callbacks = []

    if is_representor:

        if hasattr(cfg.decodability, "is_ema") and cfg.decodability.is_ema:
            # use momentum contrastive teacher, e.g. DINO
            callbacks += [MAWeightUpdate()]

    callbacks += [ModelCheckpoint(**cfg.checkpoint.kwargs)]

    for name, kwargs in cfg.callbacks.items():
        try:
            if kwargs.is_use:
                callback_kwargs = kwargs.get("kwargs", {})
                if callback_kwargs.get("dm", False):
                    callback_kwargs["dm"] = dm

                Callback = getattr(pl.callbacks, name)
                new_callback = Callback(**callback_kwargs)

                if isinstance(new_callback, BaseFinetuning) and not is_representor:
                    pass  # don't add finetuner during prediction
                else:
                    callbacks.append(new_callback)

        except AttributeError:
            pass

    return callbacks


def get_logger(cfg: NamespaceMap) -> pl.loggers.base.LightningLoggerBase:
    """Return correct logger."""

    kwargs = cfg.logger.kwargs
    # useful for different modes (e.g. wandb_kwargs)
    kwargs.update(cfg.logger.get(f"{cfg.logger.name}_kwargs", {}))

    if cfg.logger.name == "wandb":
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

    # TRAINER
    trainer = pl.Trainer(
        plugins=[SLURMEnvironment(auto_requeue=False)], # lightning automatically detects slurm and tries to handle checkpointing but we want outside #6389
        logger=get_logger(cfg),
        callbacks=get_callbacks(cfg, is_representor, dm=dm),
        **kwargs,
    )

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
    ckpt_dir = Path(cfg.checkpoint.kwargs.dirpath)
    if cfg.checkpoint.is_load_last:
        last_checkpoint = ckpt_dir / LAST_CHECKPOINT
    else:
        # don't use last.ckpt (typically if there was an issue with saving)
        last_checkpoint = get_latest_match(ckpt_dir / "epoch*.ckpt")

    if last_checkpoint.exists():
        kwargs["ckpt_path"] = str(last_checkpoint)
        logger.info(f"Continuing run from {last_checkpoint}")

    trainer.fit(module, datamodule=datamodule, **kwargs)


def placeholder_fit(
    trainer: pl.Trainer, module: pl.LightningModule
) -> None:
    """Necessary setup of trainer before testing if you don't fit it."""
    trainer.model = module


def save_pretrained(
    cfg: NamespaceMap, trainer: pl.Trainer, stage: str,
) -> None:
    """Send best checkpoint to main directory."""

    # restore best checkpoint
    best = trainer.checkpoint_callback.best_model_path
    try:
        trainer._checkpoint_connector.resume_start(best)
    except AttributeError:
        # Older versions of lightning
        trainer.checkpoint_connector.resume_start(best)

    # save
    dest_path = Path(cfg.paths.pretrained.save)
    dest_path.mkdir(parents=True, exist_ok=True)
    filename = BEST_CHECKPOINT.format(stage=stage)
    ckpt_path = dest_path / filename
    trainer.save_checkpoint(ckpt_path, weights_only=True)
    logger.info(f"Saved best checkpoint to {ckpt_path}.")


def is_trained(cfg: NamespaceMap, stage: str, is_force_retrain: bool=False) -> bool:
    """Test whether already saved the checkpoint, if yes then you already trained but might have preempted."""
    pretrained_path = Path(cfg.paths.pretrained.save)
    filename = BEST_CHECKPOINT.format(stage=stage)

    if is_force_retrain and (pretrained_path / filename).is_file():
        results_path = Path(cfg.paths.results)
        ckpt_path = Path(cfg.checkpoint.kwargs.dirpath)
        log_path = Path(cfg.paths.logs)
        logger.info(f"Forcing the retraining of {stage}, even though {pretrained_path / filename} exists. Deleting {pretrained_path} and {results_path} and {ckpt_path} and {log_path}.")
        remove_rf(pretrained_path, not_exist_ok=True)
        remove_rf(results_path, not_exist_ok=True)
        remove_rf(ckpt_path, not_exist_ok=True)
        remove_rf(log_path, not_exist_ok=True)
        pretrained_path.mkdir(parents=True)
        results_path.mkdir(parents=True)
        ckpt_path.mkdir(parents=True)
        log_path.mkdir(parents=True)
        return False
    else:
        return (pretrained_path / filename).is_file()


def load_pretrained(
    cfg: NamespaceMap, Module: Optional[Type[pl.LightningModule]], stage: str, **kwargs
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
    model: torch.nn.Module=None
) -> dict:
    """Evaluate the trainer by logging all the metrics from the test set from the best model."""
    test_res = dict()
    to_save = dict()

    try:
        if cfg.checkpoint.name == "last":
            ckpt_path = None
        else:
            ckpt_path = cfg.evaluation[stage].ckpt_path

        eval_dataloader = datamodule.eval_dataloader(cfg.evaluation.is_eval_on_test)

        # logging correct stage
        trainer.lightning_module.stage = cfg.stage

        test_res = trainer.test(dataloaders=eval_dataloader, ckpt_path=ckpt_path, model=model)[0]

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

def save_end_file(cfg):
    """"save end file to make sure that you don't retrain if preemption"""
    file_end = Path(cfg.paths.results) / f"{cfg.stage}_{FILE_END}"
    file_end.touch(exist_ok=True)
    logger.info(f"Saved {file_end}.")

    # save config to results
    cfg_save(cfg, Path(cfg.paths.results) / f"{cfg.stage}_{CONFIG_FILE}")

def finalize_stage_(
    stage: str,
    cfg: NamespaceMap,
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

    save_end_file(cfg)

    finalize_kwargs["results"][stage] = results
    finalize_kwargs["cfgs"][stage] = cfg

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


def smooth_exit(cfg: NamespaceMap) -> None:
    """Everything to run in case you get preempted / exit."""

    training_chckpnt = Path(cfg.paths.checkpoint)
    exit_chckpnt = Path(cfg.paths.exit_checkpoint)

    if training_chckpnt != exit_chckpnt:
        # if you want the checkpoints to be saved somewhere else in case exit
        exit_chckpnt.parent.mkdir(exist_ok=True, parents=True)
        shutil.copytree(training_chckpnt, exit_chckpnt, dirs_exist_ok=True)
        logging.info(f"Moved checkpoint to {exit_chckpnt} for smooth exit.")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("format", format_resolver)
    OmegaConf.register_new_resolver("list2str", list2str_resolver)

    try:
        main_except()
    except:
        logger.exception("Failed this error:")
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)
    finally:
        wandb.finish()
