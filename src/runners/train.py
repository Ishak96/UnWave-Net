import torch

import os
from typing import List, Optional

import hydra
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig

from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)

# https://github.com/Lightning-AI/lightning/pull/12014
from pytorch_lightning.loggers import Logger

from src.util import extras, finish, get_logger, log_hyperparameters
from src.util.compat import is_lightning_2

from .utils import setup_loguru

log = get_logger(__name__)


def train(cfg: DictConfig) -> Optional[float]:
    """Contains the training pipeline.
    Can additionally evaluate model on a testset, using best weights achieved during training.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    kwargs = {}
    setup_loguru()
    if cfg.get("use_deterministic"):
        logger.info("Using deterministic algorithm")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    extras(cfg)
    logger.info(f"Using CUDA {torch.version.cuda}")
    if is_lightning_2():
        logger.warning("Using Lightning 2.0")
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    # Convert relative ckpt path to absolute path if necessary
    ckpt_path = cfg.default_hp.get("resume_from_checkpoint")
    if ckpt_path is not None:
        if not os.path.isabs(ckpt_path):
            ckpt_path = to_absolute_path(ckpt_path)
        logger.critical(f"checkpoint: {ckpt_path}")

        # https://lightning.ai/docs/pytorch/stable/upgrade/from_1_4.html
        if is_lightning_2():
            kwargs["ckpt_path"] = ckpt_path
        else:
            cfg.trainer.resume_from_checkpoint = ckpt_path

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger_list: List[Logger] = []
    if "logger" in cfg:
        for _, lg_conf in cfg.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger_list.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger_list, _convert_="partial"
    )

    # Send some parameters from cfg to all lightning loggers
    log.info("Logging hyperparameters!")
    log_hyperparameters(
        config=cfg,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger_list,
    )

    # Train the model
    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            **kwargs,
        )

    # Get metric score for hyperparameter optimization
    optimized_metric = cfg.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise Exception(
            "Metric for hyperparameter optimization not found! "
            "Make sure the `optimized_metric` in `hparams_search` config is correct!"
        )
    score = trainer.callback_metrics.get(optimized_metric)

    # Test the model
    if cfg.get("test"):
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Make sure everything closed properly
    log.info("Finalizing!")
    finish(
        config=cfg,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger_list,
    )

    # Print path to best checkpoint
    if not cfg.trainer.get("fast_dev_run") and cfg.get("train"):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    return score
