"""Train MaDiS or its SiCLIP evaluator."""

import os

import pytorch_lightning as pl

from mGPT.callback import build_callbacks
from mGPT.config import instantiate_from_config, parse_args
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.utils.load_checkpoint import load_pretrained
from mGPT.utils.logger import create_logger


def main():
    cfg = parse_args(phase="train")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logger = create_logger(cfg, phase="train")
    pl.seed_everything(cfg.SEED_VALUE)

    lightning_loggers = []
    for logger_name in cfg.LOGGER.TYPE:
        lightning_loggers.append(instantiate_from_config(
            getattr(cfg.LOGGER, logger_name.upper())))

    datamodule = build_data(cfg, phase="train")
    model = build_model(cfg, datamodule)
    if cfg.TRAIN.PRETRAINED:
        load_pretrained(cfg, model, logger, phase="train")

    distributed = cfg.NUM_NODES > 1 or len(cfg.DEVICE) > 1
    trainer = pl.Trainer(
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        num_nodes=cfg.NUM_NODES,
        strategy="ddp_find_unused_parameters_true" if distributed else "auto",
        max_epochs=cfg.TRAIN.END_EPOCH,
        precision=cfg.PRECISION,
        check_val_every_n_epoch=cfg.LOGGER.VAL_EVERY_STEPS,
        default_root_dir=cfg.FOLDER_EXP,
        logger=lightning_loggers,
        callbacks=build_callbacks(cfg, logger=logger, phase="train"),
        benchmark=False,
        deterministic=False,
    )
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=cfg.TRAIN.PRETRAINED if cfg.TRAIN.RESUME else None,
    )
    logger.info("Outputs saved to %s", cfg.FOLDER_EXP)


if __name__ == "__main__":
    main()
