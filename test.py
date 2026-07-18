"""Evaluate MaDiS generation or SiCLIP checkpoints."""

import os

import numpy as np
import pytorch_lightning as pl
from rich import get_console
from rich.table import Table

from mGPT.callback import build_callbacks
from mGPT.config import parse_args
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.utils.load_checkpoint import load_pretrained
from mGPT.utils.logger import create_logger


def print_table(metrics, logger):
    table = Table(title="Evaluation metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    for key, value in metrics.items():
        table.add_row(key, value)
    get_console().print(table, justify="center")
    logger.info(metrics)


def main():
    cfg = parse_args(phase="test")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    cfg.FOLDER = cfg.TEST.FOLDER
    logger = create_logger(cfg, phase="test")
    pl.seed_everything(cfg.SEED_VALUE)

    datamodule = build_data(cfg, phase="test")
    model = build_model(cfg, datamodule)
    load_pretrained(cfg, model, logger, phase="test")

    distributed = cfg.NUM_NODES > 1 or len(cfg.DEVICE) > 1
    trainer = pl.Trainer(
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        num_nodes=cfg.NUM_NODES,
        strategy="ddp_find_unused_parameters_true" if distributed else "auto",
        default_root_dir=cfg.FOLDER_EXP,
        logger=False,
        callbacks=build_callbacks(cfg, logger=logger, phase="test"),
        benchmark=False,
        deterministic=False,
        enable_progress_bar=True,
    )

    replications = []
    for replication in range(cfg.TEST.REPLICATION_TIMES):
        logger.info("Evaluation replication %d", replication + 1)
        replications.append(trainer.test(model, datamodule=datamodule)[0])

    summary = {}
    for key in replications[0]:
        values = np.asarray([
            float(result[key]) for result in replications
        ], dtype=np.float64)
        summary[key] = f"{values.mean():.3f}"
    print_table(summary, logger)


if __name__ == "__main__":
    main()
