"""Training and evaluation callbacks for MaDiS."""

from pathlib import Path

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, RichProgressBar


def build_callbacks(cfg, logger=None, phase="test", **_):
    callbacks = [ProgressBar()]
    if phase != "train":
        return callbacks

    displayed = {
        "loss": "total/val",
        "DTW-body": "Metrics/total_DTW_JPE_body",
        "DTW-hands": "Metrics/total_DTW_JPE_hands",
        "SiBLEU-body": "Metrics/total_SiBLEU_body",
        "SiBLEU-hands": "Metrics/total_SiBLEU_hands",
        "SiCLIP-R1": "Metrics/total_SiCLIP_R@1",
    }
    callbacks.append(ProgressLogger(logger, displayed))

    if cfg.TRAIN.STAGE == "lm_pretrain":
        monitor = "total/val"
        mode = "min"
        label = "val-loss"
    elif "CLIPMetrics" in cfg.METRIC.TYPE:
        monitor = "Metrics/total_SiCLIP_R@1"
        mode = "max"
        label = "SiCLIP-R1"
    else:
        monitor = "Metrics/total_SiBLEU_body"
        mode = "max"
        label = "SiBLEU-body"
    callbacks.append(ModelCheckpoint(
        dirpath=Path(cfg.FOLDER_EXP) / "checkpoints",
        filename=f"best-{label}-{{epoch}}",
        monitor=monitor,
        mode=mode,
        every_n_epochs=cfg.LOGGER.VAL_EVERY_STEPS,
        save_top_k=1,
        save_last=True,
        save_on_train_epoch_end=False,
    ))
    return callbacks


class ProgressBar(RichProgressBar):
    def get_metrics(self, trainer, model):
        metrics = super().get_metrics(trainer, model)
        metrics.pop("v_num", None)
        return metrics


class ProgressLogger(Callback):
    def __init__(self, logger, metric_monitor):
        self.logger = logger
        self.metric_monitor = metric_monitor

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        del trainer, pl_module
        self.logger.info("Training started")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        del trainer, pl_module
        self.logger.info("Training finished")

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ):
        del pl_module
        if trainer.sanity_checking:
            self.logger.info("Sanity check passed")
            return
        values = []
        for label, key in self.metric_monitor.items():
            if key in trainer.callback_metrics:
                values.append(f"{label}={trainer.callback_metrics[key].item():.3f}")
        self.logger.info(
            "Epoch %d: %s", trainer.current_epoch, ", ".join(values))
