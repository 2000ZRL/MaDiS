"""Shared Lightning plumbing for MaDiS training and evaluation."""

import json
import pickle
from collections import OrderedDict
from pathlib import Path

import torch
from pytorch_lightning import LightningModule

from mGPT.config import get_obj_from_str
from mGPT.metrics import BaseMetrics


class BaseModel(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_metrics()
        self.output_dir = Path(
            self.hparams.cfg.FOLDER,
            "madis",
            str(self.hparams.cfg.NAME),
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def training_step(self, batch, batch_idx):
        return self.allsplit_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.allsplit_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        outputs = self.allsplit_step("test", batch, batch_idx)
        if self.hparams.stage == "lm_clip":
            return outputs
        if not self.hparams.cfg.TEST.SAVE_PREDICTIONS:
            return outputs

        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized() else 0
        )
        save_dir = self.output_dir / f"{self.hparams.cfg.TEST.SPLIT}_rank_{rank}"
        save_dir.mkdir(parents=True, exist_ok=True)
        for index, name in enumerate(outputs["name"]):
            result_length = int(outputs["lengths_rst"][index])
            reference_length = int(outputs["lengths"][index])
            prediction = {
                "feats_rst": outputs["feats_rst"][
                    index, :result_length].detach().cpu().numpy(),
                "feats_ref": outputs["feats_ref"][
                    index, :reference_length].detach().cpu().numpy(),
                "text": outputs["text"][index],
            }
            filename = Path(name).name + ".pkl"
            with open(save_dir / filename, "wb") as handle:
                pickle.dump(prediction, handle)
        return outputs

    def on_train_epoch_end(self):
        values = {
            "epoch": float(self.trainer.current_epoch),
            "#training_iters": self._batch_count(
                self.trainer.num_training_batches),
        }
        values.update(self.loss_log_dict("train"))
        if not self.trainer.sanity_checking:
            optimizer = self.optimizers()
            optimizers = optimizer if isinstance(optimizer, (list, tuple)) else [optimizer]
            for index, item in enumerate(optimizers):
                values[f"lr/group_{index}"] = item.param_groups[0]["lr"]
            self.log_dict(values, sync_dist=True, rank_zero_only=True)

    def on_validation_epoch_end(self):
        values = {
            "epoch": float(self.trainer.current_epoch),
            "#validation_iters": self._batch_count(
                self.trainer.num_val_batches),
        }
        values.update(self.loss_log_dict("val"))
        values.update(self.metrics_log_dict())
        if not self.trainer.sanity_checking:
            self.log_dict(values, sync_dist=True, rank_zero_only=True)

    def on_test_epoch_end(self):
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized() else 0
        )
        values = self.metrics_log_dict()
        values["#test_iters"] = self._batch_count(self.trainer.num_test_batches)
        if not self.trainer.sanity_checking:
            self.log_dict(values, sync_dist=True, rank_zero_only=True)

        if hasattr(self.metrics, "GenerationMetrics"):
            save_dir = self.output_dir / f"{self.hparams.cfg.TEST.SPLIT}_rank_{rank}"
            save_dir.mkdir(parents=True, exist_ok=True)
            with open(save_dir / "test_scores.json", "w") as handle:
                json.dump(self.metrics.GenerationMetrics.name2scores, handle)

    def preprocess_state_dict(self, state_dict):
        result = OrderedDict()
        for key, value in self.metrics.state_dict().items():
            result[f"metrics.{key}"] = value
        for key, value in self._losses.state_dict().items():
            result[f"_losses.{key}"] = value
        for key, value in state_dict.items():
            if "_losses" not in key and "Metrics" not in key:
                result[key] = value
        return result

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(
            self.preprocess_state_dict(state_dict), strict=strict)

    def loss_log_dict(self, split):
        return self._losses[f"losses_{split}"].compute(split)

    def metrics_log_dict(self):
        values = {}
        for metric_name in self.hparams.metrics_dict:
            metric_values = getattr(self.metrics, metric_name).compute(
                sanity_flag=self.trainer.sanity_checking)
            values.update({
                f"Metrics/{name}": value.item()
                for name, value in metric_values.items()
            })
        return values

    def configure_optimizers(self):
        cfg = self.hparams.cfg
        optimizer_name = cfg.TRAIN.OPTIM.target
        if "." not in optimizer_name:
            optimizer_name = f"torch.optim.{optimizer_name}"
        optimizer = get_obj_from_str(optimizer_name)(
            params=self.parameters(), **cfg.TRAIN.OPTIM.params)

        scheduler_name = cfg.TRAIN.LR_SCHEDULER.target
        if "." not in scheduler_name:
            scheduler_name = f"torch.optim.lr_scheduler.{scheduler_name}"
        scheduler = get_obj_from_str(scheduler_name)(
            optimizer=optimizer, **cfg.TRAIN.LR_SCHEDULER.params)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def configure_metrics(self):
        self.metrics = BaseMetrics(datamodule=self.datamodule, **self.hparams)

    @staticmethod
    def _batch_count(value):
        if isinstance(value, (list, tuple)):
            return float(sum(value))
        return float(value)
