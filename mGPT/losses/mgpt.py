"""Losses used by the release MaDiS training and SiCLIP paths."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLosses


def create_mask(lengths, device):
    """Return a ``[batch, max_length, 1]`` validity mask."""
    lengths = torch.as_tensor(lengths, device=device, dtype=torch.long)
    if lengths.numel() == 0:
        return torch.empty((0, 0, 1), device=device)
    positions = torch.arange(int(lengths.max()), device=device)
    return (positions.unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1)


class DistanceLossWithMask(nn.Module):
    def __init__(self, **_):
        super().__init__()

    def forward(self, prediction, target, length=None, dist_func="l1_smooth"):
        mask = 1 if length is None else create_mask(length, prediction.device)
        prediction = prediction * mask
        target = target * mask
        if dist_func == "l1_smooth":
            return F.smooth_l1_loss(prediction, target)
        if dist_func == "l1":
            return F.l1_loss(prediction, target)
        if dist_func == "l2":
            return F.mse_loss(prediction, target)
        raise ValueError(f"Unsupported reconstruction distance: {dist_func}")


class IdentityLoss(nn.Module):
    """Adapter for scalar losses already computed by the model."""

    def __init__(self, **_):
        super().__init__()

    def forward(self, value, _target):
        return value


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(len(logits), device=logits.device)
    return F.cross_entropy(logits, labels)


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    """Symmetric batch contrastive objective used to train SiCLIP."""
    return (contrastive_loss(similarity) + contrastive_loss(similarity.t())) / 2


class GPTLosses(BaseLosses):
    """Aggregate only the losses reported for MaDiS."""

    def __init__(self, cfg, stage, num_joints, **kwargs):
        self.stage = stage
        if stage in {"lm_pretrain", "lm_instruct"}:
            weights = {
                "gpt_loss": cfg.LOSS.LAMBDA_CLS,
                "gpthand_loss": cfg.LOSS.LAMBDA_CLS,
                "gptrhand_loss": cfg.LOSS.LAMBDA_CLS,
                "recons_motion": cfg.LOSS.LAMBDA_RECONS,
                "recons_latent": cfg.LOSS.LAMBDA_RECONS_LAT,
            }
        elif stage == "lm_clip":
            weights = {"clip_loss": 1.0}
        else:
            raise ValueError(f"Unsupported MaDiS training stage: {stage}")

        functions = {
            name: DistanceLossWithMask if name.startswith("recons_") else IdentityLoss
            for name in weights
        }
        super().__init__(
            cfg,
            list(weights),
            weights,
            functions,
            num_joints,
            **kwargs,
        )

    def update(self, result):
        total = 0.0
        if self.stage in {"lm_pretrain", "lm_instruct"}:
            outputs = result["outputs"]
            total += self._update_loss("gpt_loss", outputs["loss"], outputs["loss"])
            total += self._update_loss(
                "gpthand_loss", outputs["loss_hand"], outputs["loss_hand"])
            total += self._update_loss(
                "gptrhand_loss", outputs["loss_rhand"], outputs["loss_rhand"])
            if "recons_motions" in outputs:
                total += self._update_loss(
                    "recons_motion",
                    outputs["recons_motions"],
                    outputs["gt_motions"],
                )
            if "recons_latent" in outputs:
                total += self._update_loss(
                    "recons_latent",
                    outputs["recons_latent"],
                    outputs["gt_latent"],
                )
        else:
            total += self._update_loss(
                "clip_loss", result["clip_loss"], result["clip_loss"])

        self.total.add_(total.detach())
        self.count.add_(1)
        return total
