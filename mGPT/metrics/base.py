"""Metric registry limited to the evaluations reported for MaDiS."""

from torch import nn

from .clip import CLIPMetrics
from .generation import GenerationMetrics


class BaseMetrics(nn.Module):
    def __init__(self, cfg, datamodule, metrics_dict, **kwargs):
        super().__init__()
        for metric_name in metrics_dict:
            if metric_name == "GenerationMetrics":
                self.GenerationMetrics = GenerationMetrics(cfg=cfg)
            elif metric_name == "CLIPMetrics":
                self.CLIPMetrics = CLIPMetrics()
            else:
                raise ValueError(f"Unsupported MaDiS metric: {metric_name}")
