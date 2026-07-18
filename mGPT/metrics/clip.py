"""Retrieval metrics computed with the paper's SiCLIP encoder."""

import torch
from torchmetrics import Metric


DATASETS = ("how2sign", "csl", "phoenix", "total")


class CLIPMetrics(Metric):
    def __init__(self, **_):
        super().__init__(dist_sync_on_step=True)
        for dataset in DATASETS:
            self.add_state(
                f"{dataset}_count", torch.tensor(0.0), dist_reduce_fx="sum")
            for recall in (1, 5, 10):
                self.add_state(
                    f"{dataset}_R{recall}", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, sim, src, name):
        del name
        ranking = torch.argsort(sim, dim=1, descending=True)
        for index, dataset in enumerate(src):
            for prefix in (dataset, "total"):
                getattr(self, f"{prefix}_count").add_(1)
                for recall in (1, 5, 10):
                    hit = (ranking[index, :recall] == index).any()
                    getattr(self, f"{prefix}_R{recall}").add_(hit)

    def compute(self, sanity_flag=False):
        del sanity_flag
        result = {}
        for dataset in DATASETS:
            count = getattr(self, f"{dataset}_count").clamp_min(1)
            for recall in (1, 5, 10):
                result[f"{dataset}_SiCLIP_R@{recall}"] = (
                    100 * getattr(self, f"{dataset}_R{recall}") / count)
        self.reset()
        return result
