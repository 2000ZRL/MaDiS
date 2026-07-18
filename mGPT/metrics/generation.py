"""DTW-JPE and SiBLEU-4 evaluation for MaDiS."""

from collections import Counter, defaultdict
from time import perf_counter

import torch
from torchmetrics import Metric

from mGPT.metrics.dtw import batched_dtw_jpe
from mGPT.utils.human_models import smpl_x


DATASETS = ("how2sign", "csl", "phoenix", "total")


def sibleu_statistics(prediction, reference):
    """Return the clipped-overlap statistics used by the paper evaluator."""
    predicted = Counter(prediction)
    expected = Counter(reference)
    return torch.tensor([
        sum((predicted & expected).values()),
        len(prediction),
        len(reference),
    ], dtype=torch.float64)


def corpus_sibleu(statistics):
    overlap, predicted_length, reference_length = statistics
    if predicted_length <= 0:
        return statistics.new_tensor(0.0)
    brevity_penalty = (
        statistics.new_tensor(1.0)
        if predicted_length > reference_length
        else torch.exp(1 - reference_length / predicted_length)
    )
    return 100 * brevity_penalty * overlap / predicted_length


class GenerationMetrics(Metric):
    """Paper metrics for generated sign sequences."""

    def __init__(self, cfg, **_):
        super().__init__(dist_sync_on_step=True)
        del cfg
        self.name2scores = defaultdict(dict)
        for dataset in DATASETS:
            self.add_state(
                f"{dataset}_count", torch.tensor(0.0), dist_reduce_fx="sum")
            for metric in ("DTW_JPE_body", "DTW_JPE_hands", "time_cost"):
                self.add_state(
                    f"{dataset}_{metric}", torch.tensor(0.0), dist_reduce_fx="sum")
            for part in ("body", "lhand", "rhand"):
                self.add_state(
                    f"{dataset}_SiBLEU_{part}",
                    torch.zeros(3, dtype=torch.float64),
                    dist_reduce_fx="sum",
                )

    @torch.no_grad()
    def update(
        self,
        feats_rst,
        feats_ref,
        joints_rst,
        joints_ref,
        vertices_rst,
        vertices_ref,
        body_tokens,
        lhand_tokens,
        rhand_tokens,
        tokens_ref,
        lengths,
        lengths_rst,
        time_cost,
        split,
        src,
        name,
        **_,
    ):
        del feats_rst, feats_ref, split
        batch_size = len(lengths)
        joints_rst = joints_rst.reshape(batch_size, -1, *joints_rst.shape[1:])
        joints_ref = joints_ref.reshape(batch_size, -1, *joints_ref.shape[1:])
        vertices_rst = vertices_rst.reshape(
            batch_size, -1, *vertices_rst.shape[1:])
        vertices_ref = vertices_ref.reshape(
            batch_size, -1, *vertices_ref.shape[1:])
        left_regressor = smpl_x.orig_hand_regressor["left"].to(vertices_rst)
        right_regressor = smpl_x.orig_hand_regressor["right"].to(vertices_rst)
        body_joints = smpl_x.joint_part2idx["upper_body"]

        for index in range(batch_size):
            dataset = src[index]
            reference_length = int(lengths[index])
            generated_length = int(lengths_rst[index])
            generated_mesh = vertices_rst[index, :generated_length]
            reference_mesh = vertices_ref[index, :reference_length]
            generated_left = torch.matmul(left_regressor, generated_mesh).float()
            reference_left = torch.matmul(left_regressor, reference_mesh).float()
            generated_right = torch.matmul(right_regressor, generated_mesh).float()
            reference_right = torch.matmul(right_regressor, reference_mesh).float()

            if joints_rst.is_cuda:
                torch.cuda.synchronize(joints_rst.device)
            start = perf_counter()
            dtw = batched_dtw_jpe([
                (
                    joints_rst[index, :generated_length],
                    joints_ref[index, :reference_length],
                    body_joints,
                    0,
                ),
                (generated_left, reference_left, None, 0),
                (generated_right, reference_right, None, 0),
            ])
            if dtw.is_cuda:
                torch.cuda.synchronize(dtw.device)
            dtw_time = perf_counter() - start
            body_dtw = dtw[0]
            hands_dtw = (dtw[1] + dtw[2]) / 2

            body_prediction = body_tokens[index].tolist()[:generated_length // 4]
            left_prediction = lhand_tokens[index].tolist()[:generated_length // 4]
            right_prediction = rhand_tokens[index].tolist()[:generated_length // 4]
            body_reference = tokens_ref[index, :reference_length // 4, 0].tolist()
            left_reference = tokens_ref[index, :reference_length // 4, 1].tolist()
            right_reference = tokens_ref[index, :reference_length // 4, 2].tolist()
            body_bleu = sibleu_statistics(body_prediction, body_reference)
            left_bleu = sibleu_statistics(left_prediction, left_reference)
            right_bleu = sibleu_statistics(right_prediction, right_reference)

            for prefix in (dataset, "total"):
                getattr(self, f"{prefix}_count").add_(1)
                getattr(self, f"{prefix}_DTW_JPE_body").add_(body_dtw)
                getattr(self, f"{prefix}_DTW_JPE_hands").add_(hands_dtw)
                getattr(self, f"{prefix}_time_cost").add_(
                    torch.as_tensor(time_cost / batch_size, device=body_dtw.device))
                getattr(self, f"{prefix}_SiBLEU_body").add_(
                    body_bleu.to(body_dtw.device))
                getattr(self, f"{prefix}_SiBLEU_lhand").add_(
                    left_bleu.to(body_dtw.device))
                getattr(self, f"{prefix}_SiBLEU_rhand").add_(
                    right_bleu.to(body_dtw.device))

            self.name2scores[name[index]].update({
                f"{dataset}_DTW_JPE_body": body_dtw.item(),
                f"{dataset}_DTW_JPE_hands": hands_dtw.item(),
                f"{dataset}_SiBLEU_body": corpus_sibleu(body_bleu).item(),
                f"{dataset}_SiBLEU_hands": (
                    (corpus_sibleu(left_bleu) + corpus_sibleu(right_bleu)) / 2
                ).item(),
                f"{dataset}_DTW_time": dtw_time,
            })

    def compute(self, sanity_flag=False):
        del sanity_flag
        result = {}
        for dataset in DATASETS:
            count = getattr(self, f"{dataset}_count").clamp_min(1)
            result[f"{dataset}_DTW_JPE_body"] = (
                getattr(self, f"{dataset}_DTW_JPE_body") / count)
            result[f"{dataset}_DTW_JPE_hands"] = (
                getattr(self, f"{dataset}_DTW_JPE_hands") / count)
            result[f"{dataset}_SiBLEU_body"] = corpus_sibleu(
                getattr(self, f"{dataset}_SiBLEU_body"))
            result[f"{dataset}_SiBLEU_hands"] = (
                corpus_sibleu(getattr(self, f"{dataset}_SiBLEU_lhand"))
                + corpus_sibleu(getattr(self, f"{dataset}_SiBLEU_rhand"))
            ) / 2
            result[f"{dataset}_time_cost"] = (
                getattr(self, f"{dataset}_time_cost") / count)
        self.reset()
        return result
