"""Torch implementation of translation-aligned dynamic time warping."""

from typing import Optional, Sequence, Tuple

import torch


def dtw_from_cost_matrix(cost: torch.Tensor):
    """Compute exact DTW distances for ``[batch, Tx, Ty]`` costs."""
    if cost.ndim != 3:
        raise ValueError(f"Expected [batch, Tx, Ty] costs, got {cost.shape}")
    batch_size, rows, columns = cost.shape
    if rows == 0 or columns == 0:
        raise ValueError("DTW inputs must contain at least one frame")
    cost = cost.to(torch.float64)
    accumulated = torch.full(
        (batch_size, rows + 1, columns + 1),
        float("inf"),
        dtype=cost.dtype,
        device=cost.device,
    )
    accumulated[:, 0, 0] = 0
    for diagonal in range(2, rows + columns + 1):
        row_start = max(1, diagonal - columns)
        row_end = min(rows, diagonal - 1)
        row = torch.arange(row_start, row_end + 1, device=cost.device)
        column = diagonal - row
        predecessor = torch.minimum(
            accumulated[:, row - 1, column - 1],
            torch.minimum(
                accumulated[:, row - 1, column],
                accumulated[:, row, column - 1],
            ),
        )
        accumulated[:, row, column] = cost[:, row - 1, column - 1] + predecessor
    return accumulated[:, -1, -1]


def pairwise_jpe(
    generated: torch.Tensor,
    reference: torch.Tensor,
    wanted: Optional[Sequence[int]] = None,
    align_idx: int = 0,
    chunk_size: int = 128,
):
    """Build a pairwise root-translation-aligned joint-position-error matrix."""
    if generated.ndim != 3 or reference.ndim != 3:
        raise ValueError("Sequences must have shape [frames, joints, xyz]")
    if generated.shape[1:] != reference.shape[1:]:
        raise ValueError("Generated and reference joint layouts differ")
    if wanted is not None:
        selected = torch.as_tensor(wanted, device=generated.device)
        generated_selected = generated.index_select(1, selected)
        reference_selected = reference.index_select(1, selected)
    else:
        generated_selected = generated
        reference_selected = reference

    generated_root = generated[:, align_idx:align_idx + 1]
    reference_root = reference[:, align_idx:align_idx + 1]
    costs = []
    for start in range(0, len(generated), chunk_size):
        end = min(start + chunk_size, len(generated))
        aligned = (
            generated_selected[start:end, None]
            - generated_root[start:end, None]
            + reference_root[None]
        )
        delta = aligned - reference_selected[None]
        costs.append(delta.square().sum(-1).sqrt().mean(-1))
    return torch.cat(costs)


def batched_dtw_jpe(
    pairs: Sequence[
        Tuple[torch.Tensor, torch.Tensor, Optional[Sequence[int]], int]
    ],
):
    """Evaluate multiple body-part DTW costs in one dynamic program."""
    costs = torch.stack([
        pairwise_jpe(generated, reference, wanted, align_idx)
        for generated, reference, wanted, align_idx in pairs
    ])
    return dtw_from_cost_matrix(costs)
