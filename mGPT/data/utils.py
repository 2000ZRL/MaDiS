"""Batch collation for variable-length sign sequences."""

import torch


def _pad_tensors(tensors):
    dimensions = tensors[0].dim()
    maxima = [max(tensor.size(index) for tensor in tensors) for index in range(dimensions)]
    canvas = tensors[0].new_zeros((len(tensors), *maxima))
    for index, tensor in enumerate(tensors):
        view = canvas[index]
        for dimension in range(dimensions):
            view = view.narrow(dimension, 0, tensor.size(dimension))
        view.copy_(tensor)
    return canvas


def sign_collate(batch):
    batch = [sample for sample in batch if sample is not None]
    if not batch:
        raise ValueError("All samples in a batch were invalid")
    return {
        "text": [sample[0] for sample in batch],
        "motion": _pad_tensors([sample[1] for sample in batch]),
        "length": [sample[2] for sample in batch],
        "name": [sample[3] for sample in batch],
        "m_token_len": [sample[5] for sample in batch],
        "m_tokens": _pad_tensors([sample[6] for sample in batch]),
        "all_captions": [sample[7] for sample in batch],
        "tasks": [sample[8] for sample in batch],
        "src": [sample[9] for sample in batch],
    }
