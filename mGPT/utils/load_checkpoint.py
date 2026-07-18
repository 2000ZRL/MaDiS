"""Checkpoint loading without dependencies on legacy training datasets."""

import pickle
from collections import OrderedDict

import torch


_FINE_TUNING_ONLY_KEYS = {
    "lm.language_model.body_codebook",
    "lm.language_model.lhand_codebook",
    "lm.language_model.rhand_codebook",
    "lm.language_model.body_emb_mapper.weight",
    "lm.language_model.lhand_emb_mapper.weight",
    "lm.language_model.rhand_emb_mapper.weight",
    "lm.language_model.gate_net.linear1.weight",
    "lm.language_model.gate_net.linear1.bias",
    "lm.language_model.gate_net.linear2.weight",
    "lm.language_model.gate_net.linear2.bias",
}


class _LegacyWordVectorizer:
    """Unpickling placeholder; this metadata is never used by MaDiS."""


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (
            module == "mGPT.data.humanml.utils.word_vectorizer"
            and name == "WordVectorizer"
        ):
            return _LegacyWordVectorizer
        return super().find_class(module, name)


class _CheckpointPickleModule:
    __name__ = "madis_checkpoint_pickle"
    Unpickler = _CheckpointUnpickler
    Pickler = pickle.Pickler
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)
    dump = staticmethod(pickle.dump)
    dumps = staticmethod(pickle.dumps)


def _state_dict(path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        mmap=True,
        pickle_module=_CheckpointPickleModule,
        weights_only=False,
    )
    return checkpoint["state_dict"]


def load_pretrained(cfg, model, logger=None, phase="train"):
    path = cfg.TRAIN.PRETRAINED if phase == "train" else cfg.TEST.CHECKPOINTS
    if logger is not None:
        logger.info("Loading model checkpoint from %s", path)
    if phase != "train":
        model.load_state_dict(_state_dict(path), strict=True)
        return model

    incompatible = model.load_state_dict(_state_dict(path), strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    invalid_missing = missing - _FINE_TUNING_ONLY_KEYS
    if invalid_missing or unexpected:
        raise RuntimeError(
            "Incompatible pretraining checkpoint: "
            f"missing={sorted(invalid_missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    if missing and logger is not None:
        logger.info(
            "Initializing fine-tuning-only MoP parameters: %s",
            ", ".join(sorted(missing)),
        )
    return model


def load_pretrained_vae(cfg, model, logger=None):
    path = cfg.TRAIN.PRETRAINED_VAE
    if logger is not None:
        logger.info("Loading sign tokenizer checkpoint from %s", path)
    body = OrderedDict()
    left = OrderedDict()
    right = OrderedDict()
    for key, value in _state_dict(path).items():
        if "motion_vae" in key:
            body[key.replace("motion_vae.", "")] = value
        elif "rhand_vae" in key:
            right[key.replace("rhand_vae.", "")] = value
        elif "hand_vae" in key:
            left[key.replace("hand_vae.", "")] = value
        elif "vae" in key:
            body[key.replace("vae.", "")] = value
    model.vae.load_state_dict(body, strict=True)
    model.hand_vae.load_state_dict(left, strict=True)
    model.rhand_vae.load_state_dict(right, strict=True)
    return model
