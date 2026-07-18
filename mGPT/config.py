"""Configuration loading for the MaDiS train and test entry points."""

import glob
import importlib
import os
from argparse import ArgumentParser
from os.path import join as pjoin

from omegaconf import OmegaConf


def get_module_config(cfg, filepath="./configs"):
    """Attach reusable YAML modules (language model and tokenizers)."""
    for path in sorted(glob.glob(pjoin(filepath, "*", "*.yaml"))):
        relative = os.path.relpath(path, filepath)
        node = os.path.splitext(relative)[0].replace(os.sep, ".")
        OmegaConf.update(cfg, node, OmegaConf.load(path))
    return cfg


def get_obj_from_str(path, reload=False):
    module_name, object_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    if reload:
        module = importlib.reload(module)
    return getattr(module, object_name)


def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Expected a 'target' entry to instantiate")
    return get_obj_from_str(config["target"])(**config.get("params", {}))


def resume_config(cfg):
    """Resolve the checkpoint and logger state for an explicit resume run."""
    if not cfg.TRAIN.RESUME:
        return cfg
    resume = cfg.TRAIN.RESUME
    if not os.path.isdir(resume):
        raise ValueError(f"Resume directory does not exist: {resume}")
    cfg.TRAIN.PRETRAINED = pjoin(resume, "checkpoints", "last.ckpt")
    wandb_dir = pjoin(resume, "wandb", "latest-run")
    if os.path.isdir(wandb_dir):
        run_files = [name for name in os.listdir(wandb_dir) if "run-" in name]
        if run_files:
            cfg.LOGGER.WANDB.params.id = (
                run_files[0].replace("run-", "").replace(".wandb", "")
            )
    return cfg


def parse_args(phase="train"):
    if phase not in {"train", "test"}:
        raise ValueError(f"Unsupported MaDiS phase: {phase}")

    parser = ArgumentParser(description=f"MaDiS {phase}")
    parser.add_argument("--cfg_assets", default="./configs/assets.yaml")
    parser.add_argument("--cfg", default="./configs/default.yaml")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_nodes", type=int)
    parser.add_argument("--device", type=int, nargs="+")
    parser.add_argument("--nodebug", action="store_true")
    parser.add_argument(
        "opts",
        nargs="*",
        default=[],
        help="OmegaConf overrides, for example TEST.BATCH_SIZE=4",
    )
    params = parser.parse_args()

    cfg_assets = OmegaConf.load(params.cfg_assets)
    cfg_base = OmegaConf.load(pjoin(cfg_assets.CONFIG_FOLDER, "default.yaml"))
    cfg = OmegaConf.merge(cfg_base, OmegaConf.load(params.cfg))
    cfg = get_module_config(cfg, cfg_assets.CONFIG_FOLDER)
    cfg = OmegaConf.merge(cfg, cfg_assets)
    if params.opts:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(params.opts))

    if params.batch_size is not None:
        cfg.TRAIN.BATCH_SIZE = params.batch_size
    if params.device is not None:
        cfg.DEVICE = params.device
    if params.num_nodes is not None:
        cfg.NUM_NODES = params.num_nodes

    if phase == "test" or params.nodebug:
        cfg.DEBUG = False
    if cfg.DEBUG:
        cfg.NAME = "debug--" + cfg.NAME
        cfg.LOGGER.WANDB.params.offline = True
        cfg.LOGGER.VAL_EVERY_STEPS = 1
    return resume_config(cfg)
