"""Datasets used by MaDiS for pretraining, fine-tuning, and evaluation."""

import gzip
import json
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .sign_loader import (
    load_csl_sample,
    load_h2s_sample,
    load_phoenix_sample,
)


BAD_HOW2SIGN_IDS = {
    "0DU7wWLK-QU_0-8-rgb_front", "0ICZi26jdaQ_28-5-rgb_front",
    "0vNfEYst_tQ_11-8-rgb_front", "13X0vEMNm7M_8-5-rgb_front",
    "14weIYQswlE_23-8-rgb_front", "1B56XMJ-j1Q_13-8-rgb_front",
    "1P0oKY4FNyI_0-8-rgb_front", "1dpRaxOTfZs_0-8-rgb_front",
    "1ei1kVTw23A_29-8-rgb_front", "1spCnuBmWYk_0-8-rgb_front",
    "2-vXO7MMLJc_0-5-rgb_front", "21PbS6wnHtY_0-5-rgb_front",
    "3tyfxL2wO-M_0-8-rgb_front", "BpYDl3AO4B8_0-1-rgb_front",
    "CH7AviIr0-0_14-8-rgb_front", "CJ8RyW9pzKU_6-8-rgb_front",
    "D0T7ho08Q3o_25-2-rgb_front", "Db5SUQvNsHc_18-1-rgb_front",
    "Eh697LCFjTw_0-3-rgb_front", "F-p1IdedNbg_23-8-rgb_front",
    "aUBQCNegrYc_13-1-rgb_front", "cvn7htBA8Xc_9-8-rgb_front",
    "czBrBQgZIuc_19-5-rgb_front", "dbSAB8F8GYc_11-9-rgb_front",
    "doMosV-zfCI_7-2-rgb_front", "dvBdWGLzayI_10-8-rgb_front",
    "eBrlZcccILg_26-3-rgb_front", "39FN42e41r0_17-1-rgb_front",
    "a4Nxq0QV_WA_9-3-rgb_front", "fzrJBu2qsM8_11-8-rgb_front",
    "g3Cc_1-V31U_12-3-rgb_front",
}


class SignDataset(Dataset):
    """Unified loader for the three datasets evaluated in the paper."""

    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        dataset_name,
        csl_root,
        phoenix_root,
        max_motion_length=400,
        min_motion_length=40,
        unit_length=4,
        code_path="TOKENS_h2s_csl_phoenix",
        stage="lm_instruct",
        pred_data_dir=None,
        **_,
    ):
        if dataset_name not in {"combined", "how2sign", "csl", "phoenix"}:
            raise ValueError(
                "MaDiS supports combined, how2sign, csl, and phoenix; "
                f"received {dataset_name!r}")
        self.data_root = Path(data_root)
        self.csl_root = Path(csl_root)
        self.phoenix_root = Path(phoenix_root)
        self.split = split
        self.mean = torch.as_tensor(mean).cpu().numpy()
        self.std = torch.as_tensor(std).cpu().numpy()
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length
        self.code_root = self.data_root / code_path
        self.pred_data_dir = Path(pred_data_dir) if pred_data_dir else None

        template_name = (
            "template_pretrain.json"
            if stage == "lm_pretrain" else "template_instructions.json"
        )
        template_path = (
            Path(__file__).resolve().parents[2]
            / "prepare" / "instructions" / template_name
        )
        with open(template_path, encoding="utf-8") as handle:
            templates = json.load(handle)
        self.tasks = [
            subtask
            for task in templates.values()
            for subtask in task.values()
        ]
        # Evaluation must contain each sequence exactly once. Instruction
        # diversity is a training augmentation, not an evaluation multiplier.
        if split != "train":
            self.tasks = self.tasks[:1]

        self.samples = []
        if dataset_name in {"combined", "how2sign"}:
            self._load_how2sign()
        if dataset_name in {"combined", "csl"}:
            self._load_csl()
        if dataset_name in {"combined", "phoenix"}:
            self._load_phoenix()

        print(
            f"Loaded {len(self.samples)} {dataset_name} samples "
            f"for the {split} split")

    @staticmethod
    def _is_main_process():
        return (
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )

    def _append(self, sample, loader, root):
        poses, text, name, tokens = loader(
            sample,
            root,
            code_path=self.code_root,
            need_pose=True,
            need_code=True,
        )
        if poses is None or tokens is None:
            return
        self.samples.append({
            **sample,
            "name": name,
            "text": text,
            "poses": poses,
            "code": tokens,
        })

    def _load_how2sign(self):
        csv_path = (
            self.data_root / self.split / "re_aligned"
            / f"how2sign_realigned_{self.split}_preprocessed_fps.csv"
        )
        annotations = pd.read_csv(csv_path)
        annotations = annotations[
            annotations["END_REALIGNED"] - annotations["START_REALIGNED"] < 30
        ]
        pose_root = self.data_root / "poses_video_level" / self.split / "poses"
        for _, row in tqdm(
            annotations.iterrows(),
            total=len(annotations),
            desc=f"How2Sign {self.split}",
            disable=not self._is_main_process(),
        ):
            sample = {
                "name": row["SENTENCE_NAME"],
                "fps": row["fps"],
                "text": row["SENTENCE"],
                "src": "how2sign",
            }
            if sample["name"] not in BAD_HOW2SIGN_IDS:
                self._append(sample, load_h2s_sample, pose_root)

    def _load_csl(self):
        annotation_path = self.csl_root / f"csl_clean.{self.split}"
        with gzip.open(annotation_path, "rb") as handle:
            annotations = pickle.load(handle)
        for item in tqdm(
            annotations,
            desc=f"CSL-Daily {self.split}",
            disable=not self._is_main_process(),
        ):
            sample = deepcopy(item)
            sample["src"] = "csl"
            self._append(sample, load_csl_sample, self.csl_root)

    def _load_phoenix(self):
        split_name = "dev" if self.split == "val" else self.split
        annotation_path = self.phoenix_root / f"phoenix14t.{split_name}"
        with gzip.open(annotation_path, "rb") as handle:
            annotations = pickle.load(handle)
        for item in tqdm(
            annotations,
            desc=f"Phoenix-2014T {self.split}",
            disable=not self._is_main_process(),
        ):
            sample = deepcopy(item)
            sample["src"] = "phoenix"
            sample["_split"] = split_name
            self._append(sample, load_phoenix_sample, self.phoenix_root)

    def __len__(self):
        return len(self.samples) * len(self.tasks)

    def _prediction(self, name):
        filename = Path(name).name + ".pkl"
        matches = sorted(self.pred_data_dir.glob(f"test_rank_*/{filename}"))
        if not matches:
            raise FileNotFoundError(
                f"No generated prediction named {filename} under "
                f"{self.pred_data_dir}")
        with open(matches[0], "rb") as handle:
            return pickle.load(handle)["feats_rst"]

    def __getitem__(self, index):
        sample_index = index % len(self.samples)
        task_index = index // len(self.samples)
        sample = self.samples[sample_index]
        poses = (
            self._prediction(sample["name"])
            if self.pred_data_dir else sample["poses"]
        )
        poses = np.asarray(poses)
        if not self.pred_data_dir:
            poses = (poses - self.mean) / (self.std + 1e-10)

        length = len(poses)
        if length < self.min_motion_length:
            positions = np.linspace(
                0, length - 1, self.min_motion_length, dtype=int)
            poses = poses[positions]
        elif length > self.max_motion_length:
            positions = np.linspace(
                0, length - 1, self.max_motion_length, dtype=int)
            poses = poses[positions]
        else:
            usable = (length // self.unit_length) * self.unit_length
            start = (length - usable) // 2
            poses = poses[start:start + usable]

        tokens = torch.as_tensor(sample["code"], dtype=torch.long)
        text = sample["text"]
        return (
            text,
            torch.as_tensor(poses, dtype=torch.float32),
            len(poses),
            sample["name"],
            None,
            len(tokens),
            tokens,
            [text],
            self.tasks[task_index],
            sample["src"],
        )
