"""Lightning data module for the MaDiS sign-language datasets."""

import torch

from mGPT.utils.human_models import get_coord

from . import BASEDataModule
from .sign_dataset import SignDataset
from .utils import sign_collate


class H2SDataModule(BASEDataModule):
    def __init__(self, cfg, **_):
        super().__init__(collate_fn=sign_collate)
        self.cfg = cfg

        dataset_cfg = cfg.DATASET.H2S
        mean = torch.load(
            dataset_cfg.MEAN_PATH, map_location="cpu", weights_only=True)
        std = torch.load(
            dataset_cfg.STD_PATH, map_location="cpu", weights_only=True)
        # Drop lower-body parameters, then remove shape while retaining face.
        mean = mean[36:]
        std = std[36:]
        mean = torch.cat([mean[:-20], mean[-10:]])
        std = torch.cat([std[:-20], std[-10:]])

        self.save_hyperparameters({
            "data_root": dataset_cfg.ROOT,
            "csl_root": dataset_cfg.CSL_ROOT,
            "phoenix_root": dataset_cfg.PHOENIX_ROOT,
            "dataset_name": dataset_cfg.DATASET_NAME,
            "mean": mean,
            "std": std,
            "max_motion_length": dataset_cfg.MAX_MOTION_LEN,
            "min_motion_length": dataset_cfg.MIN_MOTION_LEN,
            "unit_length": dataset_cfg.UNIT_LEN,
            "code_path": cfg.DATASET.CODE_PATH,
            "stage": cfg.TRAIN.STAGE,
            "pred_data_dir": dataset_cfg.get("pred_data_dir"),
        }, logger=False)

        self.Dataset = SignDataset
        self.DatasetEval = SignDataset
        self.njoints = 55
        self.nfeats = 133
        self.dim_per_joint = 3
        cfg.DATASET.NFEATS = self.nfeats

    def feats2joints(self, features, return_vert=True):
        mean = self.hparams.mean.to(features)
        std = self.hparams.std.to(features)
        features = features * std + mean
        batch_size, frames = features.shape[:2]
        lower_body = torch.zeros(
            batch_size, frames, 36, dtype=features.dtype, device=features.device)
        features = torch.cat([lower_body, features], dim=-1).reshape(
            batch_size * frames, -1)
        shape = torch.tensor(
            [-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
             0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842],
            dtype=features.dtype,
            device=features.device,
        ).expand(batch_size * frames, -1)
        return get_coord(
            root_pose=features[..., :3],
            body_pose=features[..., 3:66],
            lhand_pose=features[..., 66:111],
            rhand_pose=features[..., 111:156],
            jaw_pose=features[..., 156:159],
            shape=shape,
            expr=features[..., 159:169],
            return_vert=return_vert,
        )

    def normalize(self, features):
        return (features - self.hparams.mean.to(features)) / self.hparams.std.to(features)

    def denormalize(self, features):
        return features * self.hparams.std.to(features) + self.hparams.mean.to(features)
