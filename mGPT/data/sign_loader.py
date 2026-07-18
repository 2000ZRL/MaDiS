"""SMPL-X pose and token loading for the three MaDiS datasets."""

import pickle
from pathlib import Path

import numpy as np


SMPLX_KEYS = (
    "smplx_root_pose",
    "smplx_body_pose",
    "smplx_lhand_pose",
    "smplx_rhand_pose",
    "smplx_jaw_pose",
    "smplx_shape",
    "smplx_expr",
)


def _pose_features(poses):
    features = np.concatenate([poses[key] for key in SMPLX_KEYS], axis=1)
    # Retain upper body, hands, jaw, and expression in axis-angle format.
    features = features[:, 36:]
    return np.concatenate([features[:, :-20], features[:, -10:]], axis=1)


def _load_tokens(code_path, dataset, name, split=None):
    code_path = Path(code_path)
    candidates = []
    if split is not None:
        candidates.append(code_path / dataset / split / f"{name}.npy")
    candidates.extend((
        code_path / dataset / f"{name}.npy",
        code_path / f"{name}.npy",
    ))
    for candidate in candidates:
        if candidate.is_file():
            return np.load(candidate)[0]
    raise FileNotFoundError(
        f"No token file for {dataset}/{name}; checked "
        + ", ".join(map(str, candidates)))


def load_h2s_sample(
    annotation, data_dir, need_pose=True, code_path=None, need_code=False
):
    name = annotation["name"]
    pose_root = Path(data_dir)
    if "split" in annotation:
        pose_root = pose_root / annotation["split"] / "poses"
    pose_file = pose_root / f"{name}.pkl"
    pose_dir = pose_root / name

    if pose_file.is_file():
        with open(pose_file, "rb") as handle:
            poses = pickle.load(handle)
    elif pose_dir.is_dir():
        frame_files = sorted(
            pose_dir.glob("*_3D.pkl"),
            key=lambda path: int(path.name.rsplit("_", 2)[-2]),
        )
        if not frame_files:
            raise FileNotFoundError(f"No SMPL-X frames found in {pose_dir}")
        frames = []
        for frame_file in frame_files:
            with open(frame_file, "rb") as handle:
                frames.append(pickle.load(handle))
        poses = {
            key: np.stack([frame[key] for frame in frames])
            for key in SMPLX_KEYS
        }
    else:
        raise FileNotFoundError(pose_file)

    features = _pose_features(poses)
    fps = float(annotation["fps"])
    frame_indices = np.arange(len(features))
    if fps > 24:
        target_count = int(24 * len(frame_indices) / fps)
        frame_indices = np.linspace(
            0, len(frame_indices) - 1, target_count, dtype=int)
    if len(frame_indices) < 4:
        return None, None, None, None
    features = features[frame_indices] if need_pose else None
    tokens = _load_tokens(code_path, "how2sign", name) if need_code else None
    return features, annotation["text"], name, tokens


def load_csl_sample(
    annotation, data_dir, need_pose=True, code_path=None, need_code=False
):
    name = annotation["name"]
    with open(Path(data_dir) / "poses_video_level" / f"{name}.pkl", "rb") as handle:
        poses = pickle.load(handle)
    features = _pose_features(poses)
    if len(features) < 4:
        return None, None, None, None
    tokens = _load_tokens(code_path, "csl", name) if need_code else None
    return features if need_pose else None, annotation["text"], name, tokens


def load_phoenix_sample(
    annotation, data_dir, need_pose=True, code_path=None, need_code=False
):
    name = annotation["name"]
    with open(Path(data_dir) / "poses_video_level" / f"{name}.pkl", "rb") as handle:
        poses = pickle.load(handle)
    features = _pose_features(poses)
    if len(features) < 4:
        return None, None, None, None
    tokens = (
        _load_tokens(code_path, "phoenix", name, annotation.get("_split"))
        if need_code else None
    )
    return features if need_pose else None, annotation["text"], name, tokens
