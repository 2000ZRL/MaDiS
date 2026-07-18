"""Minimal SMPL-X utilities needed for MaDiS evaluation."""

from pathlib import Path

import numpy as np
import smplx
import torch


HUMAN_MODEL_PATH = Path("deps/smpl_models")


class SMPLXAssets:
    def __init__(self):
        self.layer = smplx.create(
            HUMAN_MODEL_PATH,
            "smplx",
            gender="NEUTRAL",
            use_pca=False,
            use_face_contour=True,
            create_global_orient=False,
            create_body_pose=False,
            create_left_hand_pose=False,
            create_right_hand_pose=False,
            create_jaw_pose=False,
            create_leye_pose=False,
            create_reye_pose=False,
            create_betas=False,
            create_expression=False,
            create_transl=False,
        )
        self.vertex_num = 10475
        self.orig_hand_regressor = self._make_hand_regressor()
        # Upper-body and sparse face indices in the native SMPL-X joint set.
        self.joint_part2idx = {
            "upper_body": [12, 16, 17, 18, 19, 20, 21, 59, 58, 57, 56, 55]
        }

    def _make_hand_regressor(self):
        regressor = self.layer.J_regressor.numpy()
        identity = np.eye(self.vertex_num)
        left = np.concatenate((
            regressor[[20, 37, 38, 39]], identity[5361, None],
            regressor[[25, 26, 27]], identity[4933, None],
            regressor[[28, 29, 30]], identity[5058, None],
            regressor[[34, 35, 36]], identity[5169, None],
            regressor[[31, 32, 33]], identity[5286, None],
        ))
        right = np.concatenate((
            regressor[[21, 52, 53, 54]], identity[8079, None],
            regressor[[40, 41, 42]], identity[7669, None],
            regressor[[43, 44, 45]], identity[7794, None],
            regressor[[49, 50, 51]], identity[7905, None],
            regressor[[46, 47, 48]], identity[8022, None],
        ))
        return {
            "left": torch.from_numpy(left).float(),
            "right": torch.from_numpy(right).float(),
        }


smpl_x = SMPLXAssets()


def get_coord(
    root_pose,
    body_pose,
    lhand_pose,
    rhand_pose,
    jaw_pose,
    shape,
    expr,
    return_vert=True,
):
    """Decode SMPL-X parameters into vertices and joints."""
    layer = smpl_x.layer.to(root_pose.device)
    eye_pose = torch.zeros(
        (len(root_pose), 3), dtype=root_pose.dtype, device=root_pose.device)
    output = layer(
        betas=shape,
        body_pose=body_pose,
        global_orient=root_pose,
        right_hand_pose=rhand_pose,
        left_hand_pose=lhand_pose,
        jaw_pose=jaw_pose,
        leye_pose=eye_pose,
        reye_pose=eye_pose,
        expression=expr,
    )
    return output.vertices if return_vert else None, output.joints
