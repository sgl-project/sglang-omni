# SPDX-License-Identifier: Apache-2.0
"""smplx_motion rows built from the three motion codecs' outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional

from sglang_omni.models.miburi.formats import POSE_FEATURES, ROOT_FEATURES

ROTATION_6D_FEATURES = 6
AXIS_ANGLE_FEATURES = 3
# Where a session's root starts, in meters with y up, matching the checkpoint's own demo.
ROOT_START_POSITION_M = (0.0, 1.3, 0.0)
# The SMPL-X joints each codec drives, in SMPL-X order (0 pelvis, 22 jaw, 23 and 24 the eyes, 25-39 and 40-54 the hands): upper body with both hands, lower body with the pelvis, the jaw. Nothing drives the eyes.
UPPER_BODY_JOINTS = (3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, *range(25, 55))
LOWER_BODY_JOINTS = (0, 1, 2, 4, 5, 7, 8, 10, 11)
FACE_JOINTS = (22,)
# Below this rotation angle, sin(angle / 2) / angle takes its series expansion.
SMALL_ANGLE_RAD = 1e-6


@dataclass(frozen=True, kw_only=True)
class BodyPartMasks:
    """Per body part, a mask over the 165 SMPL-X pose features; a codec's joints are that part's joints in SMPL-X order."""

    upper: torch.Tensor
    lower: torch.Tensor
    face: torch.Tensor

    @classmethod
    def smplx(cls) -> BodyPartMasks:
        def mask(joints: tuple[int, ...]) -> torch.Tensor:
            features = torch.zeros(POSE_FEATURES, dtype=torch.bool)
            for joint in joints:
                features[
                    AXIS_ANGLE_FEATURES * joint : AXIS_ANGLE_FEATURES * (joint + 1)
                ] = True
            return features

        return cls(
            upper=mask(UPPER_BODY_JOINTS),
            lower=mask(LOWER_BODY_JOINTS),
            face=mask(FACE_JOINTS),
        )


def rotation_matrices(rotations_6d: torch.Tensor) -> torch.Tensor:
    """[..., 6] → [..., 3, 3]: Gram-Schmidt on the two 3-vectors gives the first two rows, their cross product the third."""
    first = functional.normalize(rotations_6d[..., :3], dim=-1)
    second = rotations_6d[..., 3:]
    second = functional.normalize(
        second - (first * second).sum(dim=-1, keepdim=True) * first, dim=-1
    )
    return torch.stack(
        (first, second, torch.linalg.cross(first, second, dim=-1)), dim=-2
    )


def matrix_axis_angles(matrices: torch.Tensor) -> torch.Tensor:
    """[..., 3, 3] rotations → [..., 3] axis-angles, through the unit quaternion; each vector component takes the sign of the matrix's antisymmetric part."""
    m00, m11, m22 = matrices[..., 0, 0], matrices[..., 1, 1], matrices[..., 2, 2]

    def half_root(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.where(x > 0, torch.sqrt(x), torch.zeros_like(x))

    signs = (
        matrices[..., 2, 1] - matrices[..., 1, 2],
        matrices[..., 0, 2] - matrices[..., 2, 0],
        matrices[..., 1, 0] - matrices[..., 0, 1],
    )
    magnitudes = (
        half_root(1 + m00 - m11 - m22),
        half_root(1 - m00 + m11 - m22),
        half_root(1 - m00 - m11 + m22),
    )
    quaternions = torch.stack(
        [
            half_root(1 + m00 + m11 + m22),
            *(
                torch.where(sign < 0, -magnitude, magnitude)
                for magnitude, sign in zip(magnitudes, signs, strict=True)
            ),
        ],
        dim=-1,
    )
    half_angles = torch.atan2(
        torch.linalg.vector_norm(quaternions[..., 1:], dim=-1, keepdim=True),
        quaternions[..., :1],
    )
    angles = 2 * half_angles
    sin_half_angle_over_angle = torch.where(
        angles.abs() < SMALL_ANGLE_RAD,
        0.5 - angles * angles / 48,
        torch.sin(half_angles) / angles,
    )
    return quaternions[..., 1:] / sin_half_angle_over_angle


def axis_angles(rotations_6d: torch.Tensor, num_joints: int) -> torch.Tensor:
    """[1, frames, num_joints * 6] 6D rotations → [frames, num_joints * 3] axis-angles."""
    num_frames = rotations_6d.shape[1]
    matrices = rotation_matrices(
        rotations_6d.reshape(1, num_frames, num_joints, ROTATION_6D_FEATURES)
    )
    return matrix_axis_angles(matrices).reshape(1, num_frames, num_joints * 3)[0]


def scatter_pose(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[frames, joints * 3] → [frames, 165], the values in the masked features and zero elsewhere."""
    pose = values.new_zeros(values.shape[0], POSE_FEATURES)
    pose[:, mask] = values
    return pose


def smplx_frames(
    upper: torch.Tensor,
    lower: torch.Tensor,
    face: torch.Tensor,
    masks: BodyPartMasks,
    root_position_m: torch.Tensor,
    frame_seconds: float,
) -> tuple[np.ndarray, torch.Tensor]:
    """The codecs' [1, frames, features] CPU outputs as smplx_motion rows; returns them and the root position the next frames start from.

    upper holds 6D rotations of the upper-body and hand joints; lower the lower-body joints, then the root velocity (and foot contacts); face the jaw, then the expression coefficients. Each frame's root is the previous frame's moved by the previous frame's velocity.
    """
    num_frames = upper.shape[1]
    num_upper_joints = int(masks.upper.sum()) // AXIS_ANGLE_FEATURES
    num_lower_joints = int(masks.lower.sum()) // AXIS_ANGLE_FEATURES
    num_face_joints = int(masks.face.sum()) // AXIS_ANGLE_FEATURES
    lower_rotation_features = num_lower_joints * ROTATION_6D_FEATURES
    face_rotation_features = num_face_joints * ROTATION_6D_FEATURES
    root_velocity = lower[
        :, :, lower_rotation_features : lower_rotation_features + ROOT_FEATURES
    ]
    root_positions = [root_position_m.reshape(1, ROOT_FEATURES)]
    for frame in range(num_frames):
        root_positions.append(
            root_positions[-1] + root_velocity[:, frame] * frame_seconds
        )
    pose = (
        scatter_pose(axis_angles(upper, num_upper_joints), masks.upper)
        + scatter_pose(
            axis_angles(face[:, :, :face_rotation_features], num_face_joints),
            masks.face,
        )
        + scatter_pose(
            axis_angles(lower[:, :, :lower_rotation_features], num_lower_joints),
            masks.lower,
        )
    )
    rows = torch.cat(
        [
            pose,
            face[0, :, face_rotation_features:],
            torch.cat(root_positions[:-1]),
            root_velocity[0],
        ],
        dim=-1,
    )
    return rows.numpy().astype("<f4"), root_positions[-1]


__all__ = [
    "ROOT_START_POSITION_M",
    "BodyPartMasks",
    "smplx_frames",
]
