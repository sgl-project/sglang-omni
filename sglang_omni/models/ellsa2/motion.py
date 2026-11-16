# SPDX-License-Identifier: Apache-2.0
"""Between ELLSA 2 codes and motion: the style clip's codes, and each block's codes as smplh_flame_motion rows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sglang_omni.models.ellsa2.generator import BLOCK_STEPS
from sglang_omni.models.ellsa2.rvq import RVQVAE, FlameTokenizer, StreamingDecoder

# SMPL-H body joints after the root that the upper-body tokenizer covers: spine, neck, collars, head, shoulders, elbows, wrists.
UPPER_JOINTS = (2, 5, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
BODY_JOINTS = 21
HAND_JOINTS = 15
ROTATION_6D = 6
UPPER_CODES = slice(0, 6)
HAND_CODES = slice(6, 12)
NONMOUTH_CODES = slice(12, 16)
MOUTH_CODES = slice(16, 18)
HEAD_ROTATION = slice(100, 103)
# The release's style reference: 612 SMPL-H and FLAME frames from the clip's start.
STYLE_FRAMES = 612
NORMALIZATION_PATH = Path(__file__).with_name("normalization.json")


# The rotation conversions below are PyTorch3D's (Copyright (c) Facebook, Inc. and its affiliates, BSD-3-Clause), as the release vendors them.
def sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(x)
    positive = x > 0
    result[positive] = torch.sqrt(x[positive])
    return result


def copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.where((a < 0) != (b < 0), -a, a)


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    m00, m11, m22 = matrix[..., 0, 0], matrix[..., 1, 1], matrix[..., 2, 2]
    quaternions = torch.stack(
        (
            0.5 * sqrt_positive_part(1 + m00 + m11 + m22),
            copysign(
                0.5 * sqrt_positive_part(1 + m00 - m11 - m22),
                matrix[..., 2, 1] - matrix[..., 1, 2],
            ),
            copysign(
                0.5 * sqrt_positive_part(1 - m00 + m11 - m22),
                matrix[..., 0, 2] - matrix[..., 2, 0],
            ),
            copysign(
                0.5 * sqrt_positive_part(1 - m00 - m11 + m22),
                matrix[..., 1, 0] - matrix[..., 0, 1],
            ),
        ),
        -1,
    )
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    return quaternions[..., 1:] / sin_half_over_angle(angles, half_angles)


def sin_half_over_angle(
    angles: torch.Tensor, half_angles: torch.Tensor
) -> torch.Tensor:
    """sin(angle / 2) / angle, by its series below 1e-6 rad."""
    small = angles.abs() < 1e-6
    ratio = torch.empty_like(angles)
    ratio[~small] = torch.sin(half_angles[~small]) / angles[~small]
    ratio[small] = 0.5 - (angles[small] * angles[small]) / 48
    return ratio


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_over_angle(angles, half_angles)],
        dim=-1,
    )
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    matrix = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return matrix.reshape(quaternions.shape[:-1] + (3, 3))


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    return torch.stack((b1, b2, torch.cross(b1, b2, dim=-1)), dim=-2)


def axis_angles(rotations_6d: torch.Tensor, num_joints: int) -> torch.Tensor:
    """[frames, num_joints * 6] → [frames, num_joints, 3]."""
    matrices = rotation_6d_to_matrix(rotations_6d.reshape(-1, num_joints, ROTATION_6D))
    return matrix_to_axis_angle(matrices).reshape(-1, num_joints, 3)


def rotations_6d(axis_angle: torch.Tensor) -> torch.Tensor:
    """[frames, joints, 3] → [frames, joints * 6]."""
    matrices = axis_angle_to_matrix(axis_angle)
    return matrices[..., :2, :].clone().reshape(axis_angle.shape[0], -1)


@dataclass(frozen=True, kw_only=True)
class Normalization:
    """The release's training statistics: 6D upper-body and hand rotations (torch, CPU) and FLAME motion (numpy)."""

    upper_mean: torch.Tensor
    upper_std: torch.Tensor
    left_hand_mean: torch.Tensor
    left_hand_std: torch.Tensor
    right_hand_mean: torch.Tensor
    right_hand_std: torch.Tensor
    flame_mean: np.ndarray
    flame_std: np.ndarray

    @classmethod
    def release(cls) -> Normalization:
        values = json.loads(NORMALIZATION_PATH.read_text())
        return cls(
            **{
                name: (
                    np.asarray(value, dtype=np.float32)
                    if name.startswith("flame")
                    else torch.tensor(value, dtype=torch.float32)
                )
                for name, value in values.items()
            }
        )


@dataclass(frozen=True, kw_only=True)
class Tokenizers:
    upper: RVQVAE
    hands: RVQVAE
    flame: FlameTokenizer


def pose_key(pose: np.lib.npyio.NpzFile, name: str) -> str:
    return name if name in pose else f"smplh:{name}"


@torch.no_grad()
def style_codes(
    pose_path: Path,
    flame_path: Path,
    tokenizers: Tokenizers,
    normalization: Normalization,
) -> torch.Tensor:
    """The style clip's first STYLE_FRAMES frames as codes [steps, NUM_CODEBOOKS], cut to whole blocks."""
    with np.load(pose_path) as pose:
        body = torch.from_numpy(
            pose[pose_key(pose, "body_pose")][:STYLE_FRAMES]
        ).float()
        left = torch.from_numpy(
            pose[pose_key(pose, "left_hand_pose")][:STYLE_FRAMES]
        ).float()
        right = torch.from_numpy(
            pose[pose_key(pose, "right_hand_pose")][:STYLE_FRAMES]
        ).float()
    upper = rotations_6d(body.reshape(STYLE_FRAMES, BODY_JOINTS, 3)[:, UPPER_JOINTS, :])
    upper = (upper - normalization.upper_mean) / normalization.upper_std
    left = (
        rotations_6d(left.reshape(STYLE_FRAMES, HAND_JOINTS, 3))
        - normalization.left_hand_mean
    ) / normalization.left_hand_std
    right = (
        rotations_6d(right.reshape(STYLE_FRAMES, HAND_JOINTS, 3))
        - normalization.right_hand_mean
    ) / normalization.right_hand_std
    with np.load(flame_path) as flame:
        face = np.array(flame["motioncode"][:STYLE_FRAMES], dtype=np.float32, copy=True)
    face[:, HEAD_ROTATION] -= face[:, HEAD_ROTATION].mean(axis=0, keepdims=True)
    face = torch.from_numpy(
        (face - normalization.flame_mean) / (normalization.flame_std + 1e-8)
    )
    device = tokenizers.upper.quantizer.layers[0].codebook.device
    codes = torch.cat(
        [
            tokenizers.upper.encode(upper.unsqueeze(0).to(device))[0, :, :6],
            tokenizers.hands.encode(
                torch.cat([left, right], dim=1).unsqueeze(0).to(device)
            )[0, :, :6],
            tokenizers.flame.encode(face.unsqueeze(0).to(device))[0],
        ],
        dim=1,
    )
    return codes[: codes.shape[0] // BLOCK_STEPS * BLOCK_STEPS]


class MotionDecoder:
    """One Session's decoders: each block's codes → its smplh_flame_motion rows, with the bounded RVQ context of earlier blocks."""

    def __init__(self, tokenizers: Tokenizers, normalization: Normalization) -> None:
        self.upper = StreamingDecoder(tokenizers.upper)
        self.hands = StreamingDecoder(tokenizers.hands)
        self.nonmouth = StreamingDecoder(tokenizers.flame.nonmouth_vq)
        self.mouth = StreamingDecoder(tokenizers.flame.mouth_vq)
        self.flame = tokenizers.flame
        self.normalization = normalization

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> np.ndarray:
        """Codes [BLOCK_STEPS, NUM_CODEBOOKS] → float32 smplh_flame_motion rows [BLOCK_STEPS * FRAMES_PER_CODE, FEATURES_PER_FRAME]."""
        upper = self.upper.decode(codes[None, :, UPPER_CODES])[0].cpu()
        hands = self.hands.decode(codes[None, :, HAND_CODES])[0].cpu()
        face = (
            self.flame.merge(
                self.nonmouth.decode(codes[None, :, NONMOUTH_CODES]),
                self.mouth.decode(codes[None, :, MOUTH_CODES]),
            )[0]
            .cpu()
            .numpy()
        )
        return self.rows(upper, hands, face)

    @staticmethod
    @torch.no_grad()
    def decode_together(
        decoders: list[MotionDecoder], codes: torch.Tensor
    ) -> list[np.ndarray]:
        """Each Session's block of codes [len(decoders), BLOCK_STEPS, NUM_CODEBOOKS] → its rows, as decode gives them, decoding the Sessions as one batch."""
        first = decoders[0]
        upper = StreamingDecoder.decode_together(
            [decoder.upper for decoder in decoders], codes[:, :, UPPER_CODES]
        ).cpu()
        hands = StreamingDecoder.decode_together(
            [decoder.hands for decoder in decoders], codes[:, :, HAND_CODES]
        ).cpu()
        face = (
            first.flame.merge(
                StreamingDecoder.decode_together(
                    [decoder.nonmouth for decoder in decoders],
                    codes[:, :, NONMOUTH_CODES],
                ),
                StreamingDecoder.decode_together(
                    [decoder.mouth for decoder in decoders], codes[:, :, MOUTH_CODES]
                ),
            )
            .cpu()
            .numpy()
        )
        # Each frame's row reads that frame alone: all Sessions' frames convert at once.
        rows = first.rows(
            upper.flatten(0, 1), hands.flatten(0, 1), face.reshape(-1, face.shape[-1])
        )
        return list(rows.reshape(len(decoders), -1, rows.shape[-1]))

    def rows(
        self, upper: torch.Tensor, hands: torch.Tensor, face: np.ndarray
    ) -> np.ndarray:
        """One block's decoded, normalized upper body, hands [frames, 6D features] and FLAME motion [frames, 108] → its smplh_flame_motion rows."""
        stats = self.normalization
        face = (face * (stats.flame_std + 1e-8) + stats.flame_mean).astype(np.float32)
        frames = upper.shape[0]
        body = torch.zeros(frames, BODY_JOINTS, 3, dtype=torch.float32)
        body[:, UPPER_JOINTS, :] = axis_angles(
            upper * stats.upper_std + stats.upper_mean, len(UPPER_JOINTS)
        )
        left = axis_angles(
            hands[:, : HAND_JOINTS * ROTATION_6D] * stats.left_hand_std
            + stats.left_hand_mean,
            HAND_JOINTS,
        )
        right = axis_angles(
            hands[:, HAND_JOINTS * ROTATION_6D :] * stats.right_hand_std
            + stats.right_hand_mean,
            HAND_JOINTS,
        )
        pose = torch.cat(
            [
                torch.zeros(frames, 3),
                body.reshape(frames, -1),
                left.reshape(frames, -1),
                right.reshape(frames, -1),
            ],
            dim=1,
        )
        return np.concatenate([pose.numpy(), face], axis=1).astype("<f4")


__all__ = [
    "MotionDecoder",
    "Normalization",
    "Tokenizers",
    "style_codes",
]
