# SPDX-License-Identifier: Apache-2.0
"""Codec features to smplx_motion rows: joint layout, rotations and the root path."""

import math

import torch

from sglang_omni.models.miburi.formats import (
    EXPRESSION_FEATURES,
    FEATURES_PER_FRAME,
    POSE_FEATURES,
    ROOT_FEATURES,
)
from sglang_omni.models.miburi.motion import (
    ROOT_START_POSITION_M,
    BodyPartMasks,
    axis_angles,
    smplx_frames,
)

ROOT_TRANSLATION = POSE_FEATURES + EXPRESSION_FEATURES
EYES = (23, 24)


def test_the_three_codecs_drive_every_smplx_joint_but_the_eyes_once():
    masks = BodyPartMasks.smplx()
    driven = masks.upper.int() + masks.lower.int() + masks.face.int()

    assert [
        int(mask.sum()) // 3 for mask in (masks.upper, masks.lower, masks.face)
    ] == [
        43,
        9,
        1,
    ]
    assert [joint for joint in range(55) if driven[3 * joint] == 0] == list(EYES)
    assert int(driven.max()) == 1


def test_rotations_survive_6d_and_come_back_as_axis_angles():
    generator = torch.Generator().manual_seed(0)
    directions = torch.nn.functional.normalize(
        torch.randn(64, 3, generator=generator), dim=-1
    )
    rotations = directions * torch.rand(64, 1, generator=generator) * (0.95 * math.pi)
    rotations[0] = 0.0
    skew = torch.zeros(64, 3, 3)
    skew[:, 0, 1], skew[:, 0, 2], skew[:, 1, 2] = (
        -rotations[:, 2],
        rotations[:, 1],
        -rotations[:, 0],
    )
    matrices = torch.linalg.matrix_exp(skew - skew.transpose(1, 2))

    recovered = axis_angles(matrices[:, :2].reshape(1, 1, 64 * 6), 64)

    torch.testing.assert_close(recovered.view(64, 3), rotations, atol=1e-4, rtol=0)


def test_each_frames_root_is_the_previous_one_moved_by_its_velocity():
    velocities = torch.tensor([[1.0, 0.0, -2.0], [0.5, 0.25, 0.0]])
    lower = torch.zeros(1, 2, 9 * 6 + ROOT_FEATURES + 4)
    lower[0, :, 9 * 6 : 9 * 6 + ROOT_FEATURES] = velocities

    rows, next_start = smplx_frames(
        torch.zeros(1, 2, 43 * 6),
        lower,
        torch.zeros(1, 2, 6 + EXPRESSION_FEATURES),
        BodyPartMasks.smplx(),
        torch.tensor(ROOT_START_POSITION_M),
        0.04,
    )

    start = torch.tensor(ROOT_START_POSITION_M)
    assert rows.shape == (2, FEATURES_PER_FRAME)
    torch.testing.assert_close(
        torch.from_numpy(rows[:, ROOT_TRANSLATION : ROOT_TRANSLATION + ROOT_FEATURES]),
        torch.stack([start, start + 0.04 * velocities[0]]),
    )
    torch.testing.assert_close(
        next_start, (start + 0.04 * velocities[0] + 0.04 * velocities[1])[None]
    )
    assert (rows[:, ROOT_TRANSLATION + ROOT_FEATURES :] == velocities.numpy()).all()
