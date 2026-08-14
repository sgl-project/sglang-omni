# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch

from sglang_omni.scheduling.dllm_group import (
    DllmCompanionSpec,
    DllmForwardGroup,
    DllmRequestGroupSpec,
    align_cfg_request_group,
    apply_forward_group_padding,
)


def test_group_spec_requires_physically_aligned_companions() -> None:
    spec = DllmRequestGroupSpec(
        companions=(
            DllmCompanionSpec(
                role="unconditional",
                input_ids=(0, 0, 7, 8),
                left_pad_length=2,
            ),
        ),
        algorithm_args={"cfg_scale": 4.0},
    )

    spec.validate(primary_input_length=4)

    with pytest.raises(ValueError, match="physically aligned"):
        spec.validate(primary_input_length=3)


def test_group_spec_rejects_duplicate_companion_roles() -> None:
    spec = DllmRequestGroupSpec(
        companions=(
            DllmCompanionSpec(role="unconditional", input_ids=(1, 2)),
            DllmCompanionSpec(role="unconditional", input_ids=(3, 4)),
        )
    )

    with pytest.raises(ValueError, match="unique"):
        spec.validate(primary_input_length=2)


def test_cfg_group_is_left_padded_once_at_final_request_assembly() -> None:
    conditional, group = align_cfg_request_group(
        mask_token_id=99,
        conditional_input_ids=(10, 11),
        unconditional_input_ids=(7,),
        no_image_input_ids=(6, 5, 4),
        algorithm_args={"cfg_scale": 4.0, "cfg_image_scale": 1.5},
    )

    assert conditional == (99, 10, 11)
    assert group.companions == (
        DllmCompanionSpec(
            role="unconditional", input_ids=(99, 99, 7), left_pad_length=2
        ),
        DllmCompanionSpec(role="no_image", input_ids=(6, 5, 4), left_pad_length=0),
    )
    assert group.primary_left_pad_length == 1


def test_forward_group_keeps_cpu_padding_metadata_and_updates_positions() -> None:
    forward_batch = type("ForwardBatch", (), {})()
    forward_batch.batch_size = 3
    forward_batch.forward_mode = type(
        "ForwardMode", (), {"is_extend": lambda self: True}
    )()
    forward_batch.extend_seq_lens_cpu = [3, 3, 3]
    forward_batch.positions = torch.tensor(
        [0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.int32
    )
    forward_batch.seq_lens = torch.tensor([3, 3, 3], dtype=torch.int32)
    group = DllmForwardGroup(
        group_id="request-1",
        roles=("conditional", "unconditional", "no_image"),
        left_pad_lengths=(1, 2, 0),
        algorithm_args={"cfg_scale": 4.0},
    )

    apply_forward_group_padding(forward_batch, group)

    assert forward_batch.dllm_left_pad_lens_cpu == (1, 2, 0)
    assert forward_batch.dllm_left_pad_lens.tolist() == [1, 2, 0]
    assert forward_batch.positions.view(3, 3).tolist() == [
        [0, 0, 1],
        [0, 0, 0],
        [0, 1, 2],
    ]


def test_forward_group_rejects_malformed_host_geometry() -> None:
    forward_batch = type("ForwardBatch", (), {})()
    forward_batch.batch_size = 2
    forward_batch.forward_mode = type(
        "ForwardMode", (), {"is_extend": lambda self: True}
    )()
    forward_batch.extend_seq_lens_cpu = [3, 2]
    forward_batch.positions = torch.arange(5, dtype=torch.int32)
    forward_batch.seq_lens = torch.tensor([3, 2], dtype=torch.int32)
    group = DllmForwardGroup(
        group_id="request-malformed",
        roles=("conditional", "unconditional"),
        left_pad_lengths=(0, 1),
        algorithm_args={"cfg_scale": 4.0},
    )

    with pytest.raises(RuntimeError, match="equal active-block lengths"):
        apply_forward_group_padding(forward_batch, group)
