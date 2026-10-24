# SPDX-License-Identifier: Apache-2.0
"""CPU checks for Ming-TTS packed acoustic QKV loading."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.ming_omni.talker.talker_module.packed_qkv import (
    PACKED_QKV_SHARD_IDS,
    PackedQKVLinear,
)

QKV_PREFIX = "flowloss.cfm.model.blocks.0.attn.to_qkv"


def test_ming_tts_loads_split_qkv_shards_into_bf16_projection() -> None:
    from sglang_omni.models.ming_tts.sglang_model import MingTTSSGLangModel

    projection = PackedQKVLinear(8, 8).to(dtype=torch.bfloat16)
    owner = SimpleNamespace(
        named_parameters=lambda: projection.named_parameters(prefix=QKV_PREFIX),
        llm_config=SimpleNamespace(num_experts=0),
    )
    checkpoint = []
    expected_weights = []
    expected_biases = []
    for index, shard_id in enumerate(PACKED_QKV_SHARD_IDS, start=1):
        weight = torch.arange(64, dtype=torch.bfloat16).reshape(8, 8) / index
        bias = torch.arange(8, dtype=torch.bfloat16) / index
        checkpoint.extend(
            (
                (QKV_PREFIX.replace("to_qkv", f"to_{shard_id}") + ".weight", weight),
                (QKV_PREFIX.replace("to_qkv", f"to_{shard_id}") + ".bias", bias),
            )
        )
        expected_weights.append(weight)
        expected_biases.append(bias)

    MingTTSSGLangModel.load_weights(owner, checkpoint)

    torch.testing.assert_close(projection.weight, torch.cat(expected_weights))
    torch.testing.assert_close(projection.bias, torch.cat(expected_biases))
    assert owner.weight_load_report.loaded_shards == {
        f"{QKV_PREFIX}.weight": list(PACKED_QKV_SHARD_IDS),
        f"{QKV_PREFIX}.bias": list(PACKED_QKV_SHARD_IDS),
    }

    inputs = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8) / 8
    expected = torch.cat(
        [
            F.linear(inputs, weight, bias)
            for weight, bias in zip(expected_weights, expected_biases)
        ],
        dim=-1,
    )
    torch.testing.assert_close(projection(inputs), expected, rtol=0.01, atol=0.01)


def test_ming_tts_accepts_full_packed_qkv_checkpoint() -> None:
    from sglang_omni.models.ming_tts.sglang_model import MingTTSSGLangModel

    projection = PackedQKVLinear(4, 4)
    owner = SimpleNamespace(
        named_parameters=lambda: projection.named_parameters(prefix=QKV_PREFIX),
        llm_config=SimpleNamespace(num_experts=0),
    )
    weight = torch.arange(48, dtype=torch.float32).reshape(12, 4)
    bias = torch.arange(12, dtype=torch.float32)

    MingTTSSGLangModel.load_weights(
        owner,
        [(f"{QKV_PREFIX}.weight", weight), (f"{QKV_PREFIX}.bias", bias)],
    )

    torch.testing.assert_close(projection.weight, weight)
    torch.testing.assert_close(projection.bias, bias)
    assert owner.weight_load_report.loaded_shards == {
        f"{QKV_PREFIX}.weight": list(PACKED_QKV_SHARD_IDS),
        f"{QKV_PREFIX}.bias": list(PACKED_QKV_SHARD_IDS),
    }


def test_ming_tts_rejects_incomplete_packed_qkv_checkpoint() -> None:
    from sglang_omni.models.ming_tts.sglang_model import MingTTSSGLangModel

    projection = PackedQKVLinear(4, 4)
    owner = SimpleNamespace(
        named_parameters=lambda: projection.named_parameters(prefix=QKV_PREFIX),
        llm_config=SimpleNamespace(num_experts=0),
    )
    checkpoint = [
        (f"{QKV_PREFIX.replace('to_qkv', f'to_{shard_id}')}.{suffix}", tensor)
        for shard_id in ("q", "v")
        for suffix, tensor in (
            ("weight", torch.zeros(4, 4)),
            ("bias", torch.zeros(4)),
        )
    ]

    with pytest.raises(RuntimeError, match="incomplete packed weight.*missing k"):
        MingTTSSGLangModel.load_weights(owner, checkpoint)
