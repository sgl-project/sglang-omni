# SPDX-License-Identifier: Apache-2.0
import copy

import pytest
import torch
from torch import nn

from sglang_omni.models.voxcpm2.components.minicpm import (
    MiniCPM4Config,
    MiniCPMLongRoPE,
    RopeScalingConfig,
    align_rope_buffers,
    apply_rotary_pos_emb,
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_loaded_rope_matches_upstream_model_cast(dtype):
    config = MiniCPM4Config(
        hidden_size=8,
        intermediate_size=16,
        max_position_embeddings=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        rope_theta=10000,
        scale_depth=1,
        rope_scaling=RopeScalingConfig(
            type="longrope",
            long_factor=[1.0, 1.0],
            short_factor=[1.0, 1.0],
            original_max_position_embeddings=8,
        ),
    )
    module = nn.Module()
    module.proj = nn.Linear(8, 8)
    module.rope = MiniCPMLongRoPE(config)
    upstream = copy.deepcopy(module).to(dtype=dtype)
    # Loading weights does not restore the non-persistent RoPE tables.
    module.load_state_dict(upstream.state_dict(), assign=True)
    assert module.rope.cos_cached.dtype == torch.float32
    align_rope_buffers(module)
    for name, expected in upstream.named_buffers():
        actual = dict(module.named_buffers())[name]
        assert actual.dtype == dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    q = torch.randn(1, 2, 4, 4).to(dtype)
    positions = torch.arange(4)
    actual = apply_rotary_pos_emb(q, q, *module.rope(positions))
    expected = apply_rotary_pos_emb(q, q, *upstream.rope(positions))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
