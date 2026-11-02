# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_asr import sglang_model


@pytest.mark.parametrize(
    ("positions_dtype", "expect_same_object"),
    [(torch.int64, False), (torch.int32, True)],
)
def test_fused_rope_hook_feeds_the_kernel_int32_positions(
    monkeypatch: pytest.MonkeyPatch,
    positions_dtype: torch.dtype,
    expect_same_object: bool,
) -> None:
    seen: dict[str, torch.Tensor] = {}

    def fake_kernel(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
        *rest,
    ):
        seen["position_ids"] = position_ids

    monkeypatch.setattr(sglang_model, "fused_qk_norm_rope", fake_kernel)
    attention = SimpleNamespace(
        qkv_proj=lambda hidden_states: (
            torch.zeros(hidden_states.shape[0], 16, dtype=hidden_states.dtype),
            None,
        ),
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        q_norm=SimpleNamespace(variance_epsilon=1e-6, weight=torch.ones(4)),
        k_norm=SimpleNamespace(weight=torch.ones(4)),
        rope_theta=10000.0,
        rotary_emb=SimpleNamespace(is_neox_style=True),
        q_size=8,
        kv_size=4,
    )
    positions = torch.arange(3, dtype=positions_dtype)
    hidden_states = torch.zeros((3, 8), dtype=torch.bfloat16)

    query, key, value = sglang_model.fused_asr_forward_prepare_native(
        attention, positions, hidden_states
    )

    assert seen["position_ids"].dtype == torch.int32
    assert torch.equal(seen["position_ids"].to(positions_dtype), positions)
    assert (seen["position_ids"] is positions) == expect_same_object
    assert query.shape == (3, 8)
    assert key.shape == (3, 4)
    assert value.shape == (3, 4)
