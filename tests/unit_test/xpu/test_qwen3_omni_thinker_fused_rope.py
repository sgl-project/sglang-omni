# SPDX-License-Identifier: Apache-2.0
"""Intel XPU coverage for the Qwen3-Omni thinker's fused QK-norm plus RoPE.

Upstream refuses to fuse under MRoPE, so on XPU the text-only thinker installs its own
wrapper that fuses a prefill whose three position rows collapse.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
from sglang.srt.models.utils import apply_qk_norm

from sglang_omni.models.qwen3_omni.components import thinker_fused_rope
from sglang_omni.models.qwen3_omni.components.thinker_fused_rope import (
    install_thinker_fused_rope,
)
from sglang_omni.platforms import current_platform

requires_xpu = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="requires XPU",
)
UNFUSABLE_HEAD_DIM = 96
# Qwen3-Omni-30B-A3B-Instruct thinker text_config, the checkpoint XPU serves.
MROPE_SECTION = [24, 20, 20]
QWEN3_OMNI_ROPE_THETA = 1000000.0
QWEN3_OMNI_RMS_NORM_EPSILON = 1e-6
QWEN3_OMNI_HEAD_DIM = 128
QWEN3_OMNI_QUERY_HEADS = 32
QWEN3_OMNI_KEY_VALUE_HEADS = 4
# One bf16 ULP at the scale these activations reach; measured gap 0.031.
FUSED_ROPE_ABSOLUTE_TOLERANCE = 6.25e-2
FUSED_ROPE_RELATIVE_TOLERANCE = 1.6e-2


def one_layer_model(attention: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])


@pytest.mark.accelerator
@requires_xpu
def test_the_installed_wrapper_rotates_a_text_prefill_as_mrope_would() -> None:
    if current_platform.get_fused_qk_norm_rope_with_cos_sin_cache() is None:
        pytest.skip("this sgl_kernel build has no fused QK-norm-RoPE kernel")
    else:
        pass
    device, dtype = torch.device("xpu"), torch.bfloat16
    query_heads = QWEN3_OMNI_QUERY_HEADS
    key_value_heads = QWEN3_OMNI_KEY_VALUE_HEADS
    head_dim, token_count = QWEN3_OMNI_HEAD_DIM, 6
    query_width = query_heads * head_dim
    key_width = key_value_heads * head_dim
    rotary_embedding = MRotaryEmbedding(
        head_size=head_dim,
        rotary_dim=head_dim,
        max_position_embeddings=64,
        base=QWEN3_OMNI_ROPE_THETA,
        is_neox_style=True,
        dtype=dtype,
        mrope_section=MROPE_SECTION,
        mrope_interleaved=True,
    ).to(device)
    torch.manual_seed(0)
    query_norm = RMSNorm(head_dim, eps=QWEN3_OMNI_RMS_NORM_EPSILON).to(
        device=device, dtype=dtype
    )
    key_norm = RMSNorm(head_dim, eps=QWEN3_OMNI_RMS_NORM_EPSILON).to(
        device=device, dtype=dtype
    )
    with torch.no_grad():
        query_norm.weight.copy_(torch.randn(head_dim, dtype=dtype).to(device))
        key_norm.weight.copy_(torch.randn(head_dim, dtype=dtype).to(device))

    attention = SimpleNamespace(
        apply_qk_norm_rope=lambda *unused: pytest.fail("the wrapper fell back"),
        config=SimpleNamespace(rope_scaling=None, rope_parameters=None),
        head_dim=head_dim,
        num_heads=query_heads,
        num_kv_heads=key_value_heads,
        q_size=query_width,
        kv_size=key_width,
        q_norm=query_norm,
        k_norm=key_norm,
        rotary_emb=rotary_embedding,
    )
    gate = install_thinker_fused_rope(
        one_layer_model(attention), prefill_graph_enabled=False
    )

    assert gate is not None
    positions = torch.arange(token_count, dtype=torch.int64).repeat(3, 1).to(device)
    gate.evaluate(
        positions,
        SimpleNamespace(
            mm_inputs=None, forward_mode=SimpleNamespace(is_extend=lambda: True)
        ),
    )

    qkv = torch.randn(token_count, query_width + 2 * key_width, dtype=dtype).to(device)
    value_before = qkv[:, query_width + key_width :].clone()
    expected_query, expected_key = rotary_embedding(
        positions,
        *apply_qk_norm(
            q=qkv[:, :query_width].clone(),
            k=qkv[:, query_width : query_width + key_width].clone(),
            q_norm=query_norm,
            k_norm=key_norm,
            head_dim=head_dim,
        ),
    )

    fused_query, fused_key, fused_value = attention.apply_qk_norm_rope(
        qkv, positions, None
    )
    torch.xpu.synchronize()

    assert fused_query.device.type == "xpu"
    torch.testing.assert_close(
        fused_query,
        expected_query,
        atol=FUSED_ROPE_ABSOLUTE_TOLERANCE,
        rtol=FUSED_ROPE_RELATIVE_TOLERANCE,
    )
    torch.testing.assert_close(
        fused_key,
        expected_key,
        atol=FUSED_ROPE_ABSOLUTE_TOLERANCE,
        rtol=FUSED_ROPE_RELATIVE_TOLERANCE,
    )
    assert torch.equal(fused_value, value_before)


def test_a_head_dim_the_kernel_cannot_rotate_stays_unfused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(thinker_fused_rope.current_platform, "is_xpu", lambda: True)
    attention = SimpleNamespace(
        apply_qk_norm_rope=lambda *unused: None,
        config=SimpleNamespace(rope_scaling=None, rope_parameters=None),
        head_dim=UNFUSABLE_HEAD_DIM,
        rotary_emb=SimpleNamespace(
            cos_sin_cache=torch.zeros(8, UNFUSABLE_HEAD_DIM), is_neox_style=True
        ),
    )
    original_apply_qk_norm_rope = attention.apply_qk_norm_rope

    assert (
        install_thinker_fused_rope(
            one_layer_model(attention),
            kernel_provider=lambda: (lambda *unused: None),
            prefill_graph_enabled=False,
        )
        is None
    )
    assert attention.apply_qk_norm_rope is original_apply_qk_norm_rope
