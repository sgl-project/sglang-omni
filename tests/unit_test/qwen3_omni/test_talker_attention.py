# SPDX-License-Identifier: Apache-2.0
"""GQA equivalence tests for the Qwen3-Omni predictor attention paths."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import sglang_omni.models.qwen3_omni.components.talker as talker_module
from sglang_omni.models.qwen3_omni.components.talker import (
    Qwen3OmniMoeTalkerCodePredictor,
    Qwen3OmniTalker,
)
from tests.unit_test.fixtures.qwen_predictor import (
    build_real_step_predictor_graph_talker,
)

# note (EdwardZhang1108): cpu/fp32 covers the math backend; cuda/bf16 locks the
# production-dtype evidence into CI instead of living only in the PR description.
_DEVICE_DTYPE_PARAMS = [
    pytest.param("cpu", torch.float32, id="cpu-fp32"),
    pytest.param(
        "cuda",
        torch.bfloat16,
        id="cuda-bf16",
        marks=[
            pytest.mark.accelerator,
            pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA bf16 variant requires a GPU",
            ),
        ],
    ),
]


def _build_gqa_talker(device: torch.device, dtype: torch.dtype) -> Qwen3OmniTalker:
    talker = build_real_step_predictor_graph_talker(device, num_heads=4, num_kv_heads=2)
    attn = talker.code_predictor.model.layers[0].self_attn
    # note (EdwardZhang1108): kv heads > 1, else wrong GQA group order passes by broadcast
    assert attn.num_heads != attn.num_kv_heads and attn.num_kv_heads > 1
    if dtype is not torch.float32:
        attn.qkv_proj.to(dtype)
        attn.o_proj.to(dtype)
        talker._predictor_k_cache = talker._predictor_k_cache.to(dtype)
        talker._predictor_v_cache = talker._predictor_v_cache.to(dtype)
    return talker


def _project_q_kv(attn: SimpleNamespace, hidden_states: torch.Tensor):
    """Shared projection: hidden states to per-head q/k/v, mirroring the source."""
    batch_size, seq_len, hidden_size = hidden_states.shape
    qkv, _ = attn.qkv_proj(hidden_states.reshape(-1, hidden_size))
    q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)

    def heads(t: torch.Tensor, num: int) -> torch.Tensor:
        return t.reshape(batch_size, seq_len, num, attn.head_dim).transpose(1, 2)

    return (
        heads(q, attn.num_heads),
        heads(k, attn.num_kv_heads),
        heads(v, attn.num_kv_heads),
    )


def _materialized_sdpa(
    attn: SimpleNamespace,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
) -> torch.Tensor:
    """Reference attention that materializes KV heads before SDPA."""
    num_kv_groups = attn.num_heads // attn.num_kv_heads
    k = k.repeat_interleave(num_kv_groups, dim=1)
    v = v.repeat_interleave(num_kv_groups, dim=1)
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=is_causal
    )
    batch_size, _, seq_len, _ = q.shape
    attn_output = attn_output.transpose(1, 2).reshape(
        batch_size * seq_len, attn.num_heads * attn.head_dim
    )
    attn_output, _ = attn.o_proj(attn_output)
    return attn_output.reshape(batch_size, seq_len, -1)


def _materialized_kv_direct_attention(
    *,
    attn: SimpleNamespace,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    q, k, v = _project_q_kv(attn, hidden_states)
    return _materialized_sdpa(attn, q, k, v, is_causal=True)


def _materialized_kv_cached_attention(
    *,
    talker: Qwen3OmniTalker,
    attn: SimpleNamespace,
    hidden_states: torch.Tensor,
    batch_size: int,
    cache_len: int,
) -> torch.Tensor:
    q, _, _ = _project_q_kv(attn, hidden_states)
    cached_k = talker._predictor_k_cache[0, :batch_size, :, : cache_len + 1, :]
    cached_v = talker._predictor_v_cache[0, :batch_size, :, : cache_len + 1, :]
    return _materialized_sdpa(attn, q, cached_k, cached_v, is_causal=False)


@pytest.mark.parametrize("device_name,dtype", _DEVICE_DTYPE_PARAMS)
def test_qwen_predictor_direct_attention_gqa_matches_materialized_kv(
    monkeypatch: pytest.MonkeyPatch,
    device_name: str,
    dtype: torch.dtype,
):
    """Direct-path SDPA with enable_gqa must equal materialized KV expansion."""
    monkeypatch.setattr(
        talker_module,
        "apply_qk_norm",
        lambda q, k, **_: (q, k),
    )

    device = torch.device(device_name)
    talker = _build_gqa_talker(device, dtype)
    attn = talker.code_predictor.model.layers[0].self_attn

    batch_size, seq_len, hidden_size = 2, 3, 8
    torch.manual_seed(7)
    hidden_states = torch.randn(
        batch_size, seq_len, hidden_size, device=device, dtype=dtype
    )
    positions = torch.arange(seq_len, device=device).repeat(batch_size)

    with torch.no_grad():
        actual = Qwen3OmniMoeTalkerCodePredictor._direct_self_attention(
            attn=attn,
            hidden_states=hidden_states,
            positions=positions,
        )
        expected = _materialized_kv_direct_attention(
            attn=attn,
            hidden_states=hidden_states,
        )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("device_name,dtype", _DEVICE_DTYPE_PARAMS)
def test_qwen_predictor_cached_attention_gqa_matches_materialized_kv(
    monkeypatch: pytest.MonkeyPatch,
    device_name: str,
    dtype: torch.dtype,
):
    """Cached-decode SDPA with enable_gqa must equal materialized KV expansion."""
    monkeypatch.setattr(
        talker_module,
        "apply_qk_norm",
        lambda q, k, **_: (q, k),
    )

    device = torch.device(device_name)
    talker = _build_gqa_talker(device, dtype)
    attn = talker.code_predictor.model.layers[0].self_attn
    batch_size, hidden_size = 2, 8

    torch.manual_seed(11)
    with torch.no_grad():
        for cache_len in range(3):
            hidden_states = torch.randn(
                batch_size, 1, hidden_size, device=device, dtype=dtype
            )
            positions = torch.full((batch_size,), cache_len, device=device)
            actual = talker._predictor_cached_self_attention(
                layer_idx=0,
                attn=attn,
                hidden_states=hidden_states,
                positions=positions,
                batch_size=batch_size,
                cache_len=cache_len,
            )
            expected = _materialized_kv_cached_attention(
                talker=talker,
                attn=attn,
                hidden_states=hidden_states,
                batch_size=batch_size,
                cache_len=cache_len,
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
