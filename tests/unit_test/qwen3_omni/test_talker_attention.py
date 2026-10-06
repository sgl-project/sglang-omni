# SPDX-License-Identifier: Apache-2.0
"""GQA equivalence tests for the Qwen3-Omni predictor attention paths."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
from sglang.kernels.fused_op import get_fused_op_backend, set_fused_op_backend
from sglang.kernels.spec import KernelBackend
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.runtime_context import get_context

import sglang_omni.models.qwen3_omni.components.talker as talker_module
from sglang_omni.models.qwen3_omni.components.talker import (
    Qwen3OmniMoeTalkerCodePredictor,
    Qwen3OmniTalker,
)
from sglang_omni.platforms import current_platform
from sglang_omni.vendor.sglang.layers import RMSNorm
from sglang_omni.vendor.sglang.models import apply_qk_norm
from tests.unit_test.fixtures.qwen_predictor import (
    TupleLinear,
    build_real_step_predictor_graph_talker,
)

# note (EdwardZhang1108): cpu/fp32 covers the math backend; cuda/bf16 locks the
# production-dtype evidence into CI instead of living only in the PR description.
DEVICE_DTYPE_PARAMS = [
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


def build_gqa_talker(device: torch.device, dtype: torch.dtype) -> Qwen3OmniTalker:
    talker = build_real_step_predictor_graph_talker(device, num_heads=4, num_kv_heads=2)
    attn = talker.code_predictor.model.layers[0].self_attn
    # note (EdwardZhang1108): kv heads > 1, else wrong GQA group order passes by broadcast
    assert attn.num_heads != attn.num_kv_heads and attn.num_kv_heads > 1
    if dtype is not torch.float32:
        attn.qkv_proj.to(dtype)
        attn.o_proj.to(dtype)
        talker.predictor_k_cache = talker.predictor_k_cache.to(dtype)
        talker.predictor_v_cache = talker.predictor_v_cache.to(dtype)
        talker.predictor_k_rows = [
            layer.view(
                talker.predictor_k_cache.shape[1] * talker.predictor_k_cache.shape[2],
                -1,
            )
            for layer in talker.predictor_k_cache
        ]
        talker.predictor_v_rows = [
            layer.view(
                talker.predictor_v_cache.shape[1] * talker.predictor_v_cache.shape[2],
                -1,
            )
            for layer in talker.predictor_v_cache
        ]
    return talker


def project_q_kv(attn: SimpleNamespace, hidden_states: torch.Tensor):
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


def materialized_sdpa(
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


def materialized_kv_direct_attention(
    *,
    attn: SimpleNamespace,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    q, k, v = project_q_kv(attn, hidden_states)
    return materialized_sdpa(attn, q, k, v, is_causal=True)


def materialized_kv_cached_attention(
    *,
    talker: Qwen3OmniTalker,
    attn: SimpleNamespace,
    hidden_states: torch.Tensor,
    batch_size: int,
    cache_len: int,
) -> torch.Tensor:
    q, _, _ = project_q_kv(attn, hidden_states)
    cached_k = talker.predictor_k_cache[0, :batch_size, : cache_len + 1].transpose(1, 2)
    cached_v = talker.predictor_v_cache[0, :batch_size, : cache_len + 1].transpose(1, 2)
    return materialized_sdpa(attn, q, cached_k, cached_v, is_causal=False)


@pytest.mark.parametrize("device_name,dtype", DEVICE_DTYPE_PARAMS)
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
    talker = build_gqa_talker(device, dtype)
    attn = talker.code_predictor.model.layers[0].self_attn

    batch_size, seq_len, hidden_size = 2, 3, 8
    torch.manual_seed(7)
    hidden_states = torch.randn(
        batch_size, seq_len, hidden_size, device=device, dtype=dtype
    )
    positions = torch.arange(seq_len, device=device).repeat(batch_size)

    with torch.no_grad():
        actual = Qwen3OmniMoeTalkerCodePredictor.direct_self_attention(
            attn=attn,
            hidden_states=hidden_states,
            positions=positions,
        )
        expected = materialized_kv_direct_attention(
            attn=attn,
            hidden_states=hidden_states,
        )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("device_name,dtype", DEVICE_DTYPE_PARAMS)
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
    talker = build_gqa_talker(device, dtype)
    attn = talker.code_predictor.model.layers[0].self_attn
    batch_size, hidden_size = 2, 8

    torch.manual_seed(11)
    with torch.no_grad():
        for cache_len in range(3):
            hidden_states = torch.randn(
                batch_size, 1, hidden_size, device=device, dtype=dtype
            )
            positions = torch.full((batch_size,), cache_len, device=device)
            actual = talker.predictor_cached_self_attention(
                layer_idx=0,
                attn=attn,
                hidden_states=hidden_states,
                positions=positions,
                cache_slots=talker.predictor_cache_slots[cache_len, :batch_size],
                batch_size=batch_size,
                cache_len=cache_len,
            )
            expected = materialized_kv_cached_attention(
                talker=talker,
                attn=attn,
                hidden_states=hidden_states,
                batch_size=batch_size,
                cache_len=cache_len,
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


ROPE_HEAD_DIM = 128
ROPE_NUM_HEADS = 16
ROPE_NUM_KV_HEADS = 8
ROPE_HIDDEN = 1024
ROPE_PREDICTOR_LEN = 17
ROPE_MAX_BS = 16
ROPE_DTYPE = torch.bfloat16


def rope_store_talker(device: torch.device, *, stores: bool) -> Qwen3OmniTalker:
    talker = object.__new__(Qwen3OmniTalker)
    positions = torch.arange(ROPE_PREDICTOR_LEN, device=device, dtype=torch.long)
    talker.predictor_positions = positions
    talker.predictor_k_cache = torch.zeros(
        1,
        ROPE_MAX_BS,
        ROPE_PREDICTOR_LEN,
        ROPE_NUM_KV_HEADS,
        ROPE_HEAD_DIM,
        device=device,
        dtype=ROPE_DTYPE,
    )
    talker.predictor_v_cache = torch.zeros_like(talker.predictor_k_cache)
    talker.predictor_k_rows = [
        layer.view(ROPE_MAX_BS * ROPE_PREDICTOR_LEN, -1)
        for layer in talker.predictor_k_cache
    ]
    talker.predictor_v_rows = [
        layer.view(ROPE_MAX_BS * ROPE_PREDICTOR_LEN, -1)
        for layer in talker.predictor_v_cache
    ]
    talker.predictor_cache_slots = (
        torch.arange(ROPE_MAX_BS, device=device, dtype=torch.long)[None, :]
        * ROPE_PREDICTOR_LEN
        + positions[:, None]
    ).contiguous()
    talker.predictor_rope_stores_kv = stores
    return talker


def rope_attention(device: torch.device) -> SimpleNamespace:
    return SimpleNamespace(
        q_size=ROPE_NUM_HEADS * ROPE_HEAD_DIM,
        kv_size=ROPE_NUM_KV_HEADS * ROPE_HEAD_DIM,
        num_heads=ROPE_NUM_HEADS,
        num_kv_heads=ROPE_NUM_KV_HEADS,
        head_dim=ROPE_HEAD_DIM,
        q_norm=RMSNorm(ROPE_HEAD_DIM, eps=1e-6).to(device, ROPE_DTYPE),
        k_norm=RMSNorm(ROPE_HEAD_DIM, eps=1e-6).to(device, ROPE_DTYPE),
        alt_stream=None,
        qkv_proj=TupleLinear(
            ROPE_HIDDEN, (ROPE_NUM_HEADS + 2 * ROPE_NUM_KV_HEADS) * ROPE_HEAD_DIM
        ).to(device, ROPE_DTYPE),
        o_proj=TupleLinear(ROPE_NUM_HEADS * ROPE_HEAD_DIM, ROPE_HIDDEN).to(
            device, ROPE_DTYPE
        ),
        rotary_emb=RotaryEmbedding(
            ROPE_HEAD_DIM, ROPE_HEAD_DIM, 64, 10000, True, ROPE_DTYPE
        ).to(device),
        compatible_with_fused_kv_buffer=True,
    )


def rope_copy_reference(
    attn: SimpleNamespace,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_len: int,
) -> torch.Tensor:
    """Plain RoPE followed by the former [batch, head, slot, dim] cache writes."""
    batch_size = hidden.shape[0]
    qkv, _ = attn.qkv_proj(hidden.reshape(batch_size, -1))
    q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    q, k = apply_qk_norm(
        q, k, attn.q_norm, attn.k_norm, attn.head_dim, alt_stream=attn.alt_stream
    )
    q, k = attn.rotary_emb(positions, q, k, fused_set_kv_buffer_arg=None)
    k_cache[:batch_size, :, cache_len : cache_len + 1].copy_(
        k.reshape(batch_size, 1, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
    )
    v_cache[:batch_size, :, cache_len : cache_len + 1].copy_(
        v.reshape(batch_size, 1, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
    )
    output = torch.nn.functional.scaled_dot_product_attention(
        q.reshape(batch_size, 1, attn.num_heads, attn.head_dim).transpose(1, 2),
        k_cache[:batch_size, :, : cache_len + 1],
        v_cache[:batch_size, :, : cache_len + 1],
        is_causal=False,
        enable_gqa=True,
    )
    output, _ = attn.o_proj(output.transpose(1, 2).reshape(batch_size, -1))
    return output


@pytest.fixture(params=["cuda", "torch"])
def predictor_rope_dispatch(request: pytest.FixtureRequest) -> Iterator[str]:
    mode = request.param
    with get_context().override_server_args(
        cuda_graph_config=CudaGraphConfig(
            prefill=PhaseConfig(backend=Backend.DISABLED)
        ),
    ):
        previous_backend = get_fused_op_backend()
        try:
            set_fused_op_backend(KernelBackend.TORCH if mode == "torch" else None)
            yield mode
        finally:
            set_fused_op_backend(previous_backend)


@pytest.mark.accelerator
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the rope kernel stores K and V on CUDA only"
)
@pytest.mark.parametrize("batch_size", [1, 16])
def test_rope_store_writes_the_cache_the_copy_path_writes(
    batch_size: int,
    predictor_rope_dispatch: str,
) -> None:
    device = torch.device("cuda")
    torch.manual_seed(11)
    attn = rope_attention(device)
    stores = Qwen3OmniTalker.resolve_predictor_rope_store(attn, device=device)
    stored = rope_store_talker(device, stores=stores)
    copied = rope_store_talker(device, stores=False)
    reference_k = torch.zeros(
        ROPE_MAX_BS,
        ROPE_NUM_KV_HEADS,
        ROPE_PREDICTOR_LEN,
        ROPE_HEAD_DIM,
        device=device,
        dtype=ROPE_DTYPE,
    )
    reference_v = torch.zeros_like(reference_k)
    hidden_steps = torch.randn(
        ROPE_PREDICTOR_LEN, batch_size, 1, ROPE_HIDDEN, device=device, dtype=ROPE_DTYPE
    )

    def run_attention(talker: Qwen3OmniTalker) -> torch.Tensor:
        return torch.stack(
            [
                talker.predictor_cached_self_attention(
                    layer_idx=0,
                    attn=attn,
                    hidden_states=hidden_steps[slot],
                    positions=talker.predictor_positions[slot : slot + 1].repeat(
                        batch_size
                    ),
                    cache_slots=talker.predictor_cache_slots[slot, :batch_size],
                    batch_size=batch_size,
                    cache_len=slot,
                )
                for slot in range(ROPE_PREDICTOR_LEN)
            ]
        )

    def run_reference() -> torch.Tensor:
        return torch.stack(
            [
                rope_copy_reference(
                    attn,
                    hidden_steps[slot],
                    stored.predictor_positions[slot : slot + 1].repeat(batch_size),
                    reference_k,
                    reference_v,
                    slot,
                ).reshape(batch_size, 1, ROPE_HIDDEN)
                for slot in range(ROPE_PREDICTOR_LEN)
            ]
        )

    def assert_matches_reference(output: torch.Tensor) -> None:
        expected = run_reference()
        torch.cuda.synchronize()
        assert torch.equal(output, expected)
        assert torch.equal(stored.predictor_k_cache[0].transpose(1, 2), reference_k)
        assert torch.equal(stored.predictor_v_cache[0].transpose(1, 2), reference_v)

    with torch.no_grad():
        output = run_attention(stored)
        assert torch.equal(output, run_attention(copied))
        assert torch.equal(stored.predictor_k_cache, copied.predictor_k_cache)
        assert torch.equal(stored.predictor_v_cache, copied.predictor_v_cache)
        assert_matches_reference(output)
        assert stores == (predictor_rope_dispatch == "cuda")
        if not stores:
            return
        else:
            pass
        stream = torch.cuda.Stream(device=device)
        current_stream = torch.cuda.current_stream(device)
        stream.wait_stream(current_stream)
        with torch.cuda.stream(stream):
            for _ in range(2):
                run_attention(stored)
        current_stream.wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            replay_output = run_attention(stored)
        for _ in range(2):
            hidden_steps.normal_()
            graph.replay()
            assert_matches_reference(replay_output)
