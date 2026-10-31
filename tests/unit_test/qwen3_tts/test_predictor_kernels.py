# SPDX-License-Identifier: Apache-2.0
"""Correctness gates for optional Qwen3-TTS predictor kernels."""

from __future__ import annotations

from importlib.util import find_spec
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.runtime_context import get_context, get_parallel
from torch.nn import functional as F

from sglang_omni.models.qwen3_tts import predictor_kernels
from sglang_omni.models.qwen3_tts.predictor_kernels import (
    gather_codec_embedding_and_add,
    short_cache_gqa_npu,
)
from sglang_omni.models.qwen3_tts.sglang_model import (
    Qwen3TTSTalkerTextModel,
    predictor_gqa_attention,
)
from sglang_omni.platforms import current_platform

if find_spec("cutlass") is not None:
    from sglang_omni.models.qwen3_tts.cutedsl.api import (
        PackedPostAttention,
        SlotPreAttention,
        compile_post_attention,
        compile_pre_attention,
    )
    from sglang_omni.models.qwen3_tts.cutedsl_backend import (
        MIN_DECODE_SM_COUNT,
        build_cutedsl_layers,
    )
else:
    PackedPostAttention = SlotPreAttention = None


def rounded_rmsnorm(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    normalized = values.float() * torch.rsqrt(
        values.float().square().mean(-1, keepdim=True) + 1e-6
    )
    return normalized.to(values.dtype) * weight


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available() or SlotPreAttention is None,
    reason="requires CUDA and CUTLASS DSL",
)
@pytest.mark.parametrize("hidden_size,batch_size", [(1024, 1), (1024, 16), (2048, 17)])
@torch.inference_mode()
def test_cutedsl_talker_decode_cache(
    hidden_size: int, batch_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fused talker decode must read and update engine-owned cache slots like the default loop."""
    if (
        torch.cuda.get_device_capability() not in ((9, 0), (10, 0), (10, 3))
        or torch.cuda.get_device_properties("cuda").multi_processor_count
        < MIN_DECODE_SM_COUNT
    ):
        pytest.skip("requires Hopper or datacenter Blackwell with at least 96 SMs")
    else:
        pass
    torch.manual_seed(42)
    configuration = SimpleNamespace(
        hidden_size=hidden_size,
        intermediate_size=3 * hidden_size,
        vocab_size=32,
        text_vocab_size=32,
        text_hidden_size=hidden_size,
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        rms_norm_eps=1e-6,
        attention_bias=False,
        rope_theta=1_000_000,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [24, 20, 20],
            "interleaved": True,
        },
        max_position_embeddings=64,
    )
    with (
        get_context().override_server_args(
            max_running_requests=32,
            cuda_graph_config=CudaGraphConfig(
                prefill=PhaseConfig(backend=Backend.DISABLED)
            ),
        ),
        get_parallel().override(
            tp_rank=0,
            tp_size=1,
            tp_group=None,
            attn_tp_rank=0,
            attn_tp_size=1,
            attn_tp_group=None,
        ),
    ):
        model = Qwen3TTSTalkerTextModel(configuration).to("cuda", torch.bfloat16)
        for parameter in model.parameters():
            if parameter.ndim == 1:
                parameter.fill_(1)
            else:
                parameter.normal_(std=parameter.shape[-1] ** -0.5)
        with monkeypatch.context() as patch:
            patch.setattr(
                torch.cuda,
                "get_device_properties",
                lambda device: SimpleNamespace(
                    major=9, minor=0, multi_processor_count=78
                ),
            )
            assert build_cutedsl_layers(model.layers) is None
        fused_layers = build_cutedsl_layers(model.layers)
        assert fused_layers is not None
        model.cutedsl_layers = None
        cache = MHATokenToKVPool(
            size=batch_size * 8,
            page_size=1,
            dtype=torch.bfloat16,
            head_num=8,
            head_dim=128,
            layer_num=2,
            device="cuda",
            enable_memory_saver=False,
        )
        request_pool = ReqToTokenPool(batch_size, 8, "cuda", enable_memory_saver=False)
        request_pool.req_to_token[1:].copy_(
            torch.randperm(batch_size * 8, device="cuda").reshape(batch_size, 8) + 1
        )
        attention_backend = TorchNativeAttnBackend(
            SimpleNamespace(
                device="cuda",
                req_to_token_pool=request_pool,
                token_to_kv_pool=cache,
            )
        )
        sequence_lengths = torch.arange(batch_size, device="cuda") % 3 + 3
        request_rows = torch.arange(1, batch_size + 1, device="cuda")
        batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=batch_size,
            input_ids=torch.arange(batch_size, device="cuda"),
            req_pool_indices=request_rows,
            seq_lens=sequence_lengths,
            seq_lens_cpu=sequence_lengths.cpu(),
            out_cache_loc=request_pool.req_to_token[
                request_rows, sequence_lengths - 1
            ].long(),
            seq_lens_sum=int(sequence_lengths.sum()),
        )
        batch._compute_mrope_positions_decode(  # noqa: leading-underscore - exercise SGLang's decode position builder.
            SimpleNamespace(device="cuda"),
            SimpleNamespace(multimodal_inputs=[None] * batch_size),
        )
        positions = batch.mrope_positions
        assert torch.equal(positions[0], positions[1])
        assert torch.equal(positions[0], positions[2])
        initial_cache = []
        for layer_index in range(2):
            for buffer in (
                cache.get_key_buffer(layer_index),
                cache.get_value_buffer(layer_index),
            ):
                buffer.normal_()
                initial_cache.append(buffer.clone())
        with forward_context(ForwardContext(attention_backend)):
            expected = model(batch.input_ids, positions, batch)
            expected_cache = [
                buffer.clone()
                for layer_index in range(2)
                for buffer in (
                    cache.get_key_buffer(layer_index),
                    cache.get_value_buffer(layer_index),
                )
            ]
            for layer_index in range(2):
                cache.get_key_buffer(layer_index).copy_(initial_cache[2 * layer_index])
                cache.get_value_buffer(layer_index).copy_(
                    initial_cache[2 * layer_index + 1]
                )
            model.cutedsl_layers = fused_layers
            actual = model(batch.input_ids, positions, batch)
        torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.02)
        untouched = torch.ones(cache.size + 1, device="cuda", dtype=torch.bool)
        untouched[batch.out_cache_loc] = False
        for layer_index in range(2):
            for offset, buffer in enumerate(
                (cache.get_key_buffer(layer_index), cache.get_value_buffer(layer_index))
            ):
                torch.testing.assert_close(
                    buffer,
                    expected_cache[2 * layer_index + offset],
                    atol=0.04,
                    rtol=0.02,
                )
                torch.testing.assert_close(
                    buffer[untouched],
                    initial_cache[2 * layer_index + offset][untouched],
                    atol=0,
                    rtol=0,
                )


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available() or SlotPreAttention is None,
    reason="requires CUDA and CUTLASS DSL",
)
@pytest.mark.parametrize(
    "hidden_size,batch_size", [(1024, 1), (1024, 7), (1024, 32), (2048, 16), (2048, 32)]
)
@torch.inference_mode()
def test_cutedsl_cache_slots_and_workspace_replay(
    hidden_size: int, batch_size: int
) -> None:
    """Changed positions and slots must survive graph replay without overwriting other rows."""
    check_cutedsl_cache_slots_and_workspace_replay(hidden_size, batch_size, 128)


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available() or SlotPreAttention is None,
    reason="requires CUDA and CUTLASS DSL",
)
@pytest.mark.parametrize("hidden_size", [1024, 2048])
@torch.inference_mode()
def test_cutedsl_dynamic_sizes_reuse_compilation(hidden_size: int) -> None:
    """Batch, RoPE and cache capacity changes reuse one binary per batch tile."""
    compile_pre_attention.cache_clear()
    compile_post_attention.cache_clear()
    for batch_size, rope_positions in (
        (1, 33),
        (5, 65),
        (16, 129),
        (17, 65),
        (32, 257),
    ):
        check_cutedsl_cache_slots_and_workspace_replay(
            hidden_size, batch_size, rope_positions
        )
    assert compile_pre_attention.cache_info().misses == 2
    assert compile_post_attention.cache_info().misses == 2


def check_cutedsl_cache_slots_and_workspace_replay(
    hidden_size: int, batch_size: int, rope_positions: int
) -> None:
    """Check cache writes and post-attention outputs against rounded BF16 operations."""
    if (
        torch.cuda.get_device_capability() not in ((9, 0), (10, 0), (10, 3))
        or torch.cuda.get_device_properties("cuda").multi_processor_count
        < MIN_DECODE_SM_COUNT
    ):
        pytest.skip("requires Hopper or datacenter Blackwell with at least 96 SMs")
    else:
        pass
    torch.manual_seed(42)
    norm_weight = torch.empty(
        hidden_size, device="cuda", dtype=torch.bfloat16
    ).uniform_(0.5, 1.5)
    head_weight = torch.empty(128, device="cuda", dtype=torch.bfloat16).uniform_(
        0.5, 1.5
    )
    qkv_weight = (
        torch.randn(4096, hidden_size, device="cuda", dtype=torch.bfloat16)
        / hidden_size**0.5
    )
    pre_attention = SlotPreAttention(
        norm_weight, qkv_weight, head_weight, head_weight, 1e-6
    )
    hidden_states = torch.randn(
        batch_size, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    positions = torch.arange(batch_size, device="cuda", dtype=torch.long)
    slots = positions.to(torch.int32 if batch_size == 7 else torch.int64) * 2 + 1
    key_cache = torch.randn(
        4 * batch_size + 8, 8, 128, device="cuda", dtype=torch.bfloat16
    )
    value_cache = torch.randn_like(key_cache)
    angles = (
        torch.arange(rope_positions, device="cuda")[:, None]
        * (1_000_000.0 ** (-torch.arange(0, 128, 2, device="cuda").float() / 128))[
            None, :
        ]
    )
    rope = torch.cat((angles.cos(), angles.sin()), dim=-1)
    if batch_size == 7:
        rope = rope.bfloat16()
    else:
        pass
    pre_attention(hidden_states, positions, rope, slots, key_cache, value_cache)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_queries = pre_attention(
            hidden_states, positions, rope, slots, key_cache, value_cache
        )
    for step in range(3):
        hidden_states.normal_()
        positions.copy_(
            (torch.arange(batch_size, device="cuda") * 3 + step) % rope_positions
        )
        slots.copy_(torch.arange(batch_size, device="cuda") * 2 + step)
        previous_keys, previous_values = key_cache.clone(), value_cache.clone()
        graph.replay()
        normalized = rounded_rmsnorm(hidden_states, norm_weight)
        queries, keys, values = F.linear(normalized, qkv_weight).split(
            (2048, 1024, 1024), dim=-1
        )
        queries = rounded_rmsnorm(queries.reshape(batch_size, 16, 128), head_weight)
        keys = rounded_rmsnorm(keys.reshape(batch_size, 8, 128), head_weight)
        cosine, sine = rope[positions].bfloat16().chunk(2, dim=-1)
        cosine = cosine.repeat(1, 2)[:, None, :]
        sine = sine.repeat(1, 2)[:, None, :]
        queries = (
            queries * cosine
            + torch.cat((-queries[..., 64:], queries[..., :64]), dim=-1) * sine
        )
        keys = (
            keys * cosine + torch.cat((-keys[..., 64:], keys[..., :64]), dim=-1) * sine
        )
        torch.testing.assert_close(
            actual_queries, queries.flatten(1), atol=0.02, rtol=0.02
        )
        torch.testing.assert_close(key_cache[slots], keys, atol=0.02, rtol=0.02)
        torch.testing.assert_close(
            value_cache[slots], values.reshape(batch_size, 8, 128), atol=0.02, rtol=0.02
        )
        untouched = torch.ones(key_cache.shape[0], device="cuda", dtype=torch.bool)
        untouched[slots] = False
        torch.testing.assert_close(
            key_cache[untouched], previous_keys[untouched], atol=0, rtol=0
        )
        torch.testing.assert_close(
            value_cache[untouched], previous_values[untouched], atol=0, rtol=0
        )

    output_weight = (
        torch.randn(hidden_size, 2048, device="cuda", dtype=torch.bfloat16) / 2048**0.5
    )
    gate_weight = (
        torch.randn(3 * hidden_size, hidden_size, device="cuda", dtype=torch.bfloat16)
        / hidden_size**0.5
    )
    up_weight = torch.randn_like(gate_weight) / hidden_size**0.5
    down_weight = (
        torch.randn(hidden_size, 3 * hidden_size, device="cuda", dtype=torch.bfloat16)
        / (3 * hidden_size) ** 0.5
    )
    post_attention = PackedPostAttention(
        output_weight, gate_weight, up_weight, down_weight, norm_weight, 1e-6
    )
    attention = torch.randn(batch_size, 2048, device="cuda", dtype=torch.bfloat16)
    post_attention(attention, hidden_states)
    torch.cuda.synchronize()
    post_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(post_graph):
        actual_output = post_attention(attention, hidden_states)
    for _ in range(3):
        hidden_states.normal_()
        attention.normal_()
        post_graph.replay()
        residual = F.linear(attention, output_weight) + hidden_states
        normalized = rounded_rmsnorm(residual, norm_weight)
        activated = F.silu(F.linear(normalized, gate_weight)) * F.linear(
            normalized, up_weight
        )
        expected = F.linear(activated, down_weight) + residual
        torch.testing.assert_close(actual_output, expected, atol=0.02, rtol=0.02)


def test_predictor_triton_kernel_is_disabled_on_npu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(predictor_kernels, "triton", object())
    monkeypatch.setattr(predictor_kernels.current_platform, "is_npu", lambda: True)

    assert not predictor_kernels.has_triton_runtime()


def test_predictor_gqa_attention_cpu_matches_sdpa() -> None:
    q = torch.randn(2, 4, 1, 8)
    key = torch.randn(2, 2, 5, 8)
    value = torch.randn(2, 2, 5, 8)

    actual = predictor_gqa_attention(
        q, key, value, num_heads=4, num_key_value_heads=2, is_causal=False
    )
    expected = F.scaled_dot_product_attention(
        q, key, value, is_causal=False, enable_gqa=True
    )

    torch.testing.assert_close(actual, expected)


def test_predictor_gqa_attention_pair_attends_causally() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 4, 2, 8)
    key = torch.randn(2, 2, 2, 8)
    value = torch.randn(2, 2, 2, 8)

    actual = predictor_gqa_attention(
        q, key, value, num_heads=4, num_key_value_heads=2, is_causal=True
    )
    first = F.scaled_dot_product_attention(
        q[:, :, :1], key[:, :, :1], value[:, :, :1], enable_gqa=True
    )
    second = F.scaled_dot_product_attention(q[:, :, 1:], key, value, enable_gqa=True)

    torch.testing.assert_close(actual, torch.cat((first, second), dim=2))


@pytest.mark.skipif(not current_platform.is_npu(), reason="requires Ascend NPU")
@pytest.mark.parametrize(
    "query_length,cache_length,is_causal", [(1, 5, False), (2, 2, True)]
)
def test_predictor_npu_fused_attention_matches_sdpa(
    query_length: int, cache_length: int, is_causal: bool
) -> None:
    device = torch.device("npu:0")
    q = torch.randn(2, 4, query_length, 128, device=device, dtype=torch.bfloat16)
    key = torch.randn(2, 2, cache_length, 128, device=device, dtype=torch.bfloat16)
    value = torch.randn(2, 2, cache_length, 128, device=device, dtype=torch.bfloat16)

    actual = predictor_gqa_attention(
        q, key, value, num_heads=4, num_key_value_heads=2, is_causal=is_causal
    )
    expected = F.scaled_dot_product_attention(
        q, key, value, is_causal=is_causal, enable_gqa=True
    )
    torch.npu.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_npu(), reason="requires Ascend NPU")
@pytest.mark.parametrize("batch,length", [(1, 1), (8, 5), (8, 6), (16, 16), (32, 9)])
def test_predictor_short_cache_gqa_matches_float32_reference(
    batch: int, length: int
) -> None:
    generator = torch.Generator().manual_seed(1234)
    query = torch.randn(batch, 16, 1, 128, generator=generator).bfloat16()
    key = torch.randn(batch, 17, 8, 128, generator=generator).bfloat16()
    value = torch.randn(batch, 17, 8, 128, generator=generator).bfloat16()
    key_npu = key.npu()[:, :length].transpose(1, 2)
    value_npu = value.npu()[:, :length].transpose(1, 2)
    for scale in (1.0, 10.0):
        scaled_query = query * scale
        expected = F.scaled_dot_product_attention(
            scaled_query.float(),
            key[:, :length].transpose(1, 2).float(),
            value[:, :length].transpose(1, 2).float(),
            enable_gqa=True,
        ).bfloat16()
        actual = predictor_gqa_attention(
            scaled_query.npu(),
            key_npu,
            value_npu,
            num_heads=16,
            num_key_value_heads=8,
            is_causal=False,
        )
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-3, rtol=8e-3)


@pytest.mark.skipif(not current_platform.is_npu(), reason="requires Ascend NPU")
def test_predictor_short_cache_graph_replay_and_fallback() -> None:
    generator = torch.Generator().manual_seed(5678)
    query_cpu = torch.randn(4, 16, 1, 128, generator=generator).bfloat16()
    key_cpu = torch.randn(4, 17, 8, 128, generator=generator).bfloat16()
    value_cpu = torch.randn(4, 17, 8, 128, generator=generator).bfloat16()
    query, key, value = query_cpu.npu(), key_cpu.npu(), value_cpu.npu()
    for _ in range(2):
        short_cache_gqa_npu(
            query, key[:, :7].transpose(1, 2), value[:, :7].transpose(1, 2)
        )
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = short_cache_gqa_npu(
            query, key[:, :7].transpose(1, 2), value[:, :7].transpose(1, 2)
        )
    for _ in range(4):
        key_cpu = torch.randn(4, 17, 8, 128, generator=generator).bfloat16()
        value_cpu = torch.randn(4, 17, 8, 128, generator=generator).bfloat16()
        key.copy_(key_cpu)
        value.copy_(value_cpu)
        graph.replay()
        expected = F.scaled_dot_product_attention(
            query_cpu.float(),
            key_cpu[:, :7].transpose(1, 2).float(),
            value_cpu[:, :7].transpose(1, 2).float(),
            enable_gqa=True,
        ).bfloat16()
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-3, rtol=8e-3)
    assert (
        short_cache_gqa_npu(query, key.transpose(1, 2), value.transpose(1, 2)) is None
    )
    assert (
        short_cache_gqa_npu(
            query.float(),
            key[:, :7].transpose(1, 2).float(),
            value[:, :7].transpose(1, 2).float(),
        )
        is None
    )
    assert (
        short_cache_gqa_npu(
            query[:, :8], key[:, :7].transpose(1, 2), value[:, :7].transpose(1, 2)
        )
        is None
    )


@pytest.mark.skipif(not current_platform.is_npu(), reason="requires Ascend NPU")
@pytest.mark.parametrize("invalid_tensor", ["query", "key"])
def test_predictor_short_cache_rejects_one_non_4d_tensor(invalid_tensor: str) -> None:
    """A single malformed query or cache must fall back before indexing shapes."""
    query = torch.zeros(1, 16, 1, 128, device="npu", dtype=torch.bfloat16)
    key = torch.zeros(1, 8, 4, 128, device="npu", dtype=torch.bfloat16)
    if invalid_tensor == "query":
        query = query.flatten()[0]
    else:
        key = key.flatten(0, 2)
    value = torch.zeros_like(key)
    assert short_cache_gqa_npu(query, key, value) is None


def test_gather_codec_embedding_and_add_cpu_falls_back_without_writes():
    token_ids = torch.tensor([1, 3], dtype=torch.long)
    embedding_weight = torch.randn(8, 4, dtype=torch.bfloat16)
    gathered = torch.full((2, 4), 2.0, dtype=torch.bfloat16)
    accumulated = torch.full((2, 4), -3.0, dtype=torch.bfloat16)
    expected_gathered = gathered.clone()
    expected_accumulated = accumulated.clone()

    assert not gather_codec_embedding_and_add(
        token_ids,
        embedding_weight,
        gathered,
        accumulated,
    )
    assert torch.equal(gathered, expected_gathered)
    assert torch.equal(accumulated, expected_accumulated)


@pytest.mark.accelerator
@pytest.mark.parametrize("invalid_input", ["dtype", "layout", "overlap"])
def test_gather_codec_embedding_and_add_rejects_unsafe_input_without_writes(
    invalid_input: str,
):
    if not torch.cuda.is_available():
        pytest.skip("Triton predictor kernel needs CUDA")
    device = torch.device("cuda")
    token_ids = torch.tensor([1, 3], dtype=torch.long, device=device)
    embedding_weight = torch.randn(8, 8, dtype=torch.bfloat16, device=device)
    accumulated = torch.full((2, 8), -3.0, dtype=torch.bfloat16, device=device)
    gathered = torch.full((2, 8), 2.0, dtype=torch.bfloat16, device=device)

    if invalid_input == "dtype":
        embedding_weight = embedding_weight.float()
    elif invalid_input == "layout":
        gathered = torch.full((8, 2), 2.0, dtype=torch.bfloat16, device=device).t()
    else:
        shared = torch.full((3, 8), 2.0, dtype=torch.bfloat16, device=device)
        gathered = shared[:2]
        accumulated = shared[1:]

    expected_gathered = gathered.clone()
    expected_accumulated = accumulated.clone()
    assert not gather_codec_embedding_and_add(
        token_ids,
        embedding_weight,
        gathered,
        accumulated,
    )
    assert torch.equal(gathered, expected_gathered)
    assert torch.equal(accumulated, expected_accumulated)


@pytest.mark.accelerator
@pytest.mark.parametrize(
    ("batch_size", "hidden_size", "projected_size"),
    [(1, 8, 0), (4, 8, 0), (8, 2048, 0), (1, 8, 4), (8, 2048, 1024)],
)
def test_gather_codec_embedding_and_add_matches_bf16_reference(
    batch_size: int,
    hidden_size: int,
    projected_size: int,
):
    if not torch.cuda.is_available():
        pytest.skip("Triton predictor kernel needs CUDA")
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(
        batch_size * 10000 + hidden_size
    )
    vocab_size = 53
    token_ids = torch.randint(
        0,
        vocab_size,
        (batch_size,),
        generator=generator,
        dtype=torch.long,
    ).to(device)
    if batch_size > 1:
        token_ids[1] = token_ids[0]
    embedding_weight = torch.randn(
        vocab_size,
        hidden_size,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    accumulated = torch.randn(
        batch_size,
        hidden_size,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    expected_gathered = F.embedding(token_ids, embedding_weight)
    expected_accumulated = accumulated.clone()
    expected_accumulated.add_(expected_gathered)
    gathered = torch.empty_like(expected_gathered)
    if projected_size:
        projected_weight = torch.randn(
            vocab_size,
            projected_size,
            generator=generator,
            dtype=torch.float32,
        ).to(device=device, dtype=torch.bfloat16)
        projected = torch.empty(
            batch_size, projected_size, device=device, dtype=torch.bfloat16
        )
    else:
        projected_weight = None
        projected = None

    assert gather_codec_embedding_and_add(
        token_ids,
        embedding_weight,
        gathered,
        accumulated,
        projected_weight=projected_weight,
        projected=projected,
    )
    torch.cuda.synchronize()

    assert torch.equal(gathered, expected_gathered)
    assert torch.equal(accumulated, expected_accumulated)
    if projected_size:
        assert torch.equal(projected, F.embedding(token_ids, projected_weight))
    else:
        pass


@pytest.mark.accelerator
def test_gather_codec_embedding_and_add_rejects_an_overlapping_projection():
    if not torch.cuda.is_available():
        pytest.skip("Triton predictor kernel needs CUDA")
    device = torch.device("cuda")
    token_ids = torch.tensor([1, 3], dtype=torch.long, device=device)
    embedding_weight = torch.randn(8, 8, dtype=torch.bfloat16, device=device)
    accumulated = torch.full((2, 8), -3.0, dtype=torch.bfloat16, device=device)
    gathered = torch.full((2, 8), 2.0, dtype=torch.bfloat16, device=device)
    projected_weight = torch.randn(8, 4, dtype=torch.bfloat16, device=device)
    expected_gathered = gathered.clone()

    assert not gather_codec_embedding_and_add(
        token_ids,
        embedding_weight,
        gathered,
        accumulated,
        projected_weight=projected_weight,
        projected=gathered.view(4, 4)[:2],
    )
    assert torch.equal(gathered, expected_gathered)
