# SPDX-License-Identifier: Apache-2.0
"""Correctness gates for optional Qwen3-TTS predictor kernels."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from sglang_omni.models.qwen3_tts import predictor_kernels
from sglang_omni.models.qwen3_tts.predictor_kernels import (
    gather_codec_embedding_and_add,
    short_cache_gqa_npu,
)
from sglang_omni.models.qwen3_tts.sglang_model import predictor_gqa_attention
from sglang_omni.platforms import current_platform


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
@pytest.mark.parametrize("batch,length", [(1, 1), (8, 5), (16, 16), (32, 9)])
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
            query.float(), key.transpose(1, 2).float(), value.transpose(1, 2).float()
        )
        is None
    )
    assert (
        short_cache_gqa_npu(query[:, :8], key.transpose(1, 2), value.transpose(1, 2))
        is None
    )


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
    ("batch_size", "hidden_size"),
    [(1, 8), (4, 8), (8, 2048)],
)
def test_gather_codec_embedding_and_add_matches_bf16_reference(
    batch_size: int,
    hidden_size: int,
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

    assert gather_codec_embedding_and_add(
        token_ids,
        embedding_weight,
        gathered,
        accumulated,
    )
    torch.cuda.synchronize()

    assert torch.equal(gathered, expected_gathered)
    assert torch.equal(accumulated, expected_accumulated)
