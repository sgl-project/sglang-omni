# SPDX-License-Identifier: Apache-2.0
"""Correctness gates for optional Qwen3-TTS predictor kernels."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from sglang_omni.models.qwen3_tts.predictor_kernels import (
    gather_codec_embedding_and_add,
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
