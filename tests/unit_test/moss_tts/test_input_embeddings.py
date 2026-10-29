# SPDX-License-Identifier: Apache-2.0
"""Numerical and ownership contracts for multi-channel input embeddings."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch

from sglang_omni.models.moss_tts.channel_embedding import ChannelEmbeddingTables
from sglang_omni.models.moss_tts.sglang_model import MossTTSDelaySGLangModel


@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32, torch.float32),
        (torch.float16, torch.float32, torch.float32),
        (torch.bfloat16, torch.float16, torch.float32),
    ],
)
@pytest.mark.parametrize("rows", [0, 1, 7])
@pytest.mark.parametrize("flatten", [False, True])
@pytest.mark.parametrize("freeze", [False, True])
def test_input_embeddings_preserve_ordered_sum_and_independent_outputs(
    dtypes: tuple[torch.dtype, torch.dtype, torch.dtype],
    rows: int,
    flatten: bool,
    freeze: bool,
) -> None:
    generator = torch.Generator().manual_seed(42)
    layers = [
        torch.nn.Embedding.from_pretrained(
            torch.randn(16, 8, generator=generator).to(dtype), freeze=freeze
        )
        for dtype in dtypes
    ]
    model = SimpleNamespace(
        config=SimpleNamespace(channels=3),
        hidden_size=8,
        embedding_list=layers,
        first_embedding_weight=lambda: layers[0].weight,
    )
    input_rows = torch.randint(0, 16, (rows, 6), generator=generator)[:, ::2]
    input_ids = input_rows.reshape(-1) if flatten else input_rows
    saved_ids = input_ids.clone()
    saved_weights = [layer.weight.detach().clone() for layer in layers]
    contributions = [layer(input_rows[:, index]) for index, layer in enumerate(layers)]
    expected = torch.zeros(rows, 8, dtype=dtypes[0])
    for contribution in contributions:
        expected = expected + contribution

    actual = MossTTSDelaySGLangModel.prepare_multi_modal_inputs(model, input_ids)

    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
    assert torch.equal(input_ids, saved_ids)
    saved_output = actual.detach().clone()
    next_output = MossTTSDelaySGLangModel.prepare_multi_modal_inputs(model, input_ids)
    with torch.no_grad():
        next_output.fill_(0)
    assert torch.equal(actual, saved_output)
    for layer, saved_weight in zip(layers, saved_weights):
        assert torch.equal(layer.weight, saved_weight)

    if freeze:
        assert not actual.requires_grad
        assert not next_output.requires_grad
    else:
        weights = [layer.weight for layer in layers]
        actual_gradients = torch.autograd.grad(actual.float().sum(), weights)
        expected_gradients = torch.autograd.grad(expected.float().sum(), weights)
        for actual_gradient, expected_gradient in zip(
            actual_gradients, expected_gradients
        ):
            assert torch.equal(actual_gradient, expected_gradient)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("token_dtype", [torch.int32, torch.int64])
def test_fused_channel_embeddings_preserve_rounding_and_storage_changes(
    dtype: torch.dtype, token_dtype: torch.dtype
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(42)
    weights = [
        torch.randn(128, 257, device="cuda", dtype=dtype, generator=generator)
        for _ in range(33)
    ]
    tables = ChannelEmbeddingTables()
    assert tables.refresh(weights)
    token_ids = torch.randint(
        0, 128, (17, 66), device="cuda", dtype=token_dtype, generator=generator
    )[:, ::2]
    for moved_weights in (weights, [weight.clone() * 2 for weight in weights]):
        assert tables.refresh(moved_weights)
        expected = torch.zeros(17, 257, device="cuda", dtype=dtype)
        for index, weight in enumerate(moved_weights):
            expected = expected + weight[token_ids[:, index].long()]
        actual = tables.embed(token_ids, moved_weights)
        assert torch.equal(actual, expected)
        saved = actual.clone()
        tables.embed(token_ids, moved_weights).zero_()
        assert torch.equal(actual, saved)
    assert not tables.refresh([weights[0].t()])
    assert tables.embed(token_ids[:0], moved_weights).shape == (0, 257)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
def test_cuda_model_embeddings_keep_eager_fallback_with_runtime_context(
    monkeypatch: pytest.MonkeyPatch, tensor_parallel_size: int
) -> None:
    monkeypatch.setattr(
        "sglang_omni.models.moss_tts.sglang_model.get_parallel",
        lambda: SimpleNamespace(tp_size=tensor_parallel_size),
    )
    generator = torch.Generator(device="cuda").manual_seed(42)
    layers = [
        torch.nn.Embedding.from_pretrained(
            torch.randn(16, 8, device="cuda", dtype=torch.bfloat16, generator=generator)
        )
        for _ in range(3)
    ]
    model = SimpleNamespace(
        config=SimpleNamespace(channels=3),
        hidden_size=8,
        pp_group=SimpleNamespace(is_first_rank=True),
        embedding_list=layers,
        first_embedding_weight=lambda: layers[0].weight,
    )
    model.fused_channel_embedding_weights = MethodType(
        MossTTSDelaySGLangModel.fused_channel_embedding_weights, model
    )
    token_ids = torch.randint(0, 16, (7, 3), device="cuda", generator=generator)
    expected = torch.zeros(7, 8, device="cuda", dtype=torch.bfloat16)
    for channel, layer in enumerate(layers):
        expected = expected + layer(token_ids[:, channel])
    actual = MossTTSDelaySGLangModel.prepare_multi_modal_inputs(model, token_ids)
    assert torch.equal(actual, expected)
