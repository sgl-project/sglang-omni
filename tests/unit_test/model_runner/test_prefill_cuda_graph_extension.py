# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
    has_standard_gqa_for_all_local_layers,
    index_attention_layers_by_global_id,
)
from sglang.srt.model_executor.model_runner_components.layer_setup import (
    compute_attention_and_moe_layers,
)
from sglang.srt.runtime_context import get_context
from torch import nn

import sglang_omni.model_runner.sglang_model_runner as sglang_model_runner
from sglang_omni.model_runner.request_id_prefill_cuda_graph_runner import (
    RequestIdPrefillCudaGraphRunner,
)
from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner

BACKBONE_LAYER_COUNT = 2
EXTRA_LAYER_COUNT = 2


def attention_layer(layer_id: int) -> RadixAttention:
    return RadixAttention(
        num_heads=2, head_dim=4, scaling=0.5, num_kv_heads=1, layer_id=layer_id
    )


def backbone_layer_model() -> nn.Module:
    layer_model = nn.Module()
    layer_model.layers = nn.ModuleList()
    for layer_id in range(BACKBONE_LAYER_COUNT):
        decoder_layer = nn.Module()
        decoder_layer.self_attn = nn.Module()
        decoder_layer.self_attn.attn = attention_layer(layer_id)
        layer_model.layers.append(decoder_layer)
    return layer_model


def extra_attention_layers() -> list[RadixAttention]:
    return [
        attention_layer(BACKBONE_LAYER_COUNT + index)
        for index in range(EXTRA_LAYER_COUNT)
    ]


def build_runner(monkeypatch: pytest.MonkeyPatch) -> SGLModelRunner:
    monkeypatch.setattr(SGLModelRunner, "register_omni_model", lambda self: None)
    monkeypatch.setattr(
        sglang_model_runner,
        "get_schedule",
        lambda: SimpleNamespace(mem_fraction_static=0.5),
    )
    monkeypatch.setattr(ModelRunner, "__init__", lambda self, **fields: None)
    return SGLModelRunner(
        model_config=None,
        server_args=None,
        gpu_id=0,
        nccl_port=0,
    )


def prefill_backend(backend: str) -> CudaGraphConfig:
    return CudaGraphConfig(decode=PhaseConfig(), prefill=PhaseConfig(backend=backend))


@pytest.mark.parametrize("backend", ["breakable"])
def test_default_runner_keeps_upstream_prefill_graph_layers_and_runner(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    runner = build_runner(monkeypatch)
    layer_model = backbone_layer_model()

    with get_context().override_server_args(cuda_graph_config=prefill_backend(backend)):
        assert runner.get_cuda_graph_layers(
            layer_model
        ) == compute_attention_and_moe_layers(layer_model)
        assert runner.prefill_cuda_graph_runner_cls() is None


def test_breakable_prefill_graph_layers_cover_registered_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = build_runner(monkeypatch)
    layer_model = backbone_layer_model()
    upstream_layers = compute_attention_and_moe_layers(layer_model)
    extra_layers = extra_attention_layers()
    runner.extra_prefill_graph_attention_layers = extra_layers

    with get_context().override_server_args(
        cuda_graph_config=prefill_backend("breakable")
    ):
        layers = runner.get_cuda_graph_layers(layer_model)
    indexed_layers, indexed_companions = index_attention_layers_by_global_id(
        layers.attention_layers, layers.mha_companion_layers, layer_model
    )

    assert layers.attention_layers == upstream_layers.attention_layers + extra_layers
    assert layers.mha_companion_layers == [None] * (
        BACKBONE_LAYER_COUNT + EXTRA_LAYER_COUNT
    )
    assert layers.moe_layers == upstream_layers.moe_layers
    assert layers.moe_fusions == upstream_layers.moe_fusions
    assert layers.dsa_indexers == upstream_layers.dsa_indexers
    assert [layer.layer_id for layer in indexed_layers] == list(
        range(BACKBONE_LAYER_COUNT + EXTRA_LAYER_COUNT)
    )
    assert indexed_companions == [None] * (BACKBONE_LAYER_COUNT + EXTRA_LAYER_COUNT)
    assert has_standard_gqa_for_all_local_layers(
        attention_layer_count=len(indexed_layers),
        start_layer=0,
        end_layer=BACKBONE_LAYER_COUNT + EXTRA_LAYER_COUNT,
    )
    assert not has_standard_gqa_for_all_local_layers(
        attention_layer_count=len(upstream_layers.attention_layers),
        start_layer=0,
        end_layer=BACKBONE_LAYER_COUNT + EXTRA_LAYER_COUNT,
    )


@pytest.mark.parametrize("backend", ["full"])
def test_registrations_apply_only_to_breakable_prefill_graphs(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    runner = build_runner(monkeypatch)
    layer_model = backbone_layer_model()
    runner.extra_prefill_graph_attention_layers = extra_attention_layers()
    runner.prefill_graph_runner_class = RequestIdPrefillCudaGraphRunner

    with get_context().override_server_args(cuda_graph_config=prefill_backend(backend)):
        assert runner.get_cuda_graph_layers(
            layer_model
        ) == compute_attention_and_moe_layers(layer_model)
        assert runner.prefill_cuda_graph_runner_cls() is None


def test_registered_runner_class_takes_precedence_over_whisper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = build_runner(monkeypatch)
    runner.model_arch_override = "WhisperForConditionalGeneration"
    runner.prefill_graph_runner_class = RequestIdPrefillCudaGraphRunner

    with get_context().override_server_args(
        cuda_graph_config=prefill_backend("breakable")
    ):
        assert runner.prefill_cuda_graph_runner_cls() is RequestIdPrefillCudaGraphRunner


def test_runners_do_not_share_prefill_graph_registrations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_runner = build_runner(monkeypatch)
    second_runner = build_runner(monkeypatch)

    first_runner.extra_prefill_graph_attention_layers.extend(extra_attention_layers())
    first_runner.prefill_graph_runner_class = RequestIdPrefillCudaGraphRunner

    assert second_runner.extra_prefill_graph_attention_layers == []
    assert second_runner.prefill_graph_runner_class is None


def test_request_id_prefill_runner_carries_request_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = object.__new__(RequestIdPrefillCudaGraphRunner)
    static_batch = SimpleNamespace(rids=None, extend_num_tokens=5)
    live_batch = SimpleNamespace(rids=["first", "second"], extend_num_tokens=5)
    monkeypatch.setattr(
        RequestIdPrefillCudaGraphRunner.__mro__[1],
        "load_batch",
        lambda self, forward_batch, **kwargs: static_batch,
    )

    replay_batch = runner.load_batch(live_batch)

    assert replay_batch is static_batch
    assert replay_batch.rids is live_batch.rids
    assert replay_batch.extend_num_tokens == 5
