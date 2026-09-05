# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import inspect
import textwrap
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from sglang_omni.models.moss_tts_realtime.local_cuda_graph import (
    MossTTSRealtimeLocalCudaGraphReplayError,
    MossTTSRealtimeLocalCudaGraphRunner,
)
from sglang_omni.models.moss_tts_realtime.local_transformer import (
    MossTTSRealtimeLocalTransformerForCausalLM,
)


def _config() -> PretrainedConfig:
    return PretrainedConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        rope_theta=10000.0,
        max_position_embeddings=32,
        attention_bias=False,
        attention_dropout=0.0,
        audio_vocab_size=1027,
        audio_pad_token=1024,
        rvq=16,
    )


def _fake_local(*, rvq: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(),
        device=torch.device("cuda"),
        dtype=torch.float32,
        config=SimpleNamespace(rvq=rvq, hidden_size=4),
    )


def test_local_graph_capture_uses_thread_local_error_mode() -> None:
    source = textwrap.dedent(
        inspect.getsource(MossTTSRealtimeLocalCudaGraphRunner._capture_batch_size)
    )
    tree = ast.parse(source)
    graph_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "graph"
    ]

    assert graph_calls
    assert any(
        any(
            keyword.arg == "capture_error_mode"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "thread_local"
            for keyword in call.keywords
        )
        for call in graph_calls
    )


def test_local_graph_capture_failure_cleans_partial_bucket_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    local = _fake_local()
    local.model._ensure_kv_cache = lambda batch_size, **kwargs: events.append(
        ("ensure", batch_size, kwargs)
    )
    local.model.freeze_kv_cache = lambda: events.append("freeze")
    runner = MossTTSRealtimeLocalCudaGraphRunner(
        local,
        batch_sizes=[1, 2],
        max_batch_size=2,
        min_free_gb=0.0,
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(runner, "_has_free_memory", lambda: (True, 2**40))

    def fake_capture(batch_size: int) -> None:
        runner._graph_pools[batch_size] = object()
        for codebook in range(local.config.rvq):
            runner._graphs[(batch_size, codebook)] = object()
        if batch_size == 1:
            raise RuntimeError("synthetic capture failure")

    monkeypatch.setattr(runner, "_capture_batch_size", fake_capture)

    assert runner.warmup() == [2]
    assert events[0][0:2] == ("ensure", 2)
    assert events[1] == "freeze"
    assert not runner.supports(1)
    assert runner.supports(2)
    assert 1 not in runner._graph_pools
    snapshot = runner.resource_snapshot()
    assert snapshot["local_cuda_graph_captured_batch_count"] == 1
    assert snapshot["local_cuda_graph_max_batch_size"] == 2
    assert snapshot["local_cuda_graph_failure_total"] == 1


def test_local_graph_replay_failure_disables_future_replays() -> None:
    class FailingGraph:
        def replay(self) -> None:
            raise RuntimeError("synthetic replay failure")

    runner = MossTTSRealtimeLocalCudaGraphRunner(
        _fake_local(rvq=1),
        batch_sizes=[1],
        max_batch_size=1,
        min_free_gb=0.0,
    )
    runner._graphs[(1, 0)] = SimpleNamespace(
        graph=FailingGraph(),
        static_input=torch.zeros(1, 4),
        static_logits=torch.zeros(1, 8),
    )

    assert runner.supports(1)
    with pytest.raises(
        MossTTSRealtimeLocalCudaGraphReplayError,
        match="future frames will use eager",
    ):
        runner.compute(torch.ones(1, 4), 0)

    assert not runner.supports(1)
    runner.record_fallback()
    snapshot = runner.resource_snapshot()
    assert snapshot["local_cuda_graph_replay_total"] == 0
    assert snapshot["local_cuda_graph_fallback_total"] == 1
    assert snapshot["local_cuda_graph_failure_total"] == 1
    assert snapshot["local_cuda_graph_disabled"] == 1


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_local_compute_graph_is_logit_exact() -> None:
    torch.manual_seed(17)
    local = (
        MossTTSRealtimeLocalTransformerForCausalLM(_config())
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    graph = MossTTSRealtimeLocalCudaGraphRunner(
        local,
        batch_sizes=[1],
        max_batch_size=1,
        warmup_iters=2,
        min_free_gb=0.0,
    )
    assert graph.warmup() == [1]

    hidden = torch.randn(1, 64, device="cuda", dtype=torch.bfloat16)
    prefix = torch.randint(0, 1024, (1, 15), device="cuda")
    expected = local.teacher_forced_logits(hidden, prefix)

    current = hidden
    actual = []
    for codebook in range(16):
        actual.append(graph.compute(current, codebook).clone())
        if codebook < 15:
            current = local.model.embed_tokens[codebook](prefix[:, codebook])
    actual_logits = torch.stack(actual, dim=1)

    assert torch.equal(actual_logits, expected)
    assert graph.resource_snapshot()["local_cuda_graph_replay_total"] == 16
