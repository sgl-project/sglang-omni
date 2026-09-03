# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import torch

from sglang_omni.models.qwen3_omni.components import code2wav_npu_graph
from sglang_omni.models.qwen3_omni.components.code2wav_cuda_graph import GraphKey
from sglang_omni.models.qwen3_omni.components.code2wav_npu_graph import (
    Code2WavNpuGraphRunner,
)

_DEFAULT_GRAPH_KEYS = tuple(
    GraphKey(batch_size=1, frames=frames) for frames in (10, 20, 30, 35)
)


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def __call__(self, codes: torch.Tensor) -> torch.Tensor:
        self.calls.append(codes.detach().clone())
        samples = int(codes.shape[-1]) * 2
        base = codes.float().sum(dim=(1, 2), keepdim=True)
        ramp = torch.arange(samples, dtype=torch.float32).view(1, 1, samples)
        return base + ramp


class _FakeGraph:
    def __init__(
        self,
        model: _FakeModel,
        static_input: torch.Tensor,
        static_output: torch.Tensor,
        *,
        corrupt: bool,
    ) -> None:
        self._model = model
        self.static_input = static_input
        self.static_output = static_output
        self.corrupt = corrupt
        self.fail_replay: Exception | None = None
        self.replay_inputs: list[torch.Tensor] = []

    def replay(self) -> None:
        if self.fail_replay is not None:
            raise self.fail_replay
        self.replay_inputs.append(self.static_input.clone())
        output = self._model(self.static_input)
        if self.corrupt:
            output = output + 1
        self.static_output.copy_(output)


class _FakeNpuBackend:
    def __init__(
        self,
        *,
        capture_error_at: int | None = None,
        corrupt_at: int | None = None,
        replay_error_at: int | None = None,
        after_allocated: int = 160,
        after_reserved: int = 200,
    ) -> None:
        self.capture_error_at = capture_error_at
        self.corrupt_at = corrupt_at
        self.replay_error_at = replay_error_at
        self.capture_calls = 0
        self.pool_calls = 0
        self.capture_pools: list[Any] = []
        self.new_stream_devices: list[torch.device] = []
        self.warmup_streams: list[Any | None] = []
        self.capture_streams: list[Any | None] = []
        self.warmup_iterations: list[int] = []
        self.synchronize_calls = 0
        self.empty_cache_calls = 0
        self.graphs: list[_FakeGraph] = []
        self._tensor_devices: dict[int, torch.device] = {}
        self._memory_snapshots = [
            {
                "allocated_bytes": 100,
                "reserved_bytes": 120,
                "max_reserved_bytes": 130,
                "free_bytes": 900,
                "total_bytes": 1000,
            },
            {
                "allocated_bytes": after_allocated,
                "reserved_bytes": after_reserved,
                "max_reserved_bytes": 250,
                "free_bytes": 820,
                "total_bytes": 1000,
            },
            {
                "allocated_bytes": 100,
                "reserved_bytes": 120,
                "max_reserved_bytes": 250,
                "free_bytes": 900,
                "total_bytes": 1000,
            },
        ]
        self._memory_index = 0

    def device_context(self, device: torch.device):
        del device
        return nullcontext()

    def memory_stats(self, device: torch.device) -> dict[str, int]:
        del device
        index = min(self._memory_index, len(self._memory_snapshots) - 1)
        self._memory_index += 1
        return dict(self._memory_snapshots[index])

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def new_static_input(
        self, shape: tuple[int, int, int], *, device: torch.device
    ) -> torch.Tensor:
        tensor = torch.zeros(shape, dtype=torch.long)
        self.mark_npu(tensor, device=device)
        return tensor

    def warmup(
        self,
        model: _FakeModel,
        static_input: torch.Tensor,
        *,
        iterations: int,
        device: torch.device,
        stream: object | None = None,
    ) -> None:
        del device
        self.warmup_streams.append(stream)
        self.warmup_iterations.append(iterations)
        for _ in range(iterations):
            model(static_input)

    def graph_pool_handle(self) -> object:
        self.pool_calls += 1
        return object()

    def new_stream(self, device: torch.device) -> object:
        self.new_stream_devices.append(device)
        return object()

    def capture(
        self,
        model: _FakeModel,
        static_input: torch.Tensor,
        *,
        pool: object,
        stream: object | None = None,
    ) -> tuple[_FakeGraph, torch.Tensor]:
        call_index = self.capture_calls
        self.capture_calls += 1
        self.capture_pools.append(pool)
        self.capture_streams.append(stream)
        if call_index == self.capture_error_at:
            raise torch.OutOfMemoryError("fake capture OOM")
        static_output = model(static_input).detach().clone()
        graph = _FakeGraph(
            model,
            static_input,
            static_output,
            corrupt=call_index == self.corrupt_at,
        )
        if call_index == self.replay_error_at:
            graph.fail_replay = RuntimeError("fake build replay failed")
        self.graphs.append(graph)
        return graph, static_output

    def synchronize(self, device: torch.device) -> None:
        del device
        self.synchronize_calls += 1

    def is_npu_tensor(self, tensor: torch.Tensor) -> bool:
        return id(tensor) in self._tensor_devices

    def tensor_device_matches(self, tensor: torch.Tensor, device: torch.device) -> bool:
        return self._tensor_devices.get(id(tensor)) == device

    def mark_npu(
        self, tensor: torch.Tensor, *, device: str | torch.device = "npu:0"
    ) -> torch.Tensor:
        self._tensor_devices[id(tensor)] = torch.device(device)
        return tensor


def _build_runner(
    *,
    backend: _FakeNpuBackend | None = None,
    model: _FakeModel | None = None,
    total_gpu_memory_fraction: float | None = 0.5,
) -> tuple[Code2WavNpuGraphRunner, _FakeNpuBackend, _FakeModel]:
    backend = backend or _FakeNpuBackend()
    model = model or _FakeModel()
    runner = Code2WavNpuGraphRunner.build(
        model,
        device="npu:0",
        num_quantizers=16,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        graph_keys=_DEFAULT_GRAPH_KEYS,
        npu_api=backend,
    )
    return runner, backend, model


def _codes(
    backend: _FakeNpuBackend,
    batch_size: int,
    frames: int,
    *,
    num_quantizers: int = 16,
    dtype: torch.dtype = torch.long,
    device: str = "npu:0",
) -> torch.Tensor:
    tensor = torch.arange(
        batch_size * num_quantizers * frames,
        dtype=dtype,
    ).reshape(batch_size, num_quantizers, frames)
    return backend.mark_npu(tensor, device=device)


def test_npu_runner_reuses_graph_lifecycle_and_reports_backend_mode() -> None:
    try:
        torch.device("npu:0")
    except RuntimeError:
        # CPU-only PyTorch does not register Ascend's PrivateUse1 name. This
        # only provides a device descriptor; all graph operations stay fake.
        torch.utils.rename_privateuse1_backend("npu")

    backend = _FakeNpuBackend()
    model = _FakeModel()
    graph_keys = (GraphKey(batch_size=1, frames=10),)
    runner = Code2WavNpuGraphRunner.build(
        model,
        device="npu:0",
        num_quantizers=16,
        total_gpu_memory_fraction=0.5,
        graph_keys=graph_keys,
        npu_api=backend,
    )
    codes = _codes(backend, 1, 10, device="npu:0")

    result = runner.run(codes)

    assert runner.stats()["enabled"] is True
    assert result.execution_mode == "npu_graph"
    assert result.key == graph_keys[0]
    assert torch.equal(result.output, model(codes))


def test_graph_runner_logs_first_successful_replay_per_key(caplog) -> None:
    runner, backend, _model = _build_runner()
    codes = _codes(backend, 1, 10)

    with caplog.at_level(logging.INFO, logger=code2wav_npu_graph.__name__):
        runner.run(codes)
        runner.run(codes)

    replay_records = [
        record
        for record in caplog.records
        if "graph replay active" in record.getMessage()
    ]
    assert len(replay_records) == 1
    assert "execution_mode=npu_graph" in replay_records[0].getMessage()


def test_graph_runner_logs_periodic_runtime_stats(caplog) -> None:
    runner, backend, _model = _build_runner()
    runner._RUNTIME_STATS_LOG_INTERVAL = 2
    codes = _codes(backend, 1, 10)

    with caplog.at_level(logging.INFO, logger=code2wav_npu_graph.__name__):
        runner.run(codes)
        runner.run(codes)

    runtime_records = [
        record
        for record in caplog.records
        if "graph runtime stats" in record.getMessage()
    ]
    assert len(runtime_records) == 1
    assert "graph_replays=2" in runtime_records[0].getMessage()
    assert "replay_failures=0" in runtime_records[0].getMessage()
    assert "fallback_counts={}" in runtime_records[0].getMessage()


def test_graph_runner_logs_only_first_eager_fallback_per_reason(caplog) -> None:
    runner, backend, _model = _build_runner()
    codes = _codes(backend, 1, 10)

    with caplog.at_level(logging.WARNING, logger=code2wav_npu_graph.__name__):
        runner.run(codes, eligible=False)
        runner.run(codes, eligible=False)

    fallback_records = [
        record
        for record in caplog.records
        if "graph eager fallback" in record.getMessage()
    ]
    assert len(fallback_records) == 1
    assert "reason=ineligible" in fallback_records[0].getMessage()
    assert runner.stats()["runtime"]["fallback_counts"] == {"ineligible": 2}


def test_npu_api_enables_auto_dispatch_during_capture(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    class _Stream:
        def wait_stream(self, other) -> None:
            calls.setdefault("waited_for", []).append(other)

    current_stream = _Stream()
    capture_stream = _Stream()
    graph = SimpleNamespace(replay=lambda: None)

    def _graph_context(supplied_graph, **kwargs):
        calls.update(graph=supplied_graph, **kwargs)
        return nullcontext()

    fake_npu = SimpleNamespace(
        current_stream=lambda device: current_stream,
        NPUGraph=lambda: graph,
        graph=_graph_context,
    )
    monkeypatch.setattr(code2wav_npu_graph.torch, "npu", fake_npu, raising=False)
    static_input = torch.zeros((1, 16, 10), dtype=torch.long)

    captured_graph, output = code2wav_npu_graph._TorchNpuApi().capture(
        lambda codes: codes + 1,
        static_input,
        pool="shared-pool",
        stream=capture_stream,
    )

    assert captured_graph is graph
    assert torch.equal(output, static_input + 1)
    assert calls["pool"] == "shared-pool"
    assert calls["stream"] is capture_stream
    assert calls["capture_error_mode"] == "thread_local"
    assert calls["auto_dispatch_capture"] is True
