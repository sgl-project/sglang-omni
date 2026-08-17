# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import pytest
import torch

from sglang_omni.models.qwen3_omni.components import code2wav_cuda_graph
from sglang_omni.models.qwen3_omni.components.code2wav_cuda_graph import (
    Code2WavCudaGraphRunner,
    GraphKey,
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


class _FakeCudaBackend:
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
        self.mark_cuda(tensor, device=device)
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

    def is_cuda_tensor(self, tensor: torch.Tensor) -> bool:
        return id(tensor) in self._tensor_devices

    def tensor_device_matches(self, tensor: torch.Tensor, device: torch.device) -> bool:
        return self._tensor_devices.get(id(tensor)) == device

    def mark_cuda(
        self, tensor: torch.Tensor, *, device: str | torch.device = "cuda:0"
    ) -> torch.Tensor:
        self._tensor_devices[id(tensor)] = torch.device(device)
        return tensor


def _build_runner(
    *,
    backend: _FakeCudaBackend | None = None,
    model: _FakeModel | None = None,
    total_gpu_memory_fraction: float | None = 0.5,
) -> tuple[Code2WavCudaGraphRunner, _FakeCudaBackend, _FakeModel]:
    backend = backend or _FakeCudaBackend()
    model = model or _FakeModel()
    runner = Code2WavCudaGraphRunner.build(
        model,
        device="cuda:0",
        num_quantizers=16,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        graph_keys=_DEFAULT_GRAPH_KEYS,
        cuda_api=backend,
    )
    return runner, backend, model


def _codes(
    backend: _FakeCudaBackend,
    batch_size: int,
    frames: int,
    *,
    num_quantizers: int = 16,
    dtype: torch.dtype = torch.long,
    device: str = "cuda:0",
) -> torch.Tensor:
    tensor = torch.arange(
        batch_size * num_quantizers * frames,
        dtype=dtype,
    ).reshape(batch_size, num_quantizers, frames)
    return backend.mark_cuda(tensor, device=device)


def test_build_captures_only_the_explicit_graph_keys() -> None:
    graph_keys = tuple(GraphKey(batch_size=1, frames=frames) for frames in (20, 40, 45))
    backend = _FakeCudaBackend()
    runner = Code2WavCudaGraphRunner.build(
        _FakeModel(),
        device="cuda:0",
        num_quantizers=16,
        total_gpu_memory_fraction=0.5,
        graph_keys=graph_keys,
        cuda_api=backend,
    )

    assert [tuple(graph.static_input.shape) for graph in backend.graphs] == [
        (1, 16, 45),
        (1, 16, 40),
        (1, 16, 20),
    ]
    assert runner.stats()["graph_contract"]["keys"] == [
        {"batch_size": key.batch_size, "frames": key.frames} for key in graph_keys
    ]
    assert runner.stats()["build"] == {
        "attempted_graph_count": 3,
        "published_graph_count": 3,
    }


def test_build_uses_three_warmups_one_private_pool_and_atomic_publication() -> None:
    runner, backend, _model = _build_runner()

    assert backend.warmup_iterations == [3] * 4
    assert [tuple(graph.static_input.shape) for graph in backend.graphs] == [
        (1, 16, 35),
        (1, 16, 30),
        (1, 16, 20),
        (1, 16, 10),
    ]
    assert backend.pool_calls == 1
    assert len({id(pool) for pool in backend.capture_pools}) == 1
    assert backend.synchronize_calls >= 1

    stats = runner.stats()
    assert stats["enabled"] is True
    assert stats["graph_contract"]["keys"] == [
        {"batch_size": key.batch_size, "frames": key.frames}
        for key in _DEFAULT_GRAPH_KEYS
    ]
    assert stats["build"]["published_graph_count"] == 4
    assert stats["memory"] == {
        "total_gpu_memory_fraction": 0.5,
        "stage_budget_bytes": 500,
        "loaded_model_footprint_bytes": 100,
        "graph_budget_bytes": 400,
        "graph_footprint_bytes": 80,
        "before": {
            "allocated_bytes": 100,
            "reserved_bytes": 120,
            "max_reserved_bytes": 130,
            "free_bytes": 900,
            "total_bytes": 1000,
        },
        "after": {
            "allocated_bytes": 160,
            "reserved_bytes": 200,
            "max_reserved_bytes": 250,
            "free_bytes": 820,
            "total_bytes": 1000,
        },
    }


def test_build_reuses_one_private_stream_for_all_warmups_and_captures() -> None:
    runner, backend, _model = _build_runner()

    assert runner.stats()["enabled"] is True
    assert backend.new_stream_devices == [torch.device("cuda:0")]
    assert len(backend.warmup_streams) == 4
    assert len(backend.capture_streams) == 4
    private_stream = backend.warmup_streams[0]
    assert private_stream is not None
    assert all(stream is private_stream for stream in backend.warmup_streams)
    assert all(stream is private_stream for stream in backend.capture_streams)


def test_stats_report_only_operational_state() -> None:
    runner, backend, _model = _build_runner()
    runner.run(_codes(backend, 1, 10))

    stats = runner.stats()
    assert stats["binding"] == {
        "device": "cuda:0",
        "num_quantizers": 16,
        "input_dtype": "torch.long",
        "owner_pid": stats["binding"]["owner_pid"],
    }
    assert stats["build"] == {
        "attempted_graph_count": 4,
        "published_graph_count": 4,
    }
    assert stats["runtime"] == {
        "graph_replays": 1,
        "replay_failures": 0,
        "fallback_counts": {},
    }


def test_all_serving_keys_hit_while_batch_two_misses() -> None:
    runner, backend, model = _build_runner()

    for key in _DEFAULT_GRAPH_KEYS:
        result = runner.run(_codes(backend, key.batch_size, key.frames))
        assert result.execution_mode == "cuda_graph"
        assert result.key == key
        assert result.fallback_reason is None

    calls_before = len(model.calls)
    missed = runner.run(_codes(backend, 2, 10))

    assert missed.execution_mode == "eager"
    assert missed.key == GraphKey(batch_size=2, frames=10)
    assert missed.fallback_reason == "key_miss"
    assert len(model.calls) == calls_before + 1
    assert runner.stats()["runtime"] == {
        "graph_replays": 4,
        "replay_failures": 0,
        "fallback_counts": {"key_miss": 1},
    }


def test_cuda_api_restores_original_stream_when_capture_exit_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_stream = object()
    current = {"stream": original_stream}

    class _SideStream:
        def wait_stream(self, stream: object) -> None:
            assert stream is original_stream

    side_stream = _SideStream()

    class _FailingCaptureContext:
        def __enter__(self) -> None:
            current["stream"] = side_stream

        def __exit__(self, *_args: object) -> None:
            raise RuntimeError("fake capture_end failed")

    def graph_context(*_args: object, **kwargs: object) -> _FailingCaptureContext:
        assert kwargs["stream"] is side_stream
        return _FailingCaptureContext()

    monkeypatch.setattr(
        code2wav_cuda_graph.torch.cuda,
        "current_stream",
        lambda _device: current["stream"],
    )
    monkeypatch.setattr(
        code2wav_cuda_graph.torch.cuda,
        "set_stream",
        lambda stream: current.update(stream=stream),
    )
    monkeypatch.setattr(
        code2wav_cuda_graph.torch.cuda,
        "CUDAGraph",
        lambda: object(),
    )
    monkeypatch.setattr(code2wav_cuda_graph.torch.cuda, "graph", graph_context)

    with pytest.raises(RuntimeError, match="fake capture_end failed"):
        code2wav_cuda_graph._TorchCudaApi().capture(
            _FakeModel(),
            torch.zeros((1, 16, 10), dtype=torch.long),
            pool=object(),
            stream=side_stream,
        )

    assert current["stream"] is original_stream


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_real_cuda_invalid_capture_preserves_current_stream() -> None:
    api = code2wav_cuda_graph._TorchCudaApi()
    device = torch.device("cuda", torch.cuda.current_device())
    original_stream = torch.cuda.current_stream(device)
    side_stream = api.new_stream(device)
    static_input = torch.zeros((1, 16, 10), dtype=torch.long, device=device)

    def invalidate_capture(codes: torch.Tensor) -> torch.Tensor:
        # Host synchronization is prohibited during graph capture. This marks
        # the capture invalid so capture_end itself exercises the failure path
        # that used to skip the graph context's normal stream restoration.
        torch.cuda.synchronize(device)
        return codes

    current_after: torch.cuda.Stream | None = None
    try:
        with pytest.raises(RuntimeError, match="(?i)captur"):
            api.capture(
                invalidate_capture,
                static_input,
                pool=torch.cuda.graph_pool_handle(),
                stream=side_stream,
            )
        current_after = torch.cuda.current_stream(device)
    finally:
        torch.cuda.set_stream(original_stream)

    assert current_after == original_stream


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_real_cuda_shared_pool_replays_batch_sizes_with_eager_parity() -> None:
    class _TinyCode2WavModel(torch.nn.Module):
        def forward(self, codes: torch.Tensor) -> torch.Tensor:
            return (codes.float() * 2).sum(dim=1, keepdim=True)

    device = torch.device("cuda", torch.cuda.current_device())
    model = _TinyCode2WavModel().to(device).eval()
    graph_keys = (
        GraphKey(batch_size=1, frames=10),
        GraphKey(batch_size=2, frames=10),
    )
    runner = Code2WavCudaGraphRunner.build(
        model,
        device=device,
        num_quantizers=2,
        total_gpu_memory_fraction=1.0,
        graph_keys=graph_keys,
    )

    stats = runner.stats()
    assert stats["enabled"] is True
    assert stats["build"]["published_graph_count"] == len(graph_keys)

    for replay_index, key in enumerate((*graph_keys, *graph_keys)):
        codes = (
            torch.arange(
                key.batch_size * 2 * key.frames,
                dtype=torch.long,
                device=device,
            ).reshape(key.batch_size, 2, key.frames)
            + replay_index * 1000
        )
        with torch.inference_mode():
            eager = model(codes).clone()
        result = runner.run(codes)
        graph_output = result.output.clone()

        assert result.execution_mode == "cuda_graph"
        assert result.key == key
        assert torch.equal(graph_output, eager)


def test_run_copies_live_input_replays_and_returns_borrowed_output_metadata() -> None:
    runner, backend, _model = _build_runner()
    graph = next(
        graph
        for graph in backend.graphs
        if tuple(graph.static_input.shape) == (1, 16, 10)
    )
    first_codes = _codes(backend, 1, 10)

    first = runner.run(first_codes)
    first_snapshot = first.output.clone()

    assert first.execution_mode == "cuda_graph"
    assert first.key == GraphKey(batch_size=1, frames=10)
    assert first.fallback_reason is None
    assert first.output is graph.static_output
    assert torch.equal(graph.replay_inputs[-1], first_codes)

    second_codes = _codes(backend, 1, 10) + 7
    backend.mark_cuda(second_codes)
    second = runner.run(second_codes)

    assert second.output is first.output
    assert not torch.equal(first.output, first_snapshot)
    assert torch.equal(graph.replay_inputs[-1], second_codes)
    assert runner.stats()["runtime"]["graph_replays"] == 2


@pytest.mark.parametrize(
    ("runner_state", "eligible", "batch_size", "expected_reason"),
    [
        ("disabled", True, 1, "disabled"),
        ("enabled", False, 1, "ineligible"),
        ("enabled", True, 2, "key_miss"),
    ],
)
def test_intentional_eager_fallbacks(
    runner_state: str,
    eligible: bool,
    batch_size: int,
    expected_reason: str,
) -> None:
    fraction = None if runner_state == "disabled" else 0.5
    runner, backend, model = _build_runner(total_gpu_memory_fraction=fraction)
    codes = _codes(backend, batch_size, 10)

    calls_before = len(model.calls)
    result = runner.run(codes, eligible=eligible)

    assert result.execution_mode == "eager"
    assert result.fallback_reason == expected_reason
    assert len(model.calls) == calls_before + 1
    assert runner.stats()["runtime"]["fallback_counts"] == {expected_reason: 1}
    if expected_reason == "key_miss":
        assert result.key == GraphKey(batch_size=batch_size, frames=10)


@pytest.mark.parametrize(
    ("case", "expected_error", "message"),
    [
        ("non_cuda", TypeError, "CUDA tensor"),
        ("wrong_dtype", TypeError, "torch.long"),
        ("wrong_device", ValueError, "cuda:0"),
        ("wrong_shape", ValueError, "shape"),
        ("wrong_num_quantizers", ValueError, "16 quantizers"),
    ],
)
def test_eligible_input_contract_violations_raise(
    case: str,
    expected_error: type[Exception],
    message: str,
) -> None:
    runner, backend, model = _build_runner()
    if case == "non_cuda":
        codes = torch.zeros((1, 16, 10), dtype=torch.long)
    elif case == "wrong_dtype":
        codes = _codes(backend, 1, 10, dtype=torch.int32)
    elif case == "wrong_device":
        codes = _codes(backend, 1, 10, device="cuda:1")
    elif case == "wrong_shape":
        codes = backend.mark_cuda(torch.zeros((16, 10), dtype=torch.long))
    else:
        codes = _codes(backend, 1, 10, num_quantizers=15)
    calls_before = len(model.calls)

    with pytest.raises(expected_error, match=message):
        runner.run(codes)

    assert len(model.calls) == calls_before
    assert runner.stats()["runtime"]["fallback_counts"] == {}


@pytest.mark.parametrize(
    ("runner_state", "eligible"),
    [
        ("enabled", True),
        ("ineligible", False),
        ("disabled", True),
    ],
)
def test_pid_mismatch_fails_closed_before_any_eager_model_call(
    runner_state: str,
    eligible: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fraction = None if runner_state == "disabled" else 0.5
    runner, backend, model = _build_runner(total_gpu_memory_fraction=fraction)
    calls_before = len(model.calls)
    monkeypatch.setattr(
        code2wav_cuda_graph.os,
        "getpid",
        lambda: runner.stats()["binding"]["owner_pid"] + 1,
    )

    with pytest.raises(RuntimeError, match="must be rebuilt in a spawned process"):
        runner.run(_codes(backend, 1, 10), eligible=eligible)

    assert len(model.calls) == calls_before
    runtime = runner.stats()["runtime"]
    assert runtime["fallback_counts"] == {}
    json.dumps(runner.stats(), allow_nan=False)


@pytest.mark.parametrize(
    ("backend_factory", "fraction", "reason_prefix"),
    [
        (lambda: _FakeCudaBackend(capture_error_at=3), 0.5, "capture_failed"),
        (lambda: _FakeCudaBackend(replay_error_at=3), 0.5, "capture_failed"),
        (lambda: _FakeCudaBackend(corrupt_at=3), 0.5, "equivalence_failed"),
        (lambda: _FakeCudaBackend(), 0.15, "memory_budget_exceeded"),
    ],
)
def test_any_build_failure_rolls_back_every_graph(
    backend_factory: Callable[[], _FakeCudaBackend],
    fraction: float,
    reason_prefix: str,
) -> None:
    backend = backend_factory()
    runner, backend, _model = _build_runner(
        backend=backend,
        total_gpu_memory_fraction=fraction,
    )

    stats = runner.stats()
    assert stats["enabled"] is False
    assert stats["build"]["published_graph_count"] == 0
    assert stats["disable_reason"].startswith(reason_prefix)
    assert backend.empty_cache_calls >= 1
    memory = stats["memory"]
    assert memory["after"]["allocated_bytes"] == 160
    assert memory["after_rollback"]["allocated_bytes"] == 100


@pytest.mark.parametrize("fraction", [None, 0.0, -0.1, 1.01, float("nan")])
def test_memory_fraction_must_be_explicit_and_in_range(
    fraction: float | None,
) -> None:
    runner, _backend, _model = _build_runner(
        total_gpu_memory_fraction=fraction,
    )

    stats = runner.stats()
    assert stats["enabled"] is False
    assert stats["build"]["published_graph_count"] == 0
    assert stats["disable_reason"] == "invalid_total_gpu_memory_fraction"


def test_runtime_replay_failure_is_raised_and_disables_all_graphs() -> None:
    runner, backend, _model = _build_runner()
    build_stats = runner.stats()["build"]
    graph = next(
        graph
        for graph in backend.graphs
        if tuple(graph.static_input.shape) == (1, 16, 10)
    )
    graph.fail_replay = RuntimeError("replay exploded")

    with pytest.raises(RuntimeError, match="replay exploded"):
        runner.run(_codes(backend, 1, 10))

    stats = runner.stats()
    assert stats["enabled"] is False
    assert stats["disable_reason"] == (
        "runtime_replay_failed: RuntimeError: replay exploded"
    )
    assert stats["build"] == build_stats
    assert stats["runtime"]["replay_failures"] == 1

    calls_before = len(_model.calls)
    fallback = runner.run(_codes(backend, 1, 10))
    assert fallback.execution_mode == "eager"
    assert fallback.fallback_reason == "disabled"
    assert len(_model.calls) == calls_before + 1


def test_stats_are_strictly_json_safe_after_success_and_failure() -> None:
    successful, successful_backend, _model = _build_runner()
    successful.run(_codes(successful_backend, 3, 10))
    failed, _failed_backend, _model = _build_runner(
        backend=_FakeCudaBackend(corrupt_at=0)
    )

    json.dumps(successful.stats(), allow_nan=False)
    json.dumps(failed.stats(), allow_nan=False)


_TIERED_GRAPH_KEYS = (
    GraphKey(batch_size=1, frames=10),
    GraphKey(batch_size=1, frames=20),
    GraphKey(batch_size=2, frames=10),
    GraphKey(batch_size=2, frames=20),
    GraphKey(batch_size=4, frames=10),
    GraphKey(batch_size=4, frames=20),
)


class _SequencedBackend(_FakeCudaBackend):
    """Fake backend with an explicit memory-snapshot schedule and optional
    per-capture-index errors, for driving the tiered budget logic."""

    def __init__(
        self,
        *,
        snapshots: list[tuple[int, int]],
        errors_at: dict[int, Exception] | None = None,
    ) -> None:
        super().__init__()
        self._memory_snapshots = [
            {
                "allocated_bytes": allocated,
                "reserved_bytes": reserved,
                "max_reserved_bytes": reserved,
                "free_bytes": 1000 - allocated,
                "total_bytes": 1000,
            }
            for allocated, reserved in snapshots
        ]
        self._errors_at = errors_at or {}

    def capture(self, model, static_input, *, pool, stream=None):
        error = self._errors_at.get(self.capture_calls)
        if error is not None:
            self.capture_calls += 1
            self.capture_pools.append(pool)
            raise error
        return super().capture(model, static_input, pool=pool, stream=stream)


def _build_tiered_runner(
    backend: _SequencedBackend,
) -> Code2WavCudaGraphRunner:
    return Code2WavCudaGraphRunner.build(
        _FakeModel(),
        device="cuda:0",
        num_quantizers=16,
        total_gpu_memory_fraction=0.5,
        graph_keys=_TIERED_GRAPH_KEYS,
        cuda_api=backend,
    )


def test_tier1_publishes_full_matrix_within_budget() -> None:
    backend = _SequencedBackend(
        snapshots=[
            (100, 120),  # before
            (100, 120),  # attempt baseline
            (300, 340),  # after b4t20
            (360, 400),  # after b4t10
            (390, 430),  # after b2t20
            (400, 440),  # after b2t10
            (420, 460),  # final combined footprint
        ],
    )
    runner = _build_tiered_runner(backend)

    assert [tuple(graph.static_input.shape) for graph in backend.graphs] == [
        (4, 16, 20),
        (4, 16, 10),
        (2, 16, 20),
        (2, 16, 10),
        (1, 16, 20),
        (1, 16, 10),
    ]
    assert backend.pool_calls == 1
    assert len({id(pool) for pool in backend.capture_pools}) == 1

    stats = runner.stats()
    assert stats["enabled"] is True
    assert stats["build"]["published_graph_count"] == 6
    tier1 = stats["memory"]["tier1"]
    assert tier1["attempts"] == 1
    assert tier1["published_key_count"] == 4
    assert tier1["skipped_keys"] == []
    assert runner.available_batch_sizes(10) == (4, 2, 1)
    assert runner.available_batch_sizes(20) == (4, 2, 1)
    assert runner.available_batch_sizes(99) == ()

    result = runner.run(_codes(backend, 4, 20))
    assert result.execution_mode == "cuda_graph"
    assert result.key == GraphKey(batch_size=4, frames=20)
    json.dumps(stats, allow_nan=False)


def test_tier1_budget_violation_republishes_greedy_prefix() -> None:
    backend = _SequencedBackend(
        snapshots=[
            (100, 120),  # before
            (100, 120),  # attempt 1 baseline
            (300, 340),  # after b4t20
            (400, 440),  # after b4t10
            (560, 600),  # b2t20 pushes footprint past the 400 budget
            (100, 120),  # attempt 2 baseline
            (300, 340),  # after b4t20
            (400, 440),  # after b4t10
            (420, 460),  # final combined footprint
        ],
    )
    runner = _build_tiered_runner(backend)

    stats = runner.stats()
    tier1 = stats["memory"]["tier1"]
    assert tier1["attempts"] == 2
    assert tier1["published_key_count"] == 2
    assert tier1["skipped_keys"] == [
        {"batch_size": 2, "frames": 10},
        {"batch_size": 2, "frames": 20},
    ]
    assert backend.pool_calls == 2
    assert runner.available_batch_sizes(20) == (4, 1)
    assert runner.available_batch_sizes(10) == (4, 1)
    assert stats["build"]["published_graph_count"] == 4

    hit = runner.run(_codes(backend, 4, 20))
    assert hit.execution_mode == "cuda_graph"
    miss = runner.run(_codes(backend, 2, 20))
    assert miss.execution_mode == "eager"
    assert miss.fallback_reason == "key_miss"


def test_tier1_oversized_first_key_drops_its_batch_class() -> None:
    backend = _SequencedBackend(
        snapshots=[
            (100, 120),  # before
            (100, 120),  # attempt 1 baseline
            (700, 740),  # b4t20 alone exceeds the budget
            (100, 120),  # attempt 2 baseline
            (260, 300),  # after b2t20
            (300, 340),  # after b2t10
            (320, 360),  # final combined footprint
        ],
    )
    runner = _build_tiered_runner(backend)

    tier1 = runner.stats()["memory"]["tier1"]
    assert tier1["attempts"] == 2
    assert tier1["published_key_count"] == 2
    assert tier1["skipped_keys"] == [
        {"batch_size": 4, "frames": 10},
        {"batch_size": 4, "frames": 20},
    ]
    assert runner.available_batch_sizes(20) == (2, 1)


def test_combined_footprint_violation_drops_largest_batch_class() -> None:
    backend = _SequencedBackend(
        snapshots=[
            (100, 120),  # before
            (100, 120),  # attempt 1 baseline
            (300, 340),  # after b4t20
            (360, 400),  # after b4t10
            (390, 430),  # after b2t20
            (400, 440),  # after b2t10
            (520, 560),  # tier 0 pushes the combined footprint past 400
            (100, 120),  # attempt 2 baseline
            (260, 300),  # after b2t20
            (300, 340),  # after b2t10
            (320, 360),  # final combined footprint
        ],
    )
    runner = _build_tiered_runner(backend)

    stats = runner.stats()
    assert stats["enabled"] is True
    assert stats["build"]["published_graph_count"] == 4
    assert stats["memory"]["graph_footprint_bytes"] == 240
    tier1 = stats["memory"]["tier1"]
    assert tier1["attempts"] == 2
    assert tier1["published_key_count"] == 2
    assert tier1["skipped_keys"] == [
        {"batch_size": 4, "frames": 10},
        {"batch_size": 4, "frames": 20},
    ]
    assert backend.pool_calls == 2
    assert runner.available_batch_sizes(20) == (2, 1)


def test_tier1_capture_oom_drops_the_batch_class_and_retries() -> None:
    backend = _SequencedBackend(
        snapshots=[
            (100, 120),  # before
            (100, 120),  # attempt 1 baseline
            (100, 120),  # attempt 2 baseline
            (260, 300),  # after b2t20
            (300, 340),  # after b2t10
            (320, 360),  # final combined footprint
        ],
        errors_at={0: torch.OutOfMemoryError("fake tier1 capture OOM")},
    )
    runner = _build_tiered_runner(backend)

    tier1 = runner.stats()["memory"]["tier1"]
    assert tier1["attempts"] == 2
    assert tier1["published_key_count"] == 2
    assert tier1["disable_reason"] is None
    assert runner.available_batch_sizes(20) == (2, 1)
    assert backend.empty_cache_calls >= 2


def test_tier1_capture_error_abandons_tier_but_keeps_tier0() -> None:
    backend = _SequencedBackend(
        snapshots=[
            (100, 120),  # before
            (100, 120),  # attempt 1 baseline
            (160, 200),  # tier0-only final combined footprint
        ],
        errors_at={0: RuntimeError("fake capture explosion")},
    )
    runner = _build_tiered_runner(backend)

    stats = runner.stats()
    assert stats["enabled"] is True
    assert stats["build"]["published_graph_count"] == 2
    tier1 = stats["memory"]["tier1"]
    assert tier1["attempts"] == 1
    assert tier1["published_key_count"] == 0
    assert tier1["disable_reason"].startswith("capture_failed: RuntimeError")
    assert len(tier1["skipped_keys"]) == 4

    tier0_hit = runner.run(_codes(backend, 1, 10))
    assert tier0_hit.execution_mode == "cuda_graph"
    tier1_miss = runner.run(_codes(backend, 2, 10))
    assert tier1_miss.execution_mode == "eager"
    assert tier1_miss.fallback_reason == "key_miss"
    json.dumps(stats, allow_nan=False)


def test_tier1_equivalence_failure_abandons_tier_with_original_reason() -> None:
    backend = _SequencedBackend(
        snapshots=[
            (100, 120),  # before
            (100, 120),  # attempt 1 baseline
            (160, 200),  # tier0-only final combined footprint
        ],
    )
    backend.corrupt_at = 0
    runner = _build_tiered_runner(backend)

    stats = runner.stats()
    assert stats["enabled"] is True
    assert stats["build"]["published_graph_count"] == 2
    assert stats["memory"]["tier1"]["disable_reason"].startswith("equivalence_failed")


def test_runtime_disable_clears_tier1_availability() -> None:
    backend = _SequencedBackend(
        snapshots=[
            (100, 120),  # before
            (100, 120),  # attempt baseline
            (300, 340),  # after b4t20
            (360, 400),  # after b4t10
            (390, 430),  # after b2t20
            (400, 440),  # after b2t10
            (420, 460),  # final combined footprint
        ],
    )
    runner = _build_tiered_runner(backend)
    assert runner.available_batch_sizes(10) == (4, 2, 1)
    graph = next(
        g for g in backend.graphs if tuple(g.static_input.shape) == (4, 16, 10)
    )
    graph.fail_replay = RuntimeError("replay exploded")

    with pytest.raises(RuntimeError, match="replay exploded"):
        runner.run(_codes(backend, 4, 10))

    assert runner.available_batch_sizes(10) == ()
    assert runner.stats()["enabled"] is False
