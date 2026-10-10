# SPDX-License-Identifier: Apache-2.0
"""Torch trace lifecycle and worker-thread component coverage."""

import json
import threading
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from sglang_omni.models.personaplex.profiling import component_scope
from sglang_omni.profiler import torch_profiler
from sglang_omni.profiler.torch_profiler import TorchProfiler


@pytest.fixture(autouse=True)
def reset_torch_profiler(monkeypatch: pytest.MonkeyPatch) -> Iterator[Mock]:
    compression = Mock()
    monkeypatch.setattr(torch_profiler.subprocess, "Popen", compression)
    monkeypatch.setattr(
        torch_profiler, "profiler_activities", lambda: [ProfilerActivity.CPU]
    )
    monkeypatch.delenv("SGLANG_TORCH_PROFILER_PROFILE_ALL_THREADS", raising=False)
    TorchProfiler.stop()
    yield compression
    TorchProfiler.stop()


def test_profiler_records_existing_worker_and_exports_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reset_torch_profiler: Mock,
) -> None:
    monkeypatch.setenv("SGLANG_TORCH_PROFILER_PROFILE_ALL_THREADS", "1")
    worker_ready = threading.Event()
    start_compute = threading.Event()
    tensor_results: list[torch.Tensor] = []

    def compute() -> None:
        worker_ready.set()
        assert start_compute.wait(timeout=10)
        with component_scope("depformer"):
            tensor_results.append(torch.ones(16).sum())

    worker_thread = threading.Thread(target=compute)
    worker_thread.start()
    assert worker_ready.wait(timeout=10)
    trace_path = tmp_path / "worker"
    TorchProfiler.start(str(trace_path), run_id="worker-run")
    try:
        start_compute.set()
        worker_thread.join(timeout=10)
        assert not worker_thread.is_alive()
    finally:
        start_compute.set()
        worker_thread.join(timeout=10)
        result = TorchProfiler.stop(run_id="worker-run")
    assert result is not None
    compressed_trace_path = result["trace"]
    assert isinstance(compressed_trace_path, str)
    trace = json.loads(Path(compressed_trace_path.removesuffix(".gz")).read_text())
    assert any(
        event.get("cat") == "user_annotation"
        and event.get("name") == "personaplex.depformer"
        for event in trace["traceEvents"]
    )
    assert tensor_results[0].item() == 16
    reset_torch_profiler.assert_called_once_with(
        ["gzip", "-f", compressed_trace_path.removesuffix(".gz")]
    )
    assert not TorchProfiler.is_active()


def test_component_scope_is_disabled_without_omni_profiler() -> None:
    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        with component_scope("mimi_encode"):
            result = torch.ones(16).sum()
    assert result.item() == 16
    assert not any(
        event.name == "personaplex.mimi_encode" for event in profiler.events()
    )
