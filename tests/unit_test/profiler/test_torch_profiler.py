"""Cross-thread capture through the runtime profiler."""

import gzip
import json
import threading
import time
from pathlib import Path

import pytest
import torch

from sglang_omni.profiler.torch_profiler import TorchProfiler

STRESS_ROUND_COUNT = 12
STRESS_ITERATIONS_PER_ROUND = 5


def load_trace_events(template: Path) -> list[dict[str, object]]:
    trace = Path(f"{template}_rank0.trace.json.gz")
    uncompressed = trace.with_suffix("")
    deadline = time.monotonic() + 15
    while not trace.exists() or uncompressed.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Trace compression did not complete")
        else:
            time.sleep(0.05)
    with gzip.open(trace, "rt") as handle:
        return json.load(handle)["traceEvents"]


@pytest.mark.parametrize("raise_during_work", [False, True])
def test_existing_worker_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raise_during_work: bool
) -> None:
    monkeypatch.delenv("SGLANG_TORCH_PROFILER_PROFILE_MEMORY", raising=False)
    ready = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    errors: list[str] = []

    def compute() -> None:
        try:
            tensor = torch.ones(16, 16)
            ready.set()
            if not release.wait(10):
                raise TimeoutError("Worker was not released")
            else:
                pass
            with torch.profiler.record_function("omni_scheduler_test"):
                result = torch.mm(tensor, tensor)
            assert torch.equal(
                result, torch.full((16, 16), 16.0)
            ), "Incorrect matrix product"
            if raise_during_work:
                raise RuntimeError("deliberate workload failure")
            else:
                pass
        except Exception as error:
            errors.append(repr(error))
        finally:
            completed.set()

    worker = threading.Thread(
        target=compute, name="scheduler-profiler-test", daemon=True
    )
    worker.start()
    assert ready.wait(10)
    template = tmp_path / "capture"
    TorchProfiler.start(str(template), run_id="worker-test")
    try:
        release.set()
        assert completed.wait(10)
        if raise_during_work:
            assert errors == ["RuntimeError('deliberate workload failure')"]
        else:
            assert not errors, errors
    finally:
        TorchProfiler.stop(run_id="worker-test")
        release.set()
        worker.join(timeout=10)
    assert not TorchProfiler.is_active()
    assert not worker.is_alive()
    events = load_trace_events(template)
    operators = [
        event
        for event in events
        if event.get("cat") == "cpu_op" and event.get("name") == "aten::mm"
    ]
    markers = [
        event
        for event in events
        if event.get("cat") == "user_annotation"
        and event.get("name") == "omni_scheduler_test"
    ]
    assert len(operators) == len(markers) == 1
    assert operators[0]["tid"] == markers[0]["tid"]


def test_memory_profiling_keeps_memory_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SGLANG_TORCH_PROFILER_PROFILE_MEMORY", "1")
    element_count = 1_000_003
    template = tmp_path / "memory"
    TorchProfiler.start(str(template), run_id="memory-test")
    try:
        with torch.profiler.record_function("omni_memory_test"):
            tensor = torch.empty(element_count) + 1
    finally:
        TorchProfiler.stop(run_id="memory-test")
    assert tensor.numel() == element_count
    events = load_trace_events(template)
    allocated_bytes = [
        event["args"]["Bytes"]
        for event in events
        if event.get("name") == "[memory]" and event["args"]["Bytes"] > 0
    ]
    assert element_count * tensor.element_size() in allocated_bytes
    assert any(event.get("name") == "omni_memory_test" for event in events)


def test_repeated_capture_with_running_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SGLANG_TORCH_PROFILER_PROFILE_MEMORY", raising=False)
    shutdown = threading.Event()
    iteration_count = 0
    worker_thread_ids: list[int] = []
    errors: list[str] = []

    def compute() -> None:
        nonlocal iteration_count
        worker_thread_ids.append(threading.get_native_id())
        tensor = torch.ones(16, 16)
        try:
            while not shutdown.is_set():
                with torch.profiler.record_function("omni_scheduler_stress"):
                    result = torch.mm(tensor, tensor)
                assert torch.equal(
                    result, torch.full((16, 16), 16.0)
                ), "Incorrect matrix product"
                iteration_count += 1
        except Exception as error:
            errors.append(repr(error))

    worker = threading.Thread(
        target=compute, name="scheduler-profiler-stress", daemon=True
    )
    worker.start()
    try:
        for round_index in range(STRESS_ROUND_COUNT):
            template = tmp_path / f"round{round_index}"
            run_id = f"stress-{round_index}"
            TorchProfiler.start(str(template), run_id=run_id)
            try:
                target_iteration_count = iteration_count + STRESS_ITERATIONS_PER_ROUND
                deadline = time.monotonic() + 10
                while iteration_count < target_iteration_count:
                    if time.monotonic() >= deadline or errors:
                        raise TimeoutError(f"Worker stalled in round {round_index}")
                    else:
                        time.sleep(0.001)
            finally:
                TorchProfiler.stop(run_id=run_id)
            assert not TorchProfiler.is_active()
            events = load_trace_events(template)
            worker_operators = [
                event
                for event in events
                if event.get("cat") == "cpu_op"
                and event.get("name") == "aten::mm"
                and event.get("tid") == worker_thread_ids[0]
            ]
            assert (
                len(worker_operators) >= STRESS_ITERATIONS_PER_ROUND - 1
            ), f"round {round_index} recorded {len(worker_operators)} worker ops"
    finally:
        shutdown.set()
        worker.join(timeout=10)
    assert not worker.is_alive()
    assert not errors, errors
