"""Cross-thread capture through the runtime profiler."""

import gzip
import json
import threading
import time
from pathlib import Path

import pytest
import torch

from sglang_omni.profiler.torch_profiler import TorchProfiler


@pytest.mark.parametrize("raise_during_work", [False, True])
def test_existing_worker_is_recorded(tmp_path: Path, raise_during_work: bool) -> None:
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
    trace = Path(f"{template}_rank0.trace.json.gz")
    uncompressed = trace.with_suffix("")
    deadline = time.monotonic() + 15
    while not trace.exists() or uncompressed.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Trace compression did not complete")
        else:
            time.sleep(0.05)
    with gzip.open(trace, "rt") as handle:
        events = json.load(handle)["traceEvents"]
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
