# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import sglang_omni.profiler.torch_profiler as torch_profiler_module
from sglang_omni.profiler.torch_profiler import TorchProfiler


class _FakeProfile:
    instances: list["_FakeProfile"] = []

    def __init__(self, *args, **kwargs):
        self.started = False
        self.stopped = False
        _FakeProfile.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def export_chrome_trace(self, path: str) -> None:  # pragma: no cover
        pass


@pytest.fixture(autouse=True)
def _isolated_profiler(monkeypatch):
    monkeypatch.setattr(torch_profiler_module, "profile", _FakeProfile)
    _FakeProfile.instances = []
    TorchProfiler._profiler = None
    TorchProfiler._active_run_id = None
    TorchProfiler._trace_template = ""
    yield
    TorchProfiler._profiler = None
    TorchProfiler._active_run_id = None
    TorchProfiler._trace_template = ""


def test_start_returns_rank_trace_path(tmp_path) -> None:
    template = str(tmp_path / "run")

    trace_path = TorchProfiler.start(template, run_id="run-a")

    assert trace_path == f"{template}_rank0.trace.json.gz"
    assert TorchProfiler.is_active()
    assert TorchProfiler.get_active_run_id() == "run-a"
    assert len(_FakeProfile.instances) == 1
    assert _FakeProfile.instances[0].started


def test_start_is_idempotent_for_same_run_id(tmp_path) -> None:
    template = str(tmp_path / "run")
    first_path = TorchProfiler.start(template, run_id="run-a")

    second_path = TorchProfiler.start(template, run_id="run-a")

    assert second_path == first_path
    assert len(_FakeProfile.instances) == 1
    assert not _FakeProfile.instances[0].stopped


def test_start_restarts_for_different_run_id(tmp_path) -> None:
    template = str(tmp_path / "run")
    TorchProfiler.start(template, run_id="run-a")

    TorchProfiler.start(template, run_id="run-b")

    assert TorchProfiler.get_active_run_id() == "run-b"
    assert len(_FakeProfile.instances) == 2
    assert _FakeProfile.instances[0].stopped
    assert _FakeProfile.instances[1].started
