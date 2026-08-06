# SPDX-License-Identifier: Apache-2.0
"""Regression: ``TorchProfiler.start`` used ``rank`` before assigning it.

Before scheduler-thread routing, ``Stage._on_profiler_start`` guarded the
call with ``not TorchProfiler.is_active()``, so the "profiler already
active" branch inside ``start`` was unreachable. Routing the lifecycle
through the scheduler admin queue drops that guard, which made the branch
live -- and it referenced ``rank`` above ``rank = cls._get_rank()``, raising
``UnboundLocalError`` for *any* second start while a profiler was running.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sglang_omni.profiler import torch_profiler as torch_profiler_mod
from sglang_omni.profiler.torch_profiler import TorchProfiler


class _FakeProfile:
    """Stand-in for ``torch.profiler.profile``: records lifecycle calls."""

    instances: list["_FakeProfile"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        _FakeProfile.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def export_chrome_trace(self, path: str) -> None:  # pragma: no cover - unused
        Path(path).write_text("{}")


@pytest.fixture(autouse=True)
def _fake_torch_profile(monkeypatch: pytest.MonkeyPatch):
    """Avoid touching the real Kineto profiler and reset singleton state."""
    _FakeProfile.instances = []
    monkeypatch.setattr(torch_profiler_mod, "profile", _FakeProfile)
    yield
    TorchProfiler._profiler = None
    TorchProfiler._active_run_id = None
    TorchProfiler._trace_template = ""


def test_start_is_idempotent_for_the_active_run(tmp_path: Path) -> None:
    """Re-issuing a start for the running run returns the same trace path."""
    template = str(tmp_path / "trace")

    first = TorchProfiler.start(template, run_id="run-1")
    second = TorchProfiler.start(template, run_id="run-1")

    assert second == first
    assert first.endswith("_rank0.trace.json.gz")
    # The active profiler is reused, not torn down and rebuilt.
    assert len(_FakeProfile.instances) == 1
    assert _FakeProfile.instances[0].stopped is False
    assert TorchProfiler.get_active_run_id() == "run-1"


def test_start_restarts_for_a_different_run(tmp_path: Path) -> None:
    """A start for another run stops the previous profiler and rebuilds."""
    TorchProfiler.start(str(tmp_path / "trace"), run_id="run-1")
    TorchProfiler.start(str(tmp_path / "trace"), run_id="run-2")

    assert len(_FakeProfile.instances) == 2
    assert _FakeProfile.instances[0].stopped is True
    assert _FakeProfile.instances[1].started is True
    assert TorchProfiler.get_active_run_id() == "run-2"


def test_start_restarts_when_run_id_is_omitted(tmp_path: Path) -> None:
    """The wildcard start path also resolves rank before using it."""
    TorchProfiler.start(str(tmp_path / "trace"), run_id="run-1")
    TorchProfiler.start(str(tmp_path / "trace"))

    assert len(_FakeProfile.instances) == 2
    assert _FakeProfile.instances[0].stopped is True
    assert TorchProfiler.get_active_run_id() is None
