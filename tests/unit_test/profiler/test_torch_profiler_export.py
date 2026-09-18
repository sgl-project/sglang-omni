# SPDX-License-Identifier: Apache-2.0
"""Regressions for two defects in ``sglang_omni/profiler/torch_profiler.py``.

1. ``start()`` read ``rank`` before assigning it, so every restart path raised
   ``NameError`` — both the idempotent same-``run_id`` return and the
   different-``run_id`` swap.
2. The chrome trace was exported twice. 
   Raising "Trace is already saved." on every stop starting torch 2.13.0

"""

from __future__ import annotations

import pytest

from sglang_omni.profiler.torch_profiler import TorchProfiler


@pytest.fixture(autouse=True)
def _reset_profiler_state():
    yield
    if isinstance(TorchProfiler._profiler, _DummyProfiler):
        TorchProfiler._profiler = None
    elif TorchProfiler._profiler is not None:
        try:
            TorchProfiler._profiler.stop()
        except Exception:
            pass
        TorchProfiler._profiler = None
    TorchProfiler._active_run_id = None
    TorchProfiler._trace_template = ""
    TorchProfiler._trace_exported = False


class _DummyProfiler:
    """Stands in for ``torch.profiler.profile`` so stop() can be driven alone."""

    def __init__(self) -> None:
        self.exported: list[str] = []

    def stop(self) -> None:
        pass

    def export_chrome_trace(self, path: str) -> None:
        self.exported.append(path)
        with open(path, "w") as f:
            f.write("{}")


def _arm_stop(tmp_path, *, handler_exported: bool) -> _DummyProfiler:
    dummy = _DummyProfiler()
    TorchProfiler._profiler = dummy
    TorchProfiler._trace_template = str(tmp_path / "run")
    TorchProfiler._active_run_id = "demo"
    TorchProfiler._trace_exported = handler_exported
    return dummy


def test_stop_exports_when_the_handler_did_not_fire(tmp_path):
    dummy = _arm_stop(tmp_path, handler_exported=False)

    result = TorchProfiler.stop(run_id="demo")

    assert dummy.exported == [str(tmp_path / "run_rank0.trace.json")]
    assert result["trace"] == str(tmp_path / "run_rank0.trace.json.gz")


def test_stop_does_not_export_again_when_the_handler_fired(tmp_path):
    """The double export is what raised "Trace is already saved." every stop."""
    dummy = _arm_stop(tmp_path, handler_exported=True)

    result = TorchProfiler.stop(run_id="demo")

    assert dummy.exported == []
    assert result["trace"] == str(tmp_path / "run_rank0.trace.json.gz")


def test_stop_resets_the_export_flag_for_the_next_run(tmp_path):
    _arm_stop(tmp_path, handler_exported=True)

    TorchProfiler.stop(run_id="demo")

    assert TorchProfiler._trace_exported is False


def test_mismatched_run_id_leaves_the_flag_alone(tmp_path):
    dummy = _arm_stop(tmp_path, handler_exported=True)

    assert TorchProfiler.stop(run_id="other") is None
    assert dummy.exported == []
    assert TorchProfiler._trace_exported is True


def test_restart_with_the_same_run_id_returns_the_existing_path(tmp_path):
    """Used to raise NameError: rank was read before it was assigned."""
    template = str(tmp_path / "run")

    first = TorchProfiler.start(template, run_id="a")
    again = TorchProfiler.start(template, run_id="a")

    assert again == first
    assert TorchProfiler.get_active_run_id() == "a"


def test_restart_with_a_different_run_id_swaps_the_profiler(tmp_path):
    """The other NameError path — it died before stopping the old profiler."""
    template = str(tmp_path / "run")

    TorchProfiler.start(template, run_id="a")
    first_profiler = TorchProfiler._profiler
    TorchProfiler.start(str(tmp_path / "run-b"), run_id="b")

    assert TorchProfiler.get_active_run_id() == "b"
    assert TorchProfiler._profiler is not first_profiler
