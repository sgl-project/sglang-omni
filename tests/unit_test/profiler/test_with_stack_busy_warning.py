# SPDX-License-Identifier: Apache-2.0
"""Warn when a torch profiler start with ``with_stack`` hits a busy stage.

Starting the torch profiler with ``SGLANG_TORCH_PROFILER_WITH_STACK=1``
while the stage has requests in flight can deadlock the whole stage
process: torch's PythonTracer replays every thread's Python frames while
holding a swapped thread state, and a concurrent module forward makes it
block on a GIL it already holds (#1779). The stage cannot fix torch, but
it can say loudly that this start is entering known-risky territory.
"""

from __future__ import annotations

import logging

import pytest

from sglang_omni.profiler.torch_profiler import TorchProfiler
from sglang_omni.proto.messages import ProfilerStartMessage
from tests.unit_test.pipeline.helpers import make_stage


@pytest.fixture()
def profiler_start_calls(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    """Stub TorchProfiler.start so no real global profiler is attached."""
    calls: list[str | None] = []

    def _fake_start(trace_path_template: str, run_id: str | None = None) -> str:
        calls.append(run_id)
        return f"{trace_path_template}.trace.json.gz"

    monkeypatch.setattr(TorchProfiler, "start", _fake_start)
    return calls


def _start_message() -> ProfilerStartMessage:
    return ProfilerStartMessage(
        run_id="r1",
        trace_path_template="/tmp/profiles/{run_id}/{stage}/trace",
    )


def test_with_stack_on_busy_stage_warns_but_still_starts(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    profiler_start_calls: list[str | None],
) -> None:
    monkeypatch.setenv("SGLANG_TORCH_PROFILER_WITH_STACK", "1")
    stage = make_stage()
    stage._active_requests.add("req-1")

    with caplog.at_level(logging.WARNING):
        stage._on_profiler_start(_start_message())

    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and "SGLANG_TORCH_PROFILER_WITH_STACK" in r.getMessage()
    ]
    assert warnings, "expected a with_stack-on-busy-stage warning"
    assert "deadlock" in warnings[0].getMessage()
    # Advisory only: the profiler start itself must not be blocked.
    assert profiler_start_calls == ["r1"]


def test_with_stack_off_busy_stage_is_silent(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    profiler_start_calls: list[str | None],
) -> None:
    monkeypatch.delenv("SGLANG_TORCH_PROFILER_WITH_STACK", raising=False)
    stage = make_stage()
    stage._active_requests.add("req-1")

    with caplog.at_level(logging.WARNING):
        stage._on_profiler_start(_start_message())

    assert not any(
        "SGLANG_TORCH_PROFILER_WITH_STACK" in r.getMessage() for r in caplog.records
    )
    assert profiler_start_calls == ["r1"]


def test_with_stack_on_idle_stage_is_silent(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    profiler_start_calls: list[str | None],
) -> None:
    monkeypatch.setenv("SGLANG_TORCH_PROFILER_WITH_STACK", "1")
    stage = make_stage()
    assert not stage._active_requests

    with caplog.at_level(logging.WARNING):
        stage._on_profiler_start(_start_message())

    assert not any(
        "SGLANG_TORCH_PROFILER_WITH_STACK" in r.getMessage() for r in caplog.records
    )
    assert profiler_start_calls == ["r1"]
