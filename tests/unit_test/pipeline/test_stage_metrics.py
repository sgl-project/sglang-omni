# SPDX-License-Identifier: Apache-2.0
import queue
from types import SimpleNamespace

import pytest

from sglang_omni.pipeline.stage.runtime import Stage


@pytest.mark.parametrize("outcome", ["completed", "error", "aborted"])
def test_stage_duration_is_emitted_once(monkeypatch, outcome):
    stage = object.__new__(Stage)
    stage.metrics_started = {"r": 1}
    stage.scheduler = SimpleNamespace(outbox=queue.Queue())
    monkeypatch.setattr(
        "sglang_omni.pipeline.stage.runtime.time.perf_counter", lambda: 4
    )
    stage.finish_stage_metrics("r", outcome)
    stage.finish_stage_metrics("r", "completed")
    assert stage.scheduler.outbox.get_nowait().data == {
        "event": "stage",
        "duration": 3,
        "outcome": outcome,
    }
    assert stage.scheduler.outbox.empty()
    assert not stage.metrics_started
