# SPDX-License-Identifier: Apache-2.0
import asyncio
import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.scheduling.message import OutgoingMessage


@pytest.mark.parametrize("outcome", ["completed", "error", "aborted"])
def test_stage_duration_is_emitted_once(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
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


@pytest.mark.parametrize("active", [False, True])
def test_batched_outbox_forwards_metrics_without_request_admission(
    active: bool,
) -> None:
    stage = object.__new__(Stage)
    stage.name = "decode"
    stage.running = False
    stage.active_requests = {"r"} if active else set()
    stage.aborted = set()
    stage.scheduler = SimpleNamespace(outbox=queue.Queue())
    stage.control_plane = SimpleNamespace(send_stream=AsyncMock())
    stage.record_replica_bindings = Mock()
    stage.route_stream_run = AsyncMock()
    stage.route_result = AsyncMock()
    metrics = {"prompt_tokens": 4, "completion_tokens": 1}
    stage.scheduler.outbox.put(OutgoingMessage("r", "stream", "first", "vocoder"))
    stage.scheduler.outbox.put(OutgoingMessage("r", "metrics", metrics))
    stage.scheduler.outbox.put(OutgoingMessage("r", "admitted"))
    stage.scheduler.outbox.put(OutgoingMessage("r", "result", "finished"))

    asyncio.run(stage.drain_outbox_external())

    stage.route_stream_run.assert_awaited_once()
    stage.control_plane.send_stream.assert_awaited_once()
    forwarded = stage.control_plane.send_stream.call_args.args[0]
    assert forwarded.modality == "metrics"
    assert forwarded.chunk == metrics
    assert forwarded.from_stage == "decode"
    stage.route_result.assert_awaited_once_with("r", "finished")
    assert stage.scheduler.outbox.empty()
