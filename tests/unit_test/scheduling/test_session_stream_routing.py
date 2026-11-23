# SPDX-License-Identifier: Apache-2.0
"""A session unit's stream goes to the client only from its route's last stage; before that it flows downstream as usual."""

import queue
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling import omni_scheduler
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.ar_session import SessionUnit


def scheduler_with_unit(stages: tuple[str, ...]):
    scheduler = Mock(spec=OmniScheduler)
    scheduler.aborted_request_ids = set()
    scheduler.outbox = queue.Queue()
    scheduler.first_emit_done = set()
    unit = SessionUnit(
        request_id="unit-1",
        session_identity=SessionIdentity("tts"),
        stages=stages,
        chunk=TimedChunk("text", 0, 0, 0, {"text": "hi"}, format="text_delta"),
    )
    scheduler.session_bridge = Mock()
    scheduler.session_bridge.units_by_request_id = {"unit-1": unit}
    scheduler.session_bridge.stream_messages = Mock(
        return_value=[OutgoingMessage("unit-1", "stream", "from-bridge")]
    )
    scheduler.active_session_unit = MethodType(
        OmniScheduler.active_session_unit, scheduler
    )
    scheduler.stream_output_builder = Mock(
        return_value=[OutgoingMessage("unit-1", "stream", "from-builder")]
    )
    scheduler.stream_output_builder.flush.return_value = [
        OutgoingMessage("unit-1", "stream", "flushed-builder")
    ]
    scheduler.stage_stream_messages = MethodType(
        OmniScheduler.stage_stream_messages, scheduler
    )
    scheduler.put_stream_messages = MethodType(
        OmniScheduler.put_stream_messages, scheduler
    )
    sched_output = SimpleNamespace(
        requests=[SimpleNamespace(request_id="unit-1", data="data")]
    )
    mr_output = SimpleNamespace(outputs={"unit-1": "output"})
    return scheduler, sched_output, mr_output


@pytest.mark.parametrize(
    "stages, expected",
    [(("tts_engine", "vocoder"), "from-builder"), (("tts_engine",), "from-bridge")],
)
def test_a_session_stream_leaves_through_the_bridge_only_at_the_last_stage(
    monkeypatch, stages, expected
):
    monkeypatch.setattr(omni_scheduler, "_get_active_stage", lambda: "tts_engine")
    scheduler, sched_output, mr_output = scheduler_with_unit(stages)
    OmniScheduler.emit_stream_output(scheduler, sched_output, mr_output)
    assert scheduler.outbox.get_nowait() == OutgoingMessage(
        "unit-1", "stream", expected
    )
    assert scheduler.outbox.empty()
    OmniScheduler.flush_stream_output(scheduler, "unit-1", "data")
    flushed = scheduler.outbox.get_nowait()
    if expected == "from-bridge":
        assert flushed == OutgoingMessage("unit-1", "stream", "from-bridge")
    else:
        assert flushed == OutgoingMessage("unit-1", "stream", "flushed-builder")
    assert scheduler.outbox.empty()
