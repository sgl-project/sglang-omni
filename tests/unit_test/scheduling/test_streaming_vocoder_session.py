# SPDX-License-Identifier: Apache-2.0
"""A Session's appends leave the vocoder as timed chunks: audio per unit, its reply progress, and controls in arrival order."""

from __future__ import annotations

import numpy as np
import torch

from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    SESSION_METADATA_KEY,
    SessionIdentity,
    SessionOperation,
    TimedChunk,
)
from sglang_omni.proto.streams import (
    AUDIO,
    PCM_F32,
    REPLY_PROGRESS,
    REPLY_TRUNCATED,
    SESSION_CONTROL,
    TEXT_DELTA,
    TURN_DONE,
    TURN_STARTED,
)
from sglang_omni.scheduling.message import IncomingMessage
from tests.unit_test.scheduling.test_streaming_vocoder import (
    SAMPLE_RATE,
    FakeStreamingVocoder,
    drain,
    item,
)

IDENTITY = SessionIdentity("tts")


def append_payload(request_id: str, chunk: TimedChunk) -> StagePayload:
    operation = SessionOperation("append", IDENTITY, ("tts_engine", "vocoder"), chunk)
    return StagePayload(
        request_id,
        OmniRequest(
            None,
            params={"stream": True},
            metadata={SESSION_METADATA_KEY: operation.to_dict()},
        ),
        {},
    )


def text_unit(request_id: str, text: str, seq: int) -> StagePayload:
    return append_payload(
        request_id,
        TimedChunk("text", 0, 0, seq, {"text": text}, format=TEXT_DELTA, turn=1),
    )


def control_unit(request_id: str, seq: int) -> StagePayload:
    return append_payload(
        request_id,
        TimedChunk(
            "control", 0, 0, seq, {"type": TURN_DONE}, format=SESSION_CONTROL, turn=1
        ),
    )


def chunks(messages) -> list[tuple[str, str, TimedChunk | None]]:
    return [
        (
            m.request_id,
            m.type,
            TimedChunk.from_dict(m.data) if m.type == "stream" else None,
        )
        for m in messages
    ]


def test_a_session_unit_streams_timed_audio_then_its_progress():
    vocoder = FakeStreamingVocoder()
    vocoder.handle_new_request_batch(
        [IncomingMessage("u1", "new_request", text_unit("u1", "hello there", 3))]
    )
    vocoder.handle_stream_chunk("u1", item([1, 2]))
    vocoder.handle_stream_chunk("u1", item([3]))
    vocoder.handle_stream_done("u1")
    out = chunks(drain(vocoder))
    audio = [c for _, kind, c in out if kind == "stream" and c.modality == AUDIO]
    assert [c.format for c in audio] == [PCM_F32, PCM_F32]
    assert [(c.seq, c.turn) for c in audio] == [(3, 1), (3, 1)]
    assert audio[0].t_start_ms == 0 and audio[1].t_start_ms == audio[0].duration_ms
    assert np.frombuffer(audio[0].payload, dtype="<f4").tolist() == [1.0, 2.0]
    progress = [
        c for _, kind, c in out if kind == "stream" and c.format == SESSION_CONTROL
    ]
    assert len(progress) == 1
    assert progress[0].payload == {
        "type": REPLY_PROGRESS,
        "text_end": len("hello there"),
        "audio_ms": 3 * 1000 / SAMPLE_RATE,
    }
    assert out[-1][1] == "result"


def test_an_appended_control_waits_for_the_units_before_it():
    vocoder = FakeStreamingVocoder()
    vocoder.handle_new_request_batch(
        [IncomingMessage("u1", "new_request", text_unit("u1", "one two", 1))]
    )
    vocoder.handle_stream_chunk("u1", item([1]))
    assert [c.format for _, _, c in chunks(drain(vocoder))] == [PCM_F32]
    vocoder.handle_new_request_batch(
        [IncomingMessage("u2", "new_request", control_unit("u2", 2))]
    )
    vocoder.handle_stream_done("u2")
    assert drain(vocoder) == []
    vocoder.handle_new_request_batch(
        [IncomingMessage("u3", "new_request", text_unit("u3", "three", 3))]
    )
    vocoder.handle_stream_chunk("u3", item([2]))
    vocoder.handle_stream_done("u3")
    vocoder.handle_stream_done("u1")
    out = chunks(drain(vocoder))
    order = [(rid, kind, c.format if c is not None else None) for rid, kind, c in out]
    assert order == [
        ("u3", "stream", PCM_F32),
        ("u3", "stream", SESSION_CONTROL),
        ("u3", "result", None),
        ("u1", "stream", SESSION_CONTROL),
        ("u1", "result", None),
        ("u2", "stream", SESSION_CONTROL),
        ("u2", "result", None),
    ]
    relayed = out[-2][2]
    assert relayed.payload == {"type": TURN_DONE} and relayed.turn == 1
    assert vocoder.session_units == {} and vocoder.pending_session_controls == {}


def test_a_plain_request_still_streams_waveform_payloads():
    vocoder = FakeStreamingVocoder()
    payload = StagePayload("r", OmniRequest("", params={"stream": True}), {})
    vocoder.handle_new_request_batch([IncomingMessage("r", "new_request", payload)])
    vocoder.handle_stream_chunk("r", item([1]))
    vocoder.handle_stream_done("r")
    stream = [m for m in drain(vocoder) if m.type == "stream"]
    assert stream and "audio_waveform" in stream[0].data


def test_a_session_control_is_answered_and_an_interrupt_drops_the_turn():
    vocoder = FakeStreamingVocoder()
    vocoder.handle_new_request_batch(
        [IncomingMessage("u1", "new_request", text_unit("u1", "one", 1))]
    )
    vocoder.handle_stream_chunk("u1", item([1]))
    drain(vocoder)
    interrupt = TimedChunk(
        "control", 0, 0, 0, {"type": "interrupt"}, format=SESSION_CONTROL, turn=1
    )
    operation = SessionOperation(
        "control", IDENTITY, ("tts_engine", "vocoder"), interrupt
    )
    control = StagePayload(
        "c1",
        OmniRequest(
            None,
            params={"stream": True},
            metadata={SESSION_METADATA_KEY: operation.to_dict()},
        ),
        {},
    )
    vocoder.handle_new_request_batch([IncomingMessage("c1", "new_request", control)])
    out = drain(vocoder)
    assert [(m.request_id, m.type) for m in out if m.type == "result"] == [
        ("c1", "result")
    ]
    assert out[-1].data.data == {"controlled": True}
    assert vocoder.is_aborted("u1") and "u1" not in vocoder.session_appends
    vocoder.handle_stream_done("u1")
    assert [m for m in drain(vocoder) if m.type in ("stream", "result")] == []


def operation_payload(request_id: str, kind: str) -> StagePayload:
    operation = SessionOperation(kind, IDENTITY, ("tts_engine", "vocoder"), None)
    return StagePayload(
        request_id,
        OmniRequest(
            None,
            params={"stream": True},
            metadata={SESSION_METADATA_KEY: operation.to_dict()},
        ),
        {},
    )


def test_open_is_answered_and_the_read_is_held_until_the_close():
    vocoder = FakeStreamingVocoder()
    vocoder.handle_new_request_batch(
        [
            IncomingMessage("o1", "new_request", operation_payload("o1", "open")),
            IncomingMessage("r1", "new_request", operation_payload("r1", "read")),
        ]
    )
    out = drain(vocoder)
    assert [(m.request_id, m.data.data) for m in out] == [("o1", {"opened": True})]
    vocoder.handle_new_request_batch(
        [IncomingMessage("u1", "new_request", text_unit("u1", "one", 1))]
    )
    vocoder.handle_stream_chunk("u1", item([1]))
    drain(vocoder)
    vocoder.handle_new_request_batch(
        [IncomingMessage("x1", "new_request", operation_payload("x1", "close"))]
    )
    out = drain(vocoder)
    assert [(m.request_id, m.data.data) for m in out if m.type == "result"] == [
        ("r1", {"closed": True}),
        ("x1", {"closed": True}),
    ]
    assert vocoder.is_aborted("u1")
    assert vocoder.session_units == {} and vocoder.session_reads == {}


def test_a_relayed_control_unit_streams_without_a_stream_param():
    vocoder = FakeStreamingVocoder()
    control = control_unit("c1", 1)
    control.request.params = {}
    vocoder.handle_new_request_batch([IncomingMessage("c1", "new_request", control)])
    vocoder.handle_stream_done("c1")
    out = chunks(drain(vocoder))
    assert [(request_id, kind) for request_id, kind, _ in out] == [
        ("c1", "stream"),
        ("c1", "result"),
    ]
    assert out[0][2].format == SESSION_CONTROL


REPLY_UNIT = "reply"


def reply_unit(request_id: str, seq: int) -> StagePayload:
    return append_payload(
        request_id, TimedChunk("control", 0, 0, seq, {}, format=REPLY_UNIT, turn=1)
    )


def test_a_declared_reply_unit_publishes_its_turn_around_relayed_text_and_audio():
    vocoder = FakeStreamingVocoder(reply_unit_format=REPLY_UNIT)
    vocoder.handle_new_request_batch(
        [IncomingMessage("r1", "new_request", reply_unit("r1", 5))]
    )
    text = TimedChunk("text", 0, 0, 5, {"text": "hi"}, format=TEXT_DELTA, turn=1)
    vocoder.handle_stream_chunk(
        "r1",
        StreamItem(
            0,
            torch.empty(0, dtype=torch.long),
            "talker_ar",
            {"modality": "text", "stream": True, "chunk": text.to_dict()},
        ),
    )
    vocoder.handle_stream_chunk("r1", item([1, 2]))
    vocoder.handle_stream_done("r1")
    out = chunks(drain(vocoder))
    kinds = [
        (
            (c.format, c.payload["type"] if c.format == SESSION_CONTROL else None)
            if c is not None
            else kind
        )
        for _, kind, c in out
    ]
    assert kinds == [
        (SESSION_CONTROL, TURN_STARTED),
        (TEXT_DELTA, None),
        (PCM_F32, None),
        (SESSION_CONTROL, REPLY_PROGRESS),
        (SESSION_CONTROL, TURN_DONE),
        "result",
    ]
    assert all(c is None or c.turn == 1 for _, _, c in out)
    assert out[1][2].payload == {"text": "hi"}


def test_a_reply_the_talker_cut_at_its_token_budget_reports_it_on_turn_done():
    """A talker that hits its token budget flags the reply, which ends its turn as truncated rather than as a completed one."""
    vocoder = FakeStreamingVocoder(reply_unit_format=REPLY_UNIT)
    vocoder.handle_new_request_batch(
        [IncomingMessage("r1", "new_request", reply_unit("r1", 5))]
    )
    vocoder.handle_stream_chunk(
        "r1",
        StreamItem(
            0,
            torch.empty(0, dtype=torch.long),
            "talker_ar",
            {"modality": "text", "stream": True, REPLY_TRUNCATED: True},
        ),
    )
    vocoder.handle_stream_done("r1")
    out = chunks(drain(vocoder))
    done = [
        c
        for _, _, c in out
        if c is not None
        and c.format == SESSION_CONTROL
        and c.payload["type"] == TURN_DONE
    ]
    assert [c.payload for c in done] == [{"type": TURN_DONE, REPLY_TRUNCATED: True}]
