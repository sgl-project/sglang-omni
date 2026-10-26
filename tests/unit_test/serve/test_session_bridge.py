# SPDX-License-Identifier: Apache-2.0
"""SessionBridge: units become appends and Session outputs become realtime events."""

import asyncio
from dataclasses import replace

import numpy as np
import pytest

from sglang_omni.admission import QueueFullError
from sglang_omni.models.qwen3_omni.session_node import SESSION_PORTS as QWEN3_OMNI_PORTS
from sglang_omni.proto.streams import (
    AUDIO,
    COMMIT,
    CONFIGURE,
    CONFIGURED,
    CONTROL,
    IMAGE,
    IMAGE_FRAME,
    INTERRUPT,
    PCM16,
    PCM_F32,
    REPLY_HEARD,
    REPLY_PROGRESS,
    SESSION_CONTROL,
    SPEECH_STARTED,
    SPEECH_STOPPED,
    TEXT,
    TEXT_DELTA,
    TRANSCRIPT,
    TRANSCRIPT_FORMAT,
    TRUNCATE,
    TURN_DONE,
    TURN_EMPTY,
    TURN_INTERRUPTED,
    TURN_QUEUED,
    TURN_STARTED,
    AudioPort,
    ImagePort,
    SessionPorts,
)
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    AudioFinished,
    InputTranscription,
    ResponseFinished,
    ResponseHeard,
    ResponseProgress,
    ResponseStarted,
    SpeechStarted,
    SpeechStopped,
    TextDelta,
    TextFinished,
    TurnFailure,
)
from sglang_omni.serve.realtime.runtime import SessionRuntime
from sglang_omni.serve.realtime.session_bridge import SessionBridge, capabilities_of
from sglang_omni.serve.realtime.types import ProtocolError, RuntimeLimits, Unit
from tests.unit_test.fixtures.session_client import PORTS, FakeSessionClient

RATE = 16000
UNIT_SAMPLES = 3200
CONFIG = {"audio": {"input": {"format": {"type": "audio/pcm", "rate": RATE}}}}


class Harness:
    def __init__(self, ports=PORTS, *, auto_done=True, limits=None) -> None:
        self.client = FakeSessionClient(auto_done=auto_done)
        self.events = []
        self.bridge = SessionBridge(self.client, ports, limits=limits)

    async def sink(self, event, unit=None):
        self.events.append((event, unit))

    @property
    def published(self):
        return [event for event, _ in self.events]

    async def open(self, config=CONFIG):
        await self.bridge.open("rt", config, self.sink)


def unit(index, *, eos=False, samples=UNIT_SAMPLES):
    pcm = np.full(samples, index + 1, dtype="<i2").tobytes()
    return Unit(index, index * UNIT_SAMPLES, pcm, samples, eos)


async def settle():
    await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_units_are_appended_one_per_unit_with_contiguous_seq_and_timing():
    h = Harness()
    await h.open(
        {
            "instructions": "Be brief.",
            "audio": {
                "input": {
                    **CONFIG["audio"]["input"],
                    "turn_detection": {"silence_duration_ms": 500},
                }
            },
            "sglang": {"sampling": {"temperature": 0.2}},
        }
    )
    request, _, session_id = h.client.opened[0]
    assert request.params == {
        "instructions": "Be brief.",
        "sampling": {"temperature": 0.2},
        "turn_detection": {"silence_duration_ms": 500},
    }
    assert session_id == "rt"
    for index in range(2):
        assert await h.bridge.process(unit(index)) == UNIT_SAMPLES
    assert await h.bridge.process(unit(2, eos=True, samples=800)) == 800
    assert [
        (c.modality, c.t_start_ms, c.duration_ms, c.seq, len(c.payload), c.format)
        for c in h.client.appended
    ] == [
        (AUDIO, 0.0, 200.0, 0, 6400, PCM16),
        (AUDIO, 200.0, 200.0, 1, 6400, PCM16),
        (AUDIO, 400.0, 50.0, 2, 1600, PCM16),
    ]
    assert [c.eos for c in h.client.appended] == [False, False, True]
    await h.bridge.close()
    assert h.client.closed.is_set()


@pytest.mark.asyncio
async def test_outputs_stream_before_input_done():
    h = Harness(auto_done=False)
    await h.open()
    processing = asyncio.create_task(h.bridge.process(unit(0)))
    await settle()
    h.client.control(SPEECH_STARTED, t_ms=40.0)
    h.client.emit(TEXT, TEXT_DELTA, {"text": "Hi"})
    await settle()
    assert not processing.done()
    assert h.events == [
        (SpeechStarted("item_user_0", 40.0), None),
        (ResponseStarted("resp_0"), None),
        (TextDelta("resp_0", "item_0", "Hi"), None),
    ]
    h.client.done(0)
    assert await processing == UNIT_SAMPLES
    await h.bridge.close()


@pytest.mark.asyncio
async def test_every_vocabulary_item_projects_through_the_table():
    h = Harness()
    await h.open()
    pcm_f32 = np.array([0.5, -1.5], dtype="<f4").tobytes()
    h.client.control(SPEECH_STARTED, t_ms=10.0)
    h.client.emit(TRANSCRIPT, TRANSCRIPT_FORMAT, {"text": "hi", "is_final": False})
    h.client.control(SPEECH_STOPPED, t_ms=900.0)
    h.client.emit(TRANSCRIPT, TRANSCRIPT_FORMAT, {"text": "hi you", "is_final": True})
    h.client.control(TURN_QUEUED, turn=0)
    h.client.control(TURN_STARTED, turn=0)
    h.client.emit(TEXT, TEXT_DELTA, {"text": "Hello "})
    h.client.emit(AUDIO, PCM_F32, pcm_f32)
    h.client.emit(TEXT, TEXT_DELTA, {"text": "there."})
    h.client.control(TURN_DONE)
    h.client.control(CONFIGURED, stages={"llm": "openai_chat"})
    h.client.emit(TRANSCRIPT, TRANSCRIPT_FORMAT, {"text": "", "is_final": True})
    h.client.control(TURN_EMPTY)
    await settle()
    assert h.published == [
        SpeechStarted("item_user_0", 10.0),
        InputTranscription("item_user_0", "hi", False),
        SpeechStopped("item_user_0", 900.0),
        InputTranscription("item_user_0", "hi you", True),
        ResponseStarted("resp_0"),
        TextDelta("resp_0", "item_0", "Hello "),
        AudioDelta(
            "resp_0", "item_0", np.array([16383, -32767], dtype="<i2").tobytes()
        ),
        TextDelta("resp_0", "item_0", "there."),
        TextFinished("resp_0", "item_0", "Hello there."),
        AudioFinished("resp_0", "item_0"),
        ResponseFinished("resp_0", "item_0", "Hello there.", True, "completed", "stop"),
        InputTranscription("item_user_1", "", True),
        ResponseStarted("resp_1"),
        ResponseFinished("resp_1", "item_1", "", False, "completed", "empty_turn"),
    ]
    h.client.emit(TEXT, "text", {"text": "whole"})
    await settle()
    assert isinstance(h.published[-1], TurnFailure)
    assert "unsupported session output" in h.published[-1].message
    await h.bridge.close()


@pytest.mark.asyncio
async def test_a_reply_the_talker_cut_at_its_token_budget_ends_incomplete():
    """Reaching the talker's token budget ends the response as incomplete instead of reporting a completed one."""
    h = Harness()
    await h.open()
    h.client.control(TURN_STARTED, turn=0)
    h.client.emit(TEXT, TEXT_DELTA, {"text": "Half a sen"})
    h.client.control(TURN_DONE, reply_truncated=True)
    await settle()
    assert [event for event in h.published if isinstance(event, ResponseFinished)] == [
        ResponseFinished(
            "resp_0", "item_0", "Half a sen", False, "incomplete", "max_output_tokens"
        )
    ]
    await h.bridge.close()


@pytest.mark.asyncio
async def test_reply_progress_maps_to_the_open_response_of_its_turn():
    h = Harness()
    await h.open()
    h.client.control(TURN_STARTED, turn=3)
    h.client.emit(TEXT, TEXT_DELTA, {"text": "One, two."}, turn=3)
    h.client.control(REPLY_PROGRESS, turn=3, text_end=4, audio_ms=500.0)
    h.client.control(REPLY_PROGRESS, turn=2, text_end=9, audio_ms=900.0)
    h.client.control(TURN_DONE, turn=3)
    h.client.control(REPLY_PROGRESS, turn=3, text_end=9, audio_ms=1000.0)
    await settle()
    assert [event for event in h.published if isinstance(event, ResponseProgress)] == [
        ResponseProgress("resp_0", "item_0", 4, 500.0)
    ]
    await h.bridge.close()


@pytest.mark.asyncio
async def test_reply_heard_maps_to_the_ended_response_of_its_turn():
    h = Harness()
    await h.open()
    h.client.control(TURN_STARTED, turn=3)
    h.client.emit(TEXT, TEXT_DELTA, {"text": "One, two."}, turn=3)
    h.client.control(REPLY_HEARD, turn=3, text_end=4)
    h.client.control(TURN_INTERRUPTED, turn=3)
    h.client.control(REPLY_HEARD, turn=3, text_end=5)
    h.client.control(REPLY_HEARD, turn=2, text_end=9)
    h.client.control(TURN_EMPTY, turn=4)
    h.client.control(REPLY_HEARD, turn=3, text_end=6)
    h.client.control(TURN_STARTED, turn=5)
    h.client.control(REPLY_HEARD, turn=3, text_end=9)
    await settle()
    assert [event for event in h.published if isinstance(event, ResponseHeard)] == [
        ResponseHeard("resp_0", "item_0", 5),
        ResponseHeard("resp_0", "item_0", 6),
    ]
    await h.bridge.close()


@pytest.mark.asyncio
async def test_a_truncate_names_the_turn_of_its_item():
    """A truncate reached the session without its item, so the dialog cut the reply it guessed; an item the bridge never sent is an error and appends nothing."""
    h = Harness()
    await h.open()
    h.client.control(TURN_STARTED, turn=3)
    h.client.control(TURN_INTERRUPTED, turn=3)
    h.client.control(TURN_STARTED, turn=5)
    await settle()
    await h.bridge.truncate("item_0", 400.0)
    await h.bridge.truncate("item_1", 200.0)
    with pytest.raises(ProtocolError, match="not found") as error:
        await h.bridge.truncate("item_7", 100.0)
    assert error.value.code == "item_not_found"
    assert [(chunk.payload, chunk.turn) for chunk in h.client.appended] == [
        ({"type": TRUNCATE, "t_ms": 400.0}, 3),
        ({"type": TRUNCATE, "t_ms": 200.0}, 5),
    ]
    await h.bridge.close()


@pytest.mark.asyncio
async def test_runtime_updates_cancel_and_truncate_reach_the_session_client():
    ports = replace(PORTS, controls=frozenset({COMMIT, CONFIGURE, TRUNCATE}))
    assert capabilities_of(ports).supports_truncate
    assert not capabilities_of(PORTS).supports_truncate
    h = Harness(ports)
    runtime = SessionRuntime(
        "m", capabilities_of(ports), lambda: h.bridge, RuntimeLimits()
    )
    try:
        initial = {**CONFIG, "instructions": "Be brief."}
        await runtime.update(initial, "open")
        assert h.client.opened[0][0].params["instructions"] == "Be brief."
        assert h.client.appended == []
        await h.bridge.process(unit(0))
        await runtime.update(initial, "unchanged")
        assert len(h.client.appended) == 1
        updated = {
            "instructions": "Be verbose.",
            "audio": {"input": {"turn_detection": {"silence_duration_ms": 800}}},
        }
        await runtime.update(updated, "changed")
        await runtime.update(updated, "repeated")
        configure = h.client.appended[1:]
        assert len(configure) == 1
        assert (configure[0].modality, configure[0].format) == (
            CONTROL,
            SESSION_CONTROL,
        )
        assert (configure[0].t_start_ms, configure[0].seq) == (200.0, 1)
        assert configure[0].payload == {
            "type": CONFIGURE,
            "instructions": "Be verbose.",
            "turn_detection": {"silence_duration_ms": 800},
        }
        sampling = {"sglang": {"sampling": {"temperature": 0.7}}}
        await runtime.update(sampling, "sampling")
        await runtime.update(sampling, "sampling repeated")
        assert len(h.client.appended) == 3
        assert h.client.appended[-1].payload == {
            "type": CONFIGURE,
            "sampling": {"temperature": 0.7},
        }
        await runtime.cancel()
        assert len(h.client.appended) == 3
        event, stages, should_preempt = h.client.controlled[-1]
        assert event.format == SESSION_CONTROL and event.payload == {"type": INTERRUPT}
        assert event.t_start_ms == 200.0
        assert stages is None and should_preempt
        h.client.control(TURN_STARTED, turn=1)
        await settle()
        await runtime.truncate("item_0", 1234.5)
        truncate = h.client.appended[-1]
        assert (truncate.payload, truncate.turn) == (
            {"type": TRUNCATE, "t_ms": 1234.5},
            1,
        )
        assert await h.bridge.clear() == 0
    finally:
        await runtime.close("client_closed")
    assert h.client.closed.is_set()


@pytest.mark.asyncio
async def test_reader_failure_reports_turn_failure_and_fails_the_next_unit():
    h = Harness()
    await h.open()
    h.client.outputs.put_nowait(None)
    await settle()
    assert isinstance(h.published[-1], TurnFailure)
    assert "output stream closed" in h.published[-1].message
    with pytest.raises(RuntimeError, match="output stream closed"):
        await h.bridge.process(unit(0))
    await h.bridge.close()


@pytest.mark.asyncio
async def test_rate_mismatch_and_unsupported_audio_format_are_rejected():
    h = Harness()
    with pytest.raises(ValueError, match="sample rate"):
        await h.open(
            {"audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}}}}
        )
    with pytest.raises(ValueError, match="pcm16 or pcm_f32le"):
        SessionBridge(
            FakeSessionClient(), replace(PORTS, input=(AudioPort("opus", RATE),))
        )


@pytest.mark.asyncio
async def test_a_rejected_configure_is_sent_again_when_the_update_is_retried():
    h = Harness()
    await h.open()
    append_session = h.client.append_session

    async def reject_once(session_identity, chunk):
        h.client.append_session = append_session
        raise QueueFullError()

    h.client.append_session = reject_once
    updated = {**CONFIG, "instructions": "Be verbose."}
    with pytest.raises(QueueFullError):
        await h.bridge.update(updated)
    assert h.client.appended == []
    await h.bridge.update(updated)
    assert [chunk.payload for chunk in h.client.appended] == [
        {"type": CONFIGURE, "instructions": "Be verbose."}
    ]
    await h.bridge.close()


@pytest.mark.asyncio
async def test_a_client_commit_ends_the_utterance_on_the_session():
    ports = replace(PORTS, controls=frozenset({COMMIT}))
    assert capabilities_of(ports).supports_client_commit is True
    assert (
        capabilities_of(replace(PORTS, controls=frozenset())).supports_client_commit
        is False
    )
    h = Harness(ports)
    await h.open()
    await h.bridge.process(unit(0))
    await h.bridge.commit()
    assert [
        chunk.payload for chunk in h.client.appended if chunk.format == SESSION_CONTROL
    ] == [{"type": COMMIT}]
    await h.bridge.close()


@pytest.mark.asyncio
async def test_a_null_turn_detection_reaches_the_session_as_manual_turns():
    assert capabilities_of(PORTS).turn_control == ("server_vad", None)
    manual = {"audio": {"input": {**CONFIG["audio"]["input"], "turn_detection": None}}}
    server_vad = {"audio": {"input": {"turn_detection": {"type": "server_vad"}}}}
    h = Harness()
    runtime = SessionRuntime(
        "m", capabilities_of(PORTS), lambda: h.bridge, RuntimeLimits()
    )
    try:
        await runtime.update(manual, "open")
        assert h.client.opened[0][0].params == {"turn_detection": None}
        await runtime.update(server_vad, "vad")
        await runtime.update(manual, "manual")
        assert [chunk.payload for chunk in h.client.appended] == [
            {"type": CONFIGURE, "turn_detection": {"type": "server_vad"}},
            {"type": CONFIGURE, "turn_detection": None},
        ]
    finally:
        await runtime.close("client_closed")
    h = Harness()
    await h.open()
    assert "turn_detection" not in h.client.opened[0][0].params
    await h.bridge.update(manual)
    assert [chunk.payload for chunk in h.client.appended] == [
        {"type": CONFIGURE, "turn_detection": None}
    ]
    await h.bridge.close()


def test_an_image_port_grants_image_input_and_audio_only_ports_do_not():
    assert PORTS.input_modalities == ("audio",)
    assert capabilities_of(PORTS).input_modalities == ("audio",)
    assert "input_image_format" not in capabilities_of(PORTS).to_granted_capabilities()
    seeing = replace(PORTS, input=(*PORTS.input, ImagePort(IMAGE_FRAME)))
    assert capabilities_of(seeing).input_modalities == ("audio", "image")
    granted = capabilities_of(seeing).to_granted_capabilities()
    assert granted["input_modalities"] == ["audio", "image"]
    assert granted["input_image_format"]["max_frames_per_unit"] == 1
    assert QWEN3_OMNI_PORTS.input_modalities == ("audio", "image")
    with pytest.raises(ValueError, match="one audio input port"):
        replace(PORTS, input=(ImagePort(IMAGE_FRAME),))
    with pytest.raises(ValueError, match="bridged images must be image_frame"):
        SessionBridge(
            FakeSessionClient(), replace(PORTS, input=(*PORTS.input, ImagePort("x")))
        )


@pytest.mark.asyncio
async def test_a_unit_s_camera_frames_are_appended_before_its_audio_at_its_time():
    h = Harness(replace(PORTS, input=(*PORTS.input, ImagePort(IMAGE_FRAME))))
    await h.open()
    await h.bridge.process(unit(0))
    frames = (b"\xff\xd8one", b"\x89PNGtwo")
    await h.bridge.process(replace(unit(1), images=frames))
    assert [
        (c.modality, c.format, c.t_start_ms, c.duration_ms, c.seq)
        for c in h.client.appended
    ] == [
        (AUDIO, PCM16, 0.0, 200.0, 0),
        (IMAGE, IMAGE_FRAME, 200.0, 0, 1),
        (IMAGE, IMAGE_FRAME, 200.0, 0, 2),
        (AUDIO, PCM16, 200.0, 200.0, 3),
    ]
    assert tuple(c.payload for c in h.client.appended[1:3]) == frames
    await h.bridge.close()


@pytest.mark.asyncio
async def test_qwen3_omni_node_outputs_project_without_a_model_adapter():
    vad_only = SessionPorts(
        input=(AudioPort(PCM16, RATE),),
        output=None,
        unit_ms=200,
        controls=frozenset({COMMIT, CONFIGURE, INTERRUPT, TRUNCATE}),
        hot_settings=frozenset({"turn_detection"}),
        turn_detection_types=("server_vad",),
    )
    ports = vad_only.with_node(QWEN3_OMNI_PORTS)
    assert ports.output == AudioPort(PCM_F32, 24000)
    assert ports.hot_settings == {"sampling", "turn_detection"}
    assert ports.unit_ms == 200
    with pytest.raises(ValueError, match="share its audio input port"):
        vad_only.with_node(replace(QWEN3_OMNI_PORTS, input=(AudioPort(PCM16, 24000),)))
    assert capabilities_of(ports).output_sample_rate_hz == 24000
    assert capabilities_of(ports).supports_truncate
    h = Harness(ports)
    await h.open()
    samples = np.array([0.5, -0.5], dtype="<f4")
    pcm = (samples * 32767).astype("<i2").tobytes()
    h.client.control(TURN_STARTED, turn=1)
    h.client.emit(TEXT, TEXT_DELTA, {"text": "Hi"}, turn=1)
    h.client.emit(AUDIO, PCM_F32, samples.tobytes())
    h.client.control(TURN_DONE, turn=1)
    h.client.control(TURN_STARTED, turn=2)
    h.client.emit(TEXT, TEXT_DELTA, {"text": "Well"}, turn=2)
    h.client.control(TURN_INTERRUPTED, turn=2)
    await settle()
    assert h.published == [
        ResponseStarted("resp_0"),
        TextDelta("resp_0", "item_0", "Hi"),
        AudioDelta("resp_0", "item_0", pcm),
        TextFinished("resp_0", "item_0", "Hi"),
        AudioFinished("resp_0", "item_0"),
        ResponseFinished("resp_0", "item_0", "Hi", True, "completed", "stop"),
        ResponseStarted("resp_1"),
        TextDelta("resp_1", "item_1", "Well"),
        TextFinished("resp_1", "item_1", "Well"),
        ResponseFinished("resp_1", "item_1", "Well", False, "cancelled", "interrupted"),
    ]
    await h.bridge.close()
