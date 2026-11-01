# SPDX-License-Identifier: Apache-2.0
"""Turn-based interaction: negotiation, projection and hot updates."""

from __future__ import annotations

import asyncio
import math
import threading
from dataclasses import replace
from typing import Literal

import numpy as np
import pytest

from sglang_omni.audio.vad import VADConfig
from sglang_omni.models.vad.session import SileroSpeechDetector, VADHooks
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.proto.streams import AUDIO, PCM16, SPEECH_STARTED, SPEECH_STOPPED
from sglang_omni.scheduling.session import SessionContext
from sglang_omni.serve.realtime.control import ImageAccepted, UnitCompleted
from sglang_omni.serve.realtime.negotiation import SessionNegotiation
from sglang_omni.serve.realtime.output import (
    InputTranscription,
    ResponseHeard,
    ResponseProgress,
    SpeechStarted,
    SpeechStopped,
)
from sglang_omni.serve.realtime.projection import project_output
from sglang_omni.serve.realtime.runtime import SessionRuntime
from sglang_omni.serve.realtime.schema import SessionConfiguration
from sglang_omni.serve.realtime.session_bridge import capabilities_of
from sglang_omni.serve.realtime.types import (
    Capabilities,
    InteractionAdapter,
    OutputSink,
    ProtocolError,
    RuntimeLimits,
    Unit,
)
from tests.unit_test.fixtures.realtime_fakes import ScriptedAdapter
from tests.unit_test.fixtures.session_client import PORTS

TURN_BASED = capabilities_of(PORTS)
JPEG = b"\xff\xd8frame"


def test_capabilities_are_derived_from_the_ports():
    assert TURN_BASED == Capabilities(
        interaction="turn_based",
        input_sample_rate_hz=16000,
        output_sample_rate_hz=24000,
        output_modalities=("text", "audio"),
        max_granted_modalities=2,
        turn_control=("server_vad", None),
        native_unit_ms=200,
        tail_policy="flush",
        sampling_parameters=("temperature", "max_new_tokens_per_unit"),
        supports_instruction_updates=True,
        supports_sampling_updates=True,
    )
    streaming = capabilities_of(
        replace(PORTS, output=None, hot_settings=frozenset(), turn_detection_types=())
    )
    assert streaming.output_modalities == ("text",)
    assert (streaming.max_granted_modalities, streaming.turn_control) == (1, (None,))
    assert streaming.supports_instruction_updates is False
    assert streaming.supports_sampling_updates is False


def negotiate(capabilities: Capabilities, patch: dict, current=None, state="CREATED"):
    negotiation = SessionNegotiation(
        model="m", capabilities=capabilities, limits=RuntimeLimits()
    )
    return negotiation.negotiate(current or {}, state, patch)


def test_turn_based_grants_server_vad_and_native_keeps_its_contract():
    config, granted = negotiate(
        TURN_BASED,
        {
            "output_modalities": ["text", "audio"],
            "audio": {"input": {"turn_detection": {"silence_duration_ms": 800}}},
        },
    )
    assert granted["interaction"] == "turn_based"
    assert granted["native_full_duplex"] is False
    assert granted["turn_control"] == ["server_vad", None]
    assert granted["output_modalities"] == ["text", "audio"]
    assert granted["rejections"] == []
    assert config["audio"]["input"]["turn_detection"] == {"silence_duration_ms": 800}
    with pytest.raises(ProtocolError) as info:
        negotiate(
            TURN_BASED,
            {"audio": {"input": {"turn_detection": {"type": "semantic_vad"}}}},
        )
    assert info.value.code == "not_applicable"
    assert info.value.param == "session.audio.input.turn_detection.type"
    native = Capabilities(output_modalities=("text", "audio"))
    _, granted = negotiate(native, {"output_modalities": ["text", "audio"]})
    assert granted["output_modalities"] == ["text"]
    assert granted["turn_control"] == [None]
    with pytest.raises(ValueError, match="unsupported interaction"):
        Capabilities(interaction="pipelined")  # type: ignore[arg-type]


def test_the_ports_hot_settings_decide_which_fields_a_later_update_may_change():
    opening = {
        "instructions": "Be brief.",
        "sglang": {"sampling": {"temperature": 0.7}},
    }
    current, granted = negotiate(TURN_BASED, opening)
    assert granted["supports_instruction_updates"] is True
    assert granted["supports_sampling_updates"] is True
    config, _ = negotiate(
        TURN_BASED,
        {
            "instructions": "Be verbose.",
            "audio": {"input": {"turn_detection": {"type": "server_vad"}}},
            "sglang": {"sampling": {"temperature": 0.2}},
        },
        current=current,
        state="OPEN",
    )
    assert config["audio"]["input"]["turn_detection"] == {"type": "server_vad"}
    assert config["instructions"] == "Be verbose."
    assert config["sglang"]["sampling"] == {"temperature": 0.2}
    sampling_only = capabilities_of(
        replace(PORTS, hot_settings=frozenset({"sampling"}))
    )
    current, granted = negotiate(sampling_only, opening)
    assert granted["supports_instruction_updates"] is False
    assert granted["supports_sampling_updates"] is True
    config, _ = negotiate(
        sampling_only,
        {"sglang": {"sampling": {"temperature": 0.2}}},
        current=current,
        state="OPEN",
    )
    assert config["sglang"]["sampling"] == {"temperature": 0.2}
    with pytest.raises(ProtocolError, match="frozen"):
        negotiate(
            sampling_only,
            {"instructions": "Be verbose."},
            current=current,
            state="OPEN",
        )
    with pytest.raises(ProtocolError, match="unsupported sampling"):
        negotiate(
            sampling_only,
            {"sglang": {"sampling": {"top_k": 3}}},
            current=current,
            state="OPEN",
        )
    frozen = capabilities_of(replace(PORTS, output=None, hot_settings=frozenset()))
    current, granted = negotiate(frozen, opening)
    assert granted["supports_instruction_updates"] is False
    assert granted["supports_sampling_updates"] is False
    for patch in (
        {"instructions": "Be verbose."},
        {"sglang": {"sampling": {"temperature": 0.2}}},
    ):
        with pytest.raises(ProtocolError, match="frozen"):
            negotiate(frozen, patch, current=current, state="OPEN")


def test_speech_and_transcription_events_project_to_openai_names():
    assert project_output(SpeechStarted("item_user_0", 10.0)) == {
        "type": "input_audio_buffer.speech_started",
        "item_id": "item_user_0",
        "audio_start_ms": 10.0,
    }
    assert project_output(SpeechStopped("item_user_0", 100.0)) == {
        "type": "input_audio_buffer.speech_stopped",
        "item_id": "item_user_0",
        "audio_end_ms": 100.0,
    }
    assert project_output(InputTranscription("item_user_0", "hi", False)) == {
        "type": "conversation.item.input_audio_transcription.delta",
        "item_id": "item_user_0",
        "content_index": 0,
        "delta": "hi",
    }
    assert project_output(InputTranscription("item_user_0", "hi there", True)) == {
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "item_user_0",
        "content_index": 0,
        "transcript": "hi there",
    }


def test_reply_progress_projects_to_a_response_event():
    assert project_output(ResponseProgress("resp_0", "item_0", 4, 500.0)) == {
        "type": "sglang.response.progress",
        "response_id": "resp_0",
        "item_id": "item_0",
        "output_index": 0,
        "content_index": 0,
        "text_end": 4,
        "audio_ms": 500.0,
    }


def test_reply_heard_projects_to_a_response_event():
    assert project_output(ResponseHeard("resp_0", "item_0", 3)) == {
        "type": "sglang.response.heard",
        "response_id": "resp_0",
        "item_id": "item_0",
        "output_index": 0,
        "content_index": 0,
        "text_end": 3,
    }


class PlainAdapter(InteractionAdapter):
    async def open(
        self, session_id: str, config: SessionConfiguration, emit: OutputSink
    ) -> None:
        pass

    async def process(self, unit: Unit) -> int:
        return unit.real_samples

    async def close(self) -> None:
        pass


class CommittingAdapter(PlainAdapter):
    def __init__(self) -> None:
        self.events: list[tuple] = []

    async def process(self, unit: Unit) -> int:
        self.events.append(("unit", unit.real_samples))
        return unit.real_samples

    async def commit(self) -> None:
        self.events.append(("commit",))


class HeldSpanAdapter(CommittingAdapter):
    """Holds each processed span until the test releases it."""

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()

    async def process(self, unit: Unit) -> int:
        await self.release.wait()
        return await super().process(unit)


def unit_receipts(runtime: SessionRuntime) -> list[tuple[str, int, int]]:
    return [
        (envelope.unit.unit_id, envelope.unit.start_sample, envelope.unit.real_samples)
        for envelope in iter(runtime.output_buffer.dequeue, None)
        if isinstance(envelope.event, UnitCompleted)
    ]


@pytest.mark.asyncio
async def test_a_client_commit_follows_the_audio_appended_before_it():
    adapter = CommittingAdapter()
    manual = replace(TURN_BASED, supports_client_commit=True)
    runtime = SessionRuntime("m", manual, lambda: adapter, RuntimeLimits())
    await runtime.update({"audio": {"input": {"turn_detection": None}}}, "e1")
    half_unit = manual.native_unit_bytes // 2
    await runtime.append(b"\0" * half_unit, 0, 0.0, "e2")
    await runtime.commit()
    for _ in range(100):
        if ("commit",) in adapter.events:
            break
        else:
            await asyncio.sleep(0.01)
    assert adapter.events == [("unit", half_unit // 2), ("commit",)]
    await runtime.close("client_closed")
    vad_only = SessionRuntime("m", TURN_BASED, lambda: PlainAdapter(), RuntimeLimits())
    await vad_only.update({"instructions": "Be brief."}, "e1")
    with pytest.raises(ProtocolError, match="not granted"):
        await vad_only.commit()
    await vad_only.close("client_closed")


@pytest.mark.asyncio
async def test_an_adapter_without_cancel_reports_not_supported():
    runtime = SessionRuntime("m", TURN_BASED, lambda: PlainAdapter(), RuntimeLimits())
    await runtime.update({"instructions": "Be brief."}, "e1")
    with pytest.raises(ProtocolError, match="not cancel"):
        await runtime.cancel()
    with pytest.raises(ProtocolError, match="not truncate"):
        await runtime.truncate("item_0", 10.0)
    await runtime.close("client_closed")


class LoudFrameDetector(SileroSpeechDetector):
    """The VAD stage's detector with any non-silent frame as speech, in place of Silero's probability."""

    def infer(self, frame: np.ndarray) -> float:
        return float(np.abs(frame).max() > 0)


class VADAdapter(PlainAdapter):
    """Runs each processed span through the VAD stage, keeping the boundaries it emits and each span's sample range."""

    def __init__(self) -> None:
        self.hooks = VADHooks(VADConfig(silence_duration_ms=800), LoudFrameDetector)
        self.identity = SessionIdentity("vad")
        self.boundaries: list[TimedChunk] = []
        self.spans: list[tuple[int, int]] = []
        self.context = SessionContext(
            session_identity=self.identity,
            cancelled=threading.Event(),
            emit=self.boundaries.append,
        )

    async def open(
        self, session_id: str, config: SessionConfiguration, emit: OutputSink
    ) -> None:
        self.hooks.open(self.identity, OmniRequest(None))

    async def process(self, unit: Unit) -> int:
        self.spans.append((unit.start_sample, unit.real_samples))
        self.hooks.append(
            TimedChunk(
                AUDIO,
                unit.start_sample * 1000 / 16000,
                unit.real_samples * 1000 / 16000,
                unit.index,
                unit.pcm[: unit.real_samples * 2],
                format=PCM16,
            ),
            StagePayload("vad", OmniRequest(None), None),
            self.context,
        )
        return unit.real_samples


@pytest.mark.asyncio
async def test_server_vad_ends_the_turn_at_the_client_packet_that_completes_the_silence():
    """Audio reaches the session as each 80 ms packet arrives, while receipts stay one per 200 ms unit."""
    packet_ms, speech_ms, silence_ms = 80, 1024, 800
    pcm = b"\x10\x27" * (16 * speech_ms) + b"\0" * (2 * 16 * 1376)
    adapter = VADAdapter()
    runtime = SessionRuntime("m", TURN_BASED, lambda: adapter, RuntimeLimits())
    await runtime.update({"instructions": "Be brief."}, "e0")
    stopped_after_packet = None
    for packet_index, offset in enumerate(range(0, len(pcm), 2 * 16 * packet_ms)):
        await runtime.append(
            pcm[offset : offset + 2 * 16 * packet_ms],
            packet_index,
            float(packet_index * packet_ms),
            f"e{packet_index + 1}",
        )
        for _ in range(10):
            await asyncio.sleep(0)
        if stopped_after_packet is None and any(
            chunk.payload["type"] == SPEECH_STOPPED for chunk in adapter.boundaries
        ):
            stopped_after_packet = packet_index
        else:
            pass
    assert [(c.payload["type"], c.t_start_ms) for c in adapter.boundaries] == [
        (SPEECH_STARTED, 0.0),
        (SPEECH_STOPPED, float(speech_ms)),
    ]
    assert stopped_after_packet == math.ceil((speech_ms + silence_ms) / packet_ms) - 1
    span_ends = [
        start_sample + sample_count for start_sample, sample_count in adapter.spans
    ]
    assert [start_sample for start_sample, _ in adapter.spans] == [0, *span_ends[:-1]]
    assert unit_receipts(runtime) == [(f"unit_{i}", 3200 * i, 3200) for i in range(12)]
    await runtime.close("client_closed")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("is_span_in_flight", "receipts_at_clear"),
    [(False, [("unit_0", 0, 1280)]), (True, [])],
    ids=["idle", "span-in-flight"],
)
async def test_a_clear_ends_the_unit_whose_audio_already_reached_the_session(
    is_span_in_flight: bool, receipts_at_clear: list[tuple[str, int, int]]
):
    adapter = HeldSpanAdapter()
    if not is_span_in_flight:
        adapter.release.set()
    else:
        pass
    runtime = SessionRuntime("m", TURN_BASED, lambda: adapter, RuntimeLimits())
    await runtime.update({"instructions": "Be brief."}, "e0")
    await runtime.append(b"\1\0" * 1280, 0, 0.0, "e1")
    for _ in range(10):
        await asyncio.sleep(0)
    await runtime.clear("e2")
    receipts = unit_receipts(runtime)
    assert receipts == receipts_at_clear
    adapter.release.set()
    await runtime.append(b"\1\0" * 3200, 1, 80.0, "e3")
    for _ in range(10):
        await asyncio.sleep(0)
    receipts += unit_receipts(runtime)
    assert adapter.events == [("unit", 1280), ("unit", 3200)]
    assert receipts == [
        ("unit_0", 0, 1280),
        ("unit_1", 1280, 3200),
    ]
    await runtime.close("client_closed")


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["commit", "end"])
async def test_a_commit_or_the_input_end_completes_the_unit_whose_audio_already_reached_the_session(
    command: Literal["commit", "end"],
):
    manual = replace(TURN_BASED, supports_client_commit=True)
    runtime = SessionRuntime("m", manual, lambda: CommittingAdapter(), RuntimeLimits())
    await runtime.update({"audio": {"input": {"turn_detection": None}}}, "e0")
    await runtime.append(b"\1\0" * 1280, 0, 0.0, "e1")
    for _ in range(10):
        await asyncio.sleep(0)
    if command == "commit":
        await runtime.commit()
    else:
        await runtime.end("e2")
    for _ in range(10):
        await asyncio.sleep(0)
    assert unit_receipts(runtime) == [("unit_0", 0, 1280)]
    await runtime.close("client_closed")


def accepted_image_units(runtime: SessionRuntime) -> list[str]:
    return [
        envelope.event.unit_id
        for envelope in iter(runtime.output_buffer.dequeue, None)
        if isinstance(envelope.event, ImageAccepted)
    ]


@pytest.mark.asyncio
async def test_an_image_binds_to_its_unit_after_part_of_that_unit_reached_the_session():
    with_images = replace(TURN_BASED, input_modalities=("audio", "image"))
    adapter = ScriptedAdapter()
    runtime = SessionRuntime("m", with_images, lambda: adapter, RuntimeLimits())
    await runtime.update({"instructions": "Be brief."}, "e0")
    await runtime.append(b"\1\0" * 1280, 0, 0.0, "e1")
    for _ in range(10):
        await asyncio.sleep(0)
    await runtime.append_image(JPEG, 210.0, "e2")
    assert accepted_image_units(runtime) == ["unit_1"]
    await runtime.append_image(JPEG, 40.0, "e3")
    assert accepted_image_units(runtime) == ["unit_0"]
    await runtime.append(b"\1\0" * 3200, 1, 80.0, "e4")
    for _ in range(10):
        await asyncio.sleep(0)
    assert [(unit.index, unit.images) for unit in adapter.units] == [
        (0, ()),
        (0, (JPEG,)),
        (1, (JPEG,)),
    ]
    await runtime.close("client_closed")
