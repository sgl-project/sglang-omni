# SPDX-License-Identifier: Apache-2.0
"""Turn-based interaction: negotiation, projection and hot updates."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

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
from tests.unit_test.fixtures.session_client import PORTS

TURN_BASED = capabilities_of(PORTS)


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
    )
    streaming = capabilities_of(
        replace(PORTS, output=None, hot_settings=frozenset(), turn_detection_types=())
    )
    assert streaming.output_modalities == ("text",)
    assert (streaming.max_granted_modalities, streaming.turn_control) == (1, (None,))
    assert streaming.supports_instruction_updates is False


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


def test_turn_detection_and_instructions_are_hot_updates():
    current, _ = negotiate(TURN_BASED, {"instructions": "Be brief."})
    config, _ = negotiate(
        TURN_BASED,
        {
            "instructions": "Be verbose.",
            "audio": {"input": {"turn_detection": {"type": "server_vad"}}},
        },
        current=current,
        state="OPEN",
    )
    assert config["audio"]["input"]["turn_detection"] == {"type": "server_vad"}
    assert config["instructions"] == "Be verbose."
    frozen = capabilities_of(replace(PORTS, output=None, hot_settings=frozenset()))
    current, _ = negotiate(frozen, {"instructions": "Be brief."})
    with pytest.raises(ProtocolError, match="frozen"):
        negotiate(
            frozen, {"instructions": "Be verbose."}, current=current, state="OPEN"
        )


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
        await runtime.truncate(10.0)
    await runtime.close("client_closed")
