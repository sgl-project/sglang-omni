# SPDX-License-Identifier: Apache-2.0
"""The /v1/realtime deployment: what it grants, what it opens, the events it sends."""

import asyncio
import base64
from collections.abc import AsyncIterator
from typing import Literal

import pytest
from fastapi.testclient import TestClient

from sglang_omni.client.client import Client
from sglang_omni.models.personaplex.architecture import SAMPLE_RATE, SAMPLES_PER_FRAME
from sglang_omni.models.personaplex.config import (
    PersonaPlexPipelineConfig,
    PersonaPlexSessionPipelineConfig,
    PersonaPlexSessionSampling,
    Variants,
)
from sglang_omni.models.personaplex.realtime import (
    PersonaPlexOutputConverter,
    build_session_request,
    create_realtime_deployment,
)
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import (
    ChunkPayload,
    OutputChunk,
    SessionIdentity,
    SessionLimits,
    TimedChunk,
)
from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.realtime.adapters import CoordinatorAdapter
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    AudioFinished,
    ResponseFinished,
    ResponseStarted,
    TextDelta,
    TextFinished,
)
from sglang_omni.utils.imports import import_string

CALL = SessionIdentity("call")
UNIT_BYTES = SAMPLES_PER_FRAME * 2


def output(
    text: str = "", pcm: bytes = b"", *, eos: bool = False, open_index: int = 1
) -> OutputChunk:
    return OutputChunk(
        SessionIdentity("call", open_index),
        0,
        0,
        "audio",
        0.0,
        80.0,
        {"text": text, "pcm": pcm},
        eos=eos,
    )


def test_the_call_is_one_response_until_the_caller_stops():
    converter = PersonaPlexOutputConverter()
    first = converter(output(pcm=b"\x01\x00"))
    assert [type(event) for event in first] == [ResponseStarted, AudioDelta]
    assert [type(event) for event in converter(output("hi"))] == [TextDelta]
    last = converter(output("!", eos=True))
    assert [type(event) for event in last] == [
        TextDelta,
        TextFinished,
        AudioFinished,
        ResponseFinished,
    ]
    assert last[1].text == last[3].text == "hi!"
    assert last[3].has_audio
    assert len({event.response_id for event in first + last}) == 1

    reopened = converter(output("again", open_index=2, eos=True))
    assert [type(event) for event in reopened] == [
        ResponseStarted,
        TextDelta,
        TextFinished,
        ResponseFinished,
    ]
    assert reopened[0].response_id != first[0].response_id
    assert not reopened[-1].has_audio


def test_a_call_that_ends_before_any_output_opens_no_response():
    assert PersonaPlexOutputConverter()(output(eos=True)) == []


@pytest.fixture
def model_dir(tmp_path):
    (tmp_path / "voices").mkdir()
    for voice in ("NATF2", "NATM1"):
        (tmp_path / "voices" / f"{voice}.pt").write_bytes(b"")
    return str(tmp_path)


def test_the_deployment_takes_24khz_audio_in_80ms_units_up_to_max_sessions_calls(
    model_dir,
):
    client = Client(None)
    config = PersonaPlexSessionPipelineConfig(model_path=model_dir, max_sessions=2)
    # Note (wilsonzheng0327): Called the way the launcher calls it.
    deployment = import_string(config.realtime_deployment_factory)(client, config)
    capabilities = deployment.capabilities
    assert capabilities.input_sample_rate_hz == SAMPLE_RATE
    assert capabilities.output_sample_rate_hz == SAMPLE_RATE
    assert capabilities.native_unit_ms == 80
    assert capabilities.native_unit_bytes == UNIT_BYTES
    assert capabilities.tail_policy == "pad"
    assert capabilities.voices == ("NATF2", "NATM1")
    assert set(capabilities.sampling_parameters) >= {"temperature", "audio_top_k"}
    assert deployment.max_connections == 2

    first, second = deployment.adapter_factory(), deployment.adapter_factory()
    assert isinstance(first, CoordinatorAdapter)
    assert first.client is client
    assert first.output_converter is not second.output_converter


def test_only_the_session_variant_serves_and_needs_realtime():
    assert Variants["session"] is PersonaPlexSessionPipelineConfig
    assert Variants["offline"] is PersonaPlexPipelineConfig
    assert PersonaPlexPipelineConfig.realtime_deployment_factory is None
    assert PersonaPlexSessionPipelineConfig.is_realtime_only
    assert not PersonaPlexPipelineConfig.is_realtime_only


def test_session_settings_override_the_deployment_for_that_session_only(model_dir):
    config = PersonaPlexSessionPipelineConfig(
        model_path=model_dir,
        voice="NATM1",
        instructions="Be kind.",
        sampling=PersonaPlexSessionSampling(temperature=0.5, seed=3),
    )
    session = {
        "instructions": "Be brief.",
        "sglang": {"voice": "NATF2", "sampling": {"temperature": 0.9}},
    }
    assert build_session_request(config, session).params == {
        "temperature": 0.9,
        "seed": 3,
        "voice": "NATF2",
        "instructions": "Be brief.",
    }
    assert build_session_request(config, {}).params == {
        "temperature": 0.5,
        "seed": 3,
        "voice": "NATM1",
        "instructions": "Be kind.",
    }
    default = PersonaPlexSessionPipelineConfig(model_path=model_dir)
    assert build_session_request(default, {}).params == {}


class ScriptedPipeline:
    """Stands in for the coordinator: each unit comes back as code2wav emits it."""

    def __init__(self) -> None:
        self.opened: list[OmniRequest] = []
        self.appended: list[TimedChunk] = []
        self.outputs: asyncio.Queue[OutputChunk | None] = asyncio.Queue()

    def health(self) -> dict[str, bool]:
        return {"running": True}

    async def open_session(
        self,
        request: OmniRequest,
        *,
        limits: SessionLimits | None,
        session_id: str | None,
    ) -> SessionIdentity:
        self.opened.append(request)
        return SessionIdentity(session_id or "call")

    def put(
        self,
        chunk: TimedChunk,
        modality: str,
        payload: ChunkPayload,
        *,
        eos: bool = False,
        kind: Literal["data", "input_done"] = "data",
    ) -> None:
        self.outputs.put_nowait(
            OutputChunk(
                CALL,
                0,
                chunk.seq,
                modality,
                chunk.t_start_ms,
                80.0,
                payload,
                eos=eos,
                kind=kind,
            )
        )

    async def append_session(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        *,
        ready_timestamp_ns: int | None = None,
        ready_run_id: str | None = None,
    ) -> int:
        self.appended.append(chunk)
        if chunk.payload:
            unit = {"text": f"w{chunk.seq} ", "pcm": b"\1\0" * SAMPLES_PER_FRAME}
        else:
            unit = {"text": "", "pcm": b""}
        self.put(chunk, "audio", unit, eos=chunk.eos)
        self.put(chunk, "audio", None, kind="input_done")
        return chunk.seq

    async def session_outputs(
        self, session_identity: SessionIdentity
    ) -> AsyncIterator[OutputChunk]:
        while (chunk := await self.outputs.get()) is not None:
            yield chunk

    async def close_session(self, session_identity: SessionIdentity) -> None:
        self.outputs.put_nowait(None)


def test_a_call_over_the_websocket_streams_speech_and_its_transcript(model_dir):
    pipeline = ScriptedPipeline()
    client = Client(pipeline)
    app = create_app(
        client,
        model_name="personaplex",
        realtime_deployment=create_realtime_deployment(
            client, PersonaPlexSessionPipelineConfig(model_path=model_dir)
        ),
    )
    with TestClient(app).websocket_connect("/v1/realtime") as websocket:
        assert websocket.receive_json()["type"] == "session.created"
        websocket.send_json(
            {
                "event_id": "update",
                "type": "session.update",
                "session": {
                    "instructions": "Be brief.",
                    "sglang": {"voice": "NATM1", "sampling": {"seed": 7}},
                },
            }
        )
        events = [websocket.receive_json()]
        while events[-1]["type"] != "session.updated":
            events.append(websocket.receive_json())
        granted = events[-1]["session"]["sglang"]["granted"]
        for seq in range(2):
            websocket.send_json(
                {
                    "event_id": f"audio-{seq}",
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(b"\0" * UNIT_BYTES).decode("ascii"),
                    "sglang": {"seq": seq},
                }
            )
        websocket.send_json({"event_id": "end", "type": "sglang.input_audio.end"})
        events = [websocket.receive_json()]
        while events[-1]["type"] != "response.done":
            events.append(websocket.receive_json())

    assert granted["native_unit_ms"] == 80
    assert granted["output_modalities"] == ["audio"]
    assert pipeline.opened[0].params == {
        "seed": 7,
        "voice": "NATM1",
        "instructions": "Be brief.",
    }
    # Note (wilsonzheng0327): Input ending on a unit boundary is closed by an
    # empty end-of-input unit.
    assert [chunk.eos for chunk in pipeline.appended] == [False, False, True]
    assert pipeline.appended[-1].payload == b""
    types = [event["type"] for event in events]
    assert types.count("response.created") == 1
    assert types.count("response.output_audio.delta") == 2
    transcript = [
        event["delta"]
        for event in events
        if event["type"] == "response.output_audio_transcript.delta"
    ]
    assert "".join(transcript) == "w0 w1 "
    assert events[-1]["response"]["status"] == "completed"
