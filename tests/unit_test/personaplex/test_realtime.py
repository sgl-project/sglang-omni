# SPDX-License-Identifier: Apache-2.0
"""The /v1/realtime deployment: what it grants, what it opens, the events it sends."""

import asyncio
import base64
from collections.abc import AsyncIterator
from typing import Literal

from fastapi.testclient import TestClient

from sglang_omni.client.client import Client
from sglang_omni.models.personaplex.architecture import SAMPLE_RATE, SAMPLES_PER_FRAME
from sglang_omni.models.personaplex.config import (
    REALTIME_STAGES,
    PersonaPlexPipelineConfig,
    PersonaPlexRealtimePipelineConfig,
    Variants,
)
from sglang_omni.models.personaplex.realtime import (
    PersonaPlexOutputConverter,
    build_call_request,
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

CALL = SessionIdentity("call")
UNIT_BYTES = SAMPLES_PER_FRAME * 2


def output(modality: str, payload: ChunkPayload, *, eos: bool = False) -> OutputChunk:
    return OutputChunk(CALL, 0, 0, modality, 0.0, 80.0, payload, eos=eos)


def test_the_call_is_one_response_until_the_caller_stops():
    converter = PersonaPlexOutputConverter()
    first = converter(output("audio", b"\x01\x00"))
    assert [type(event) for event in first] == [ResponseStarted, AudioDelta]
    assert [type(event) for event in converter(output("text", {"text": "hi"}))] == [
        TextDelta
    ]
    last = converter(output("audio", None, eos=True))
    assert [type(event) for event in last] == [
        TextFinished,
        AudioFinished,
        ResponseFinished,
    ]
    assert last[0].text == last[2].text == "hi"
    assert len({event.response_id for event in first + last}) == 1
    assert converter(output("audio", b"\x01\x00")) == []


def test_a_call_that_ends_before_any_output_opens_no_response():
    assert PersonaPlexOutputConverter()(output("audio", None, eos=True)) == []


def test_the_deployment_takes_24khz_audio_in_80ms_units_one_call_at_a_time():
    client = Client(None)
    deployment = create_realtime_deployment(client)
    capabilities = deployment.capabilities
    assert capabilities.input_sample_rate_hz == SAMPLE_RATE
    assert capabilities.output_sample_rate_hz == SAMPLE_RATE
    assert capabilities.native_unit_ms == 80
    assert capabilities.native_unit_bytes == UNIT_BYTES
    assert capabilities.tail_policy == "pad"
    assert deployment.max_connections == 1

    first, second = deployment.adapter_factory(), deployment.adapter_factory()
    assert isinstance(first, CoordinatorAdapter)
    assert first.client is client and first.stages == list(REALTIME_STAGES)
    assert first.output_converter is not second.output_converter


def test_the_realtime_variant_is_one_linear_route_that_streams_nothing():
    config = PersonaPlexRealtimePipelineConfig(model_path="personaplex")
    stages = {stage.name: stage for stage in config.stages}
    assert tuple(stages) == REALTIME_STAGES
    for name, following in zip(REALTIME_STAGES, REALTIME_STAGES[1:]):
        assert stages[name].next == following and not stages[name].stream_to
    assert stages[REALTIME_STAGES[-1]].terminal
    assert Variants["realtime"] is PersonaPlexRealtimePipelineConfig
    assert Variants["offline"] is PersonaPlexPipelineConfig
    assert PersonaPlexPipelineConfig.realtime_deployment_factory is None


def test_instructions_become_the_role_prompt():
    assert build_call_request({"instructions": "Be brief."}).params == {
        "instructions": "Be brief."
    }
    assert build_call_request({}).params == {}


class ScriptedPipeline:
    """Stands in for the coordinator: each unit comes back as code2wav emits it."""

    def __init__(self) -> None:
        self.opened: list[tuple[OmniRequest, list[str]]] = []
        self.appended: list[TimedChunk] = []
        self.outputs: asyncio.Queue[OutputChunk | None] = asyncio.Queue()

    def health(self) -> dict[str, bool]:
        return {"running": True}

    async def open_session(
        self,
        request: OmniRequest,
        *,
        stages: list[str],
        limits: SessionLimits | None,
        session_id: str | None,
    ) -> SessionIdentity:
        self.opened.append((request, stages))
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
        self, session_identity: SessionIdentity, chunk: TimedChunk
    ) -> int:
        self.appended.append(chunk)
        if chunk.payload:
            self.put(chunk, "audio", b"\1\0" * SAMPLES_PER_FRAME)
            self.put(chunk, "text", {"text": f"w{chunk.seq} "})
        else:
            pass
        if chunk.eos:
            self.put(chunk, "audio", None, eos=True)
        else:
            pass
        self.put(chunk, "audio", None, kind="input_done")
        return chunk.seq

    async def session_outputs(
        self, session_identity: SessionIdentity
    ) -> AsyncIterator[OutputChunk]:
        while (chunk := await self.outputs.get()) is not None:
            yield chunk

    async def close_session(self, session_identity: SessionIdentity) -> None:
        self.outputs.put_nowait(None)


def test_a_call_over_the_websocket_streams_speech_and_its_transcript():
    pipeline = ScriptedPipeline()
    client = Client(pipeline)
    app = create_app(
        client,
        model_name="personaplex",
        realtime_deployment=create_realtime_deployment(client),
    )
    with TestClient(app).websocket_connect("/v1/realtime") as websocket:
        assert websocket.receive_json()["type"] == "session.created"
        websocket.send_json(
            {
                "event_id": "update",
                "type": "session.update",
                "session": {"instructions": "Be brief."},
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
    request, stages = pipeline.opened[0]
    assert request.params == {"instructions": "Be brief."}
    assert stages == list(REALTIME_STAGES)
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
