# SPDX-License-Identifier: Apache-2.0
"""The realtime deployment: its route, what it grants, and the events it emits."""

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

SESSION = SessionIdentity("call")


def output(modality: str, payload, *, eos: bool = False) -> OutputChunk:
    return OutputChunk(SESSION, 0, 0, modality, 0.0, 80.0, payload, eos=eos)


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
    response_ids = {event.response_id for event in first + last}
    assert len(response_ids) == 1
    assert converter(output("audio", b"\x01\x00")) == []


def test_realtime_route_is_linear_and_streams_nothing():
    config = PersonaPlexRealtimePipelineConfig(model_path="personaplex")
    stages = {stage.name: stage for stage in config.stages}
    assert tuple(stages) == REALTIME_STAGES
    for name, following in zip(REALTIME_STAGES, REALTIME_STAGES[1:]):
        assert stages[name].next == following
        assert not stages[name].stream_to
    assert stages[REALTIME_STAGES[-1]].terminal
    assert Variants == {
        "offline": PersonaPlexPipelineConfig,
        "realtime": PersonaPlexRealtimePipelineConfig,
    }
    assert PersonaPlexPipelineConfig.realtime_deployment_factory is None


def test_deployment_takes_24khz_audio_in_80ms_units():
    client = object()
    deployment = create_realtime_deployment(client)
    capabilities = deployment.capabilities
    assert capabilities.input_sample_rate_hz == SAMPLE_RATE
    assert capabilities.output_sample_rate_hz == SAMPLE_RATE
    assert capabilities.native_unit_bytes == SAMPLES_PER_FRAME * 2
    assert capabilities.tail_policy == "pad"

    first, second = deployment.adapter_factory(), deployment.adapter_factory()
    assert isinstance(first, CoordinatorAdapter)
    assert first.client is client
    assert first.stages == list(REALTIME_STAGES)
    assert first.output_converter is not second.output_converter


def test_instructions_become_the_text_prompt():
    assert build_session_request({"instructions": "Be brief."}).params == {
        "instructions": "Be brief."
    }
    assert build_session_request({}).params == {}


class PipelineCoordinator:
    """Stands in for the pipeline: each unit comes back as code2wav emits it."""

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
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        modality: str,
        payload: ChunkPayload,
        *,
        eos: bool = False,
        kind: Literal["data", "input_done"] = "data",
    ) -> None:
        self.outputs.put_nowait(
            OutputChunk(
                session_identity,
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
            self.put(session_identity, chunk, "audio", b"\1\0" * SAMPLES_PER_FRAME)
            self.put(session_identity, chunk, "text", {"text": f"w{chunk.seq} "})
        else:
            pass
        if chunk.eos:
            self.put(session_identity, chunk, "audio", None, eos=True)
        else:
            pass
        self.put(session_identity, chunk, "audio", None, kind="input_done")
        return chunk.seq

    async def session_outputs(
        self, session_identity: SessionIdentity
    ) -> AsyncIterator[OutputChunk]:
        while (output := await self.outputs.get()) is not None:
            yield output

    async def close_session(self, session_identity: SessionIdentity) -> None:
        self.outputs.put_nowait(None)


def test_a_call_over_the_websocket_streams_speech_and_its_transcript():
    coordinator = PipelineCoordinator()
    app = create_app(
        Client(coordinator),
        model_name="personaplex",
        realtime_deployment=create_realtime_deployment(Client(coordinator)),
    )
    unit_bytes = SAMPLES_PER_FRAME * 2
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
                    "audio": base64.b64encode(b"\0" * unit_bytes).decode("ascii"),
                    "sglang": {"seq": seq},
                }
            )
        websocket.send_json({"event_id": "end", "type": "sglang.input_audio.end"})
        events = [websocket.receive_json()]
        while events[-1]["type"] != "response.done":
            events.append(websocket.receive_json())

    assert granted["native_unit_ms"] == 80
    assert granted["output_modalities"] == ["audio"]
    request, stages = coordinator.opened[0]
    assert request.params == {"instructions": "Be brief."}
    assert stages == list(REALTIME_STAGES)
    # Note (wilsonzheng0327): Input that ends on a unit boundary is closed by an
    # empty end-of-input unit.
    assert [chunk.eos for chunk in coordinator.appended] == [False, False, True]
    assert coordinator.appended[-1].payload == b""
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
