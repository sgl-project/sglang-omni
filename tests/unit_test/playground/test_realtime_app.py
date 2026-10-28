# SPDX-License-Identifier: Apache-2.0
"""The realtime demo's startup warmup against two native full-duplex deployments."""

from __future__ import annotations

import base64
import importlib.util
import json
import math
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from pydantic import ValidationError

from sglang_omni.models.minicpm_o.native_config import MiniCPMODuplexPipelineConfig
from sglang_omni.models.minicpm_o.session_adapters import build_realtime_deployment
from sglang_omni.models.personaplex.config import PersonaPlexSessionPipelineConfig
from sglang_omni.models.personaplex.realtime import create_realtime_deployment
from sglang_omni.serve.realtime.negotiation import SessionNegotiation
from sglang_omni.serve.realtime.runtime import MEDIA_TIME_TOLERANCE_MS
from sglang_omni.serve.realtime.schema import CLIENT_EVENT
from sglang_omni.serve.realtime.types import (
    PCM16_BYTES_PER_SAMPLE,
    Capabilities,
    ProtocolError,
    RuntimeLimits,
)

APP_PATH = Path(__file__).resolve().parents[3] / "playground" / "realtime" / "app.py"


@pytest.fixture
def playground() -> ModuleType:
    spec = importlib.util.spec_from_file_location("realtime_playground", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScriptedUpstream:
    """One deployment's realtime endpoint: its capabilities, the server's own event schema and session admission, and the protocol events the warmup waits for."""

    def __init__(self, model: str, capabilities: Capabilities) -> None:
        self.model = model
        self.capabilities = capabilities
        self.negotiation = SessionNegotiation(
            model=model, capabilities=capabilities, limits=RuntimeLimits()
        )
        self.session_updates: list[dict[str, object]] = []
        self.image_appends = 0
        self.accepted_end_ms: list[float] = []

    def application(self) -> web.Application:
        application = web.Application()
        application.router.add_get("/v1/realtime/capabilities", self.serve_capabilities)
        application.router.add_get("/v1/realtime", self.serve_session)
        return application

    async def serve_capabilities(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "model": self.model,
                **self.capabilities.to_granted_capabilities(),
                "limits": asdict(RuntimeLimits()),
            }
        )

    async def refuse(
        self, websocket: web.WebSocketResponse, code: str, message: str
    ) -> None:
        await websocket.send_json(
            {"type": "error", "error": {"code": code, "message": message}}
        )

    async def serve_session(self, request: web.Request) -> web.WebSocketResponse:
        websocket = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)
        await websocket.prepare(request)
        await websocket.send_json({"type": "session.created"})
        accepted_samples = 0
        async for message in websocket:
            event = json.loads(message.data)
            try:
                CLIENT_EVENT.validate_python(event)
                kind = event["type"]
                if kind == "session.update":
                    self.session_updates.append(event["session"])
                    self.negotiation.negotiate({}, "CREATED", event["session"])
                    await websocket.send_json({"type": "session.updated"})
                elif kind == "sglang.input_image.append":
                    self.image_appends += 1
                elif kind == "input_audio_buffer.append":
                    # Note (ruoyu): The runtime's media clock, so audio stamped at another sample rate fails here the way it fails upstream.
                    if not math.isclose(
                        event["sglang"]["t_start_ms"],
                        self.capabilities.input_duration_ms(accepted_samples),
                        rel_tol=0,
                        abs_tol=MEDIA_TIME_TOLERANCE_MS,
                    ):
                        raise ProtocolError(
                            "invalid_state",
                            "input media time must be sample-contiguous",
                        )
                    else:
                        accepted_samples += (
                            len(base64.b64decode(event["audio"]))
                            // PCM16_BYTES_PER_SAMPLE
                        )
                elif kind == "sglang.input_audio.end":
                    self.accepted_end_ms.append(
                        self.capabilities.input_duration_ms(accepted_samples)
                    )
                    await websocket.send_json(
                        {
                            "type": "response.output_audio.delta",
                            "delta": base64.b64encode(b"\0\0").decode(),
                        }
                    )
                    await websocket.send_json({"type": "sglang.input_audio.drained"})
                elif kind == "session.close":
                    await websocket.send_json({"type": "session.closed"})
                else:
                    pass
            except ValidationError as error:
                await self.refuse(
                    websocket, "invalid_request", error.errors()[0]["msg"]
                )
            except ProtocolError as error:
                await self.refuse(websocket, error.code, str(error))
        return websocket


async def warm_up(upstream: ScriptedUpstream, playground: ModuleType) -> None:
    async with TestServer(upstream.application()) as server:
        await playground.warmup(str(server.make_url("/")).rstrip("/"))


@pytest.mark.asyncio
async def test_personaplex_warmup_sends_only_what_the_deployment_declares(
    tmp_path: Path, playground: ModuleType
) -> None:
    """The warmup sent MiniCPM-o's greedy, force_listen_count, reference audio and 16 kHz timestamps to any native deployment, so the page never started on PersonaPlex."""
    (tmp_path / "voices").mkdir()
    (tmp_path / "voices" / "NATF2.pt").write_bytes(b"")
    deployment = create_realtime_deployment(
        Mock(), PersonaPlexSessionPipelineConfig(model_path=str(tmp_path))
    )
    upstream = ScriptedUpstream("nvidia/personaplex-7b-v1", deployment.capabilities)

    await warm_up(upstream, playground)

    assert upstream.image_appends == 0
    assert upstream.accepted_end_ms
    assert min(upstream.accepted_end_ms) > 0


@pytest.mark.asyncio
async def test_minicpmo_warmup_keeps_its_settings_voice_and_image_round(
    playground: ModuleType,
) -> None:
    deployment = build_realtime_deployment(
        Mock(), MiniCPMODuplexPipelineConfig(model_path="unused")
    )
    upstream = ScriptedUpstream("openbmb/MiniCPM-o-4_5", deployment.capabilities)

    await warm_up(upstream, playground)

    audio_round, video_round = upstream.session_updates
    assert audio_round == video_round
    assert audio_round["sglang"]["sampling"] == playground.WARMUP_SAMPLING
    assert audio_round["sglang"]["reference_audio"]["media_type"] == "audio/wav"
    assert upstream.image_appends == 1
    assert len(upstream.accepted_end_ms) == 2
    assert min(upstream.accepted_end_ms) > 0
