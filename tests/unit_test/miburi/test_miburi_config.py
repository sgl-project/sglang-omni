# SPDX-License-Identifier: Apache-2.0
"""MIBURI served from its checkpoint: model selection, the graph, the checkpoint each stage loads and the realtime deployment, built without weights."""

import asyncio
import base64
from collections.abc import AsyncIterator
from typing import Literal

import pytest
from fastapi.testclient import TestClient

from sglang_omni.client.client import Client
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import (
    resolve_stage_factory_arg_defaults,
    resolve_stage_factory_args,
    resolve_stage_factory_kwargs,
)
from sglang_omni.models.miburi.config import (
    GESTURE_STAGE,
    MIBURI_ARCH,
    MOSHIKO_MODEL_PATH,
    PERSONAPLEX_MODEL_PATH,
    MiburiPersonaPlexPipelineConfig,
    MiburiPipelineConfig,
)
from sglang_omni.models.miburi.formats import MOTION, SMPLX_MOTION, SMPLX_MOTION_PORT
from sglang_omni.models.personaplex.architecture import SAMPLES_PER_FRAME
from sglang_omni.models.personaplex.config import PersonaPlexSessionPipelineConfig
from sglang_omni.models.personaplex.realtime import build_session_request
from sglang_omni.models.personaplex.session_hooks import AGENT_FRAMES, PERSONAPLEX_REPLY
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import (
    ChunkPayload,
    OutputChunk,
    SessionIdentity,
    SessionLimits,
    TimedChunk,
)
from sglang_omni.serve.openai_api import create_app
from sglang_omni.utils.hf import try_resolve_arch_from_miburi_layout
from sglang_omni.utils.imports import import_string

SPEECH_STAGES = ("preprocessing", "mimi_encode", "lm", "code2wav")
CALL = SessionIdentity("call")
UNIT_BYTES = SAMPLES_PER_FRAME * 2


def test_the_checkpoint_layout_selects_miburi_and_the_variant_its_backbone(tmp_path):
    marker = (
        tmp_path / "experiments/demoexp_release_gtdm3_goodspk/speaker_id_to_index.json"
    )
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")

    default = ConfigManager.from_model_path(str(tmp_path)).config
    personaplex = ConfigManager.from_model_path(str(tmp_path), variant="personaplex")

    assert type(default) is MiburiPipelineConfig
    assert type(personaplex.config) is MiburiPersonaPlexPipelineConfig


def test_the_hub_checkpoint_is_recognised_by_its_marker(monkeypatch):
    monkeypatch.setattr(
        "sglang_omni.utils.hf.hf_hub_download", lambda **_: "/cache/marker"
    )
    assert (
        try_resolve_arch_from_miburi_layout("m-hamza-mughal/miburi-release-checkpoints")
        == MIBURI_ARCH
    )


@pytest.mark.parametrize(
    "config_cls", [MiburiPipelineConfig, MiburiPersonaPlexPipelineConfig]
)
def test_both_backbones_build_the_speech_route_and_the_gesture_node(config_cls):
    config = config_cls(model_path="miburi-checkpoint")
    graph = config.session_graph()

    assert [stage.name for stage in config.stages] == [*SPEECH_STAGES, GESTURE_STAGE]
    assert graph.reading_nodes("speech", AGENT_FRAMES) == ("gesture",)
    assert not graph.is_published("speech", AGENT_FRAMES)
    assert graph.is_published("speech", PERSONAPLEX_REPLY)
    assert graph.is_published("gesture", SMPLX_MOTION)
    assert config.plugin_output_ports() == (SMPLX_MOTION_PORT,)
    assert graph.plugin_formats == {SMPLX_MOTION}
    assert config.stage_factory_kwargs("code2wav")["relays_agent_frames"]


@pytest.mark.parametrize(
    ("config_cls", "arguments", "speech_model_path"),
    [
        (MiburiPipelineConfig, [], MOSHIKO_MODEL_PATH),
        (MiburiPersonaPlexPipelineConfig, [], PERSONAPLEX_MODEL_PATH),
        (
            MiburiPipelineConfig,
            ["--speech-model-path", "/models/moshiko"],
            "/models/moshiko",
        ),
    ],
)
def test_speech_stages_load_the_backbone_and_the_gesture_stage_the_miburi_checkpoint(
    config_cls, arguments, speech_model_path
):
    manager = ConfigManager(config_cls(model_path="miburi-checkpoint"))
    config = manager.merge_config(manager.parse_extra_args(arguments))
    gesture = config.stage_named(GESTURE_STAGE)

    for stage in SPEECH_STAGES:
        resolved = resolve_stage_factory_args(config.stage_named(stage), config)
        assert resolved["model_path"] == speech_model_path
    assert {
        **resolve_stage_factory_arg_defaults(gesture, config),
        **resolve_stage_factory_kwargs(gesture, config),
    }["model_path"] == "miburi-checkpoint"


@pytest.mark.parametrize(
    "config_cls", [MiburiPipelineConfig, MiburiPersonaPlexPipelineConfig]
)
def test_session_lm_overrides_resolve_as_for_a_personaplex_session(config_cls):
    """MIBURI's LM keeps the session LM's typed factory through a command-line override."""
    arguments = ["--lm.factory.depformer_cuda_graph_batch_sizes", "[1, 3]"]
    resolved_arguments = []
    for pipeline_type in (config_cls, PersonaPlexSessionPipelineConfig):
        manager = ConfigManager(pipeline_type(model_path="model", max_sessions=3))
        updated = manager.merge_config(manager.parse_extra_args(arguments))
        resolved = pipeline_type.model_validate(updated.model_dump())
        resolved_arguments.append(
            resolve_stage_factory_args(resolved.stage_named("lm"), resolved)
        )
    miburi_arguments, speech_arguments = resolved_arguments
    del miburi_arguments["model_path"], speech_arguments["model_path"]
    assert miburi_arguments == speech_arguments
    assert miburi_arguments["depformer_cuda_graph_batch_sizes"] == [1, 3]


def test_the_moshiko_backbone_opens_sessions_without_voice_or_role_prompt():
    moshiko = MiburiPipelineConfig(model_path="miburi-checkpoint")
    personaplex = MiburiPersonaPlexPipelineConfig(model_path="miburi-checkpoint")

    assert build_session_request(moshiko, {}).params == {
        "voice": "",
        "instructions": "",
    }
    assert build_session_request(personaplex, {}).params == {}


@pytest.mark.parametrize("voices", [(), ("NATF2", "NATM1")])
def test_the_realtime_voices_are_the_speech_checkpoints(tmp_path, voices):
    (tmp_path / "voices").mkdir()
    for voice in voices:
        (tmp_path / "voices" / f"{voice}.pt").write_bytes(b"")
    config = MiburiPersonaPlexPipelineConfig(
        model_path="miburi-checkpoint", speech_model_path=str(tmp_path)
    )

    deployment = import_string(config.realtime_deployment_factory)(Client(None), config)

    assert deployment.capabilities.voices == voices
    assert deployment.capabilities.plugin_output_ports == (SMPLX_MOTION_PORT,)


class ScriptedPipeline:
    """Stands in for the coordinator: each unit comes back as the speech node's reply and the gesture node's motion."""

    def __init__(self) -> None:
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
        return SessionIdentity(session_id or "call")

    def put(
        self,
        chunk: TimedChunk,
        modality: str,
        payload: ChunkPayload,
        format: str,
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
                format=format,
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
        if chunk.payload:
            reply = {"text": f"w{chunk.seq} ", "pcm": b"\1\0" * SAMPLES_PER_FRAME}
            self.put(chunk, "audio", reply, PERSONAPLEX_REPLY, eos=chunk.eos)
            self.put(chunk, MOTION, b"\0" * 16, SMPLX_MOTION)
        else:
            reply = {"text": "", "pcm": b""}
            self.put(chunk, "audio", reply, PERSONAPLEX_REPLY, eos=chunk.eos)
        self.put(chunk, "audio", None, PERSONAPLEX_REPLY, kind="input_done")
        return chunk.seq

    async def session_outputs(
        self, session_identity: SessionIdentity
    ) -> AsyncIterator[OutputChunk]:
        while (chunk := await self.outputs.get()) is not None:
            yield chunk

    async def close_session(self, session_identity: SessionIdentity) -> None:
        self.outputs.put_nowait(None)


def test_a_realtime_session_advertises_motion_and_streams_it_beside_the_reply(
    tmp_path,
):
    config = MiburiPipelineConfig(
        model_path="miburi-checkpoint", speech_model_path=str(tmp_path)
    )
    client = Client(ScriptedPipeline())
    app = create_app(
        client,
        model_name="miburi",
        realtime_deployment=import_string(config.realtime_deployment_factory)(
            client, config
        ),
    )
    with TestClient(app).websocket_connect("/v1/realtime") as websocket:
        created = websocket.receive_json()
        websocket.send_json(
            {"event_id": "update", "type": "session.update", "session": {}}
        )
        while websocket.receive_json()["type"] != "session.updated":
            pass
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

    assert created["session"]["sglang"]["plugin_output_ports"] == [
        {"modality": MOTION, "format": SMPLX_MOTION, "params": {"fps": 25}}
    ]
    motion = [event for event in events if event["type"] == "sglang.output_chunk"]
    assert [(event["format"], event["payload_b64"]) for event in motion] == [
        (SMPLX_MOTION, base64.b64encode(b"\0" * 16).decode("ascii"))
    ] * 2
    assert [event["type"] for event in events].count("response.output_audio.delta") == 2
