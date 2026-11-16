# SPDX-License-Identifier: Apache-2.0
"""The avatar_worker stage and what its node of a Session graph accepts and publishes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from sglang_omni.config.schema import FactoryArgs, StageConfig
from sglang_omni.models.avatar_worker.formats import (
    BVH_MOTION,
    JPEG_FRAME,
    bvh_motion_port,
    jpeg_frame_port,
)
from sglang_omni.proto.streams import (
    PCM16,
    SESSION_INPUT_SAMPLE_RATE_HZ,
    SESSION_UNIT_MS,
    AudioPort,
    PluginPort,
    SessionPorts,
)

AVATAR_WORKER_STAGE = "avatar_worker"
# A stage with nothing configured publishes 25 fps 512x512 frames, five a chunk, hearing 200 ms ahead.
DEFAULT_FPS = 25
DEFAULT_CHUNK_FRAMES = 5
DEFAULT_LOOKAHEAD_MS = 200
DEFAULT_RENDER_MS = 500
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512


class AvatarWorkerFactoryArgs(FactoryArgs):
    """The chunk workers of one avatar model, the chunk geometry they serve and the medium the node publishes; the geometry must match each worker's /info."""

    renderer_urls: list[str] | None = Field(default=None, min_length=1)
    "The model's chunk workers, one http://host:port each."
    media: Literal["jpeg_frame", "bvh_motion"] = JPEG_FRAME
    "jpeg_frame: one JPEG a frame at width x height; bvh_motion: one row of the skeleton's root position, joint quaternions and face channels a frame."
    fps: float | None = Field(default=None, gt=0)
    chunk_frames: int | None = Field(default=None, ge=1)
    "Frames a worker renders per request."
    lookahead_ms: float | None = Field(default=None, ge=0)
    "The agent's audio past a chunk a worker reads to render it."
    listens: bool | None = None
    "The model listens: each request also carries the user's audio of a window as long as the speech window, lagging the chunk by that window, so the chunk renders as soon as the user's audio reaches it."
    render_ms: float | None = Field(default=None, ge=0)
    "A chunk's render time on the deployment's GPUs, which the port's reply_media_wait_ms covers."
    width: int | None = Field(default=None, ge=16)
    height: int | None = Field(default=None, ge=16)
    skeleton_path: str | None = None
    "bvh_motion: a JSON file of the skeleton's joints, parents, offsets and face channel names."
    speech_sample_rate_hz: int | None = Field(default=None, gt=0)
    "The rate of the speech node's audio: pcm_f32le replies, or personaplex_reply units."
    input_sample_rate_hz: int | None = Field(default=None, gt=0)
    "The rate of the Session input's pcm16 audio, 16 kHz unless the speech node takes another, such as PersonaPlex's 24 kHz."
    reply_stream: bool | None = None
    "The speech node streams the agent's audio as units from the Session's start (PersonaPlex's personaplex_reply): every chunk waits until the units reach its window's end."
    sessions_per_renderer: int | None = Field(default=None, ge=1)
    "Sessions one worker keeps at real time; the stage holds this many per worker."
    render_timeout_s: float | None = Field(default=None, gt=0)
    "A worker that has not answered a chunk for this long is treated as gone."
    startup_timeout_s: float | None = Field(default=None, gt=0)
    "How long the stage waits at startup for each worker's /health, while it loads its model."


class AvatarWorkerStageConfig(StageConfig):
    factory: AvatarWorkerFactoryArgs = Field(default_factory=AvatarWorkerFactoryArgs)

    def session_ports(self) -> SessionPorts:
        return session_ports(plugin_port(self.factory))


def read_skeleton(path: str) -> dict:
    skeleton = json.loads(Path(path).read_text())
    joints = skeleton["joints"]
    if len(skeleton["parents"]) != len(joints) or len(skeleton["offsets"]) != len(
        joints
    ):
        raise ValueError(
            f"the skeleton at {path} needs a parent and an offset for each of its {len(joints)} joints"
        )
    else:
        return {
            "joints": joints,
            "parents": skeleton["parents"],
            "offsets": skeleton["offsets"],
            "face": skeleton.get("face", []),
            "up": skeleton.get("up", "y"),
        }


def plugin_port(factory: AvatarWorkerFactoryArgs) -> PluginPort:
    fps = factory.fps or DEFAULT_FPS
    chunk_ms = 1000 * (factory.chunk_frames or DEFAULT_CHUNK_FRAMES) / fps
    timing = {
        "fps": fps,
        "chunk_ms": chunk_ms,
        "lookahead_ms": (
            DEFAULT_LOOKAHEAD_MS
            if factory.lookahead_ms is None
            else factory.lookahead_ms
        ),
        "render_ms": (
            DEFAULT_RENDER_MS if factory.render_ms is None else factory.render_ms
        ),
    }
    if factory.media == BVH_MOTION:
        if factory.skeleton_path is None:
            raise ValueError("a bvh_motion avatar_worker stage needs a skeleton_path")
        else:
            return bvh_motion_port(**timing, **read_skeleton(factory.skeleton_path))
    else:
        return jpeg_frame_port(
            **timing,
            width=factory.width or DEFAULT_WIDTH,
            height=factory.height or DEFAULT_HEIGHT,
        )


def session_ports(port: PluginPort) -> SessionPorts:
    return SessionPorts(
        input=(AudioPort(PCM16, SESSION_INPUT_SAMPLE_RATE_HZ),),
        output=None,
        unit_ms=SESSION_UNIT_MS,
        plugin_output=(port,),
    )


SESSION_PORTS = session_ports(plugin_port(AvatarWorkerFactoryArgs()))


def avatar_worker_stage() -> AvatarWorkerStageConfig:
    return AvatarWorkerStageConfig(
        name=AVATAR_WORKER_STAGE,
        process=AVATAR_WORKER_STAGE,
        terminal=True,
        factory_path="sglang_omni.models.avatar_worker.renderer.create_scheduler",
    )
