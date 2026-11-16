# SPDX-License-Identifier: Apache-2.0
"""The avatar stage and what its node of a Session graph accepts and publishes."""

from __future__ import annotations

from pydantic import Field

from sglang_omni.config.schema import FactoryArgs, StageConfig
from sglang_omni.models.avtr.formats import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    jpeg_frame_port,
)
from sglang_omni.proto.streams import (
    PCM16,
    SESSION_INPUT_SAMPLE_RATE_HZ,
    SESSION_UNIT_MS,
    AudioPort,
    SessionPorts,
)

AVATAR_STAGE = "avatar"


def session_ports(width: int, height: int) -> SessionPorts:
    return SessionPorts(
        input=(AudioPort(PCM16, SESSION_INPUT_SAMPLE_RATE_HZ),),
        output=None,
        unit_ms=SESSION_UNIT_MS,
        plugin_output=(jpeg_frame_port(width, height),),
    )


SESSION_PORTS = session_ports(DEFAULT_WIDTH, DEFAULT_HEIGHT)


class AvatarFactoryArgs(FactoryArgs):
    """The AVTR-1 renderer workers, the portrait every Session of the stage animates and how its frames are encoded."""

    renderer_urls: list[str] | None = Field(default=None, min_length=1)
    "The renderer service workers, one http://host:port each; each renders one chunk at a time, of any Session."
    avatar: str | None = None
    "A portrait every worker loaded: the name of a PNG in its reference_frames directory."
    background: str | None = None
    "One of the backgrounds every worker lists at /avatars, such as plain_white."
    speech_sample_rate_hz: int | None = Field(default=None, gt=0)
    "The rate of the speech node's pcm_f32le audio the avatar lip-syncs."
    width: int | None = Field(default=None, ge=16)
    height: int | None = Field(default=None, ge=16)
    jpeg_quality: int | None = Field(default=None, ge=1, le=95)
    sessions_per_renderer: int | None = Field(default=None, ge=1)
    "Sessions one worker keeps at real time; the stage holds this many per worker."
    requests_per_renderer: int | None = Field(default=None, ge=1)
    "Chunks of different Sessions the stage keeps in flight at one worker; set it to sessions_per_renderer for a service that batches them (the session-batched AVTR-1 service), leave it at 1 for the upstream one, which renders one at a time. A worker fails a Session after as many failed chunks in a row as the pool has requests in flight."
    service_jpeg: bool | None = None
    "Ask the workers for JPEG frames of the stage's size and quality instead of I420 to encode here; a worker that does not encode (the upstream service) still sends I420, which the stage then encodes."


class AvatarStageConfig(StageConfig):
    factory: AvatarFactoryArgs = Field(default_factory=AvatarFactoryArgs)

    def session_ports(self) -> SessionPorts:
        """The node's ports at the frame size this stage encodes."""
        return session_ports(
            self.factory.width or DEFAULT_WIDTH, self.factory.height or DEFAULT_HEIGHT
        )


def avatar_stage() -> AvatarStageConfig:
    return AvatarStageConfig(
        name=AVATAR_STAGE,
        process=AVATAR_STAGE,
        terminal=True,
        factory_path="sglang_omni.models.avtr.renderer.create_scheduler",
    )
