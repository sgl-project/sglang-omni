# SPDX-License-Identifier: Apache-2.0
"""The Live Avatar stage and what its node of a Session graph accepts and publishes."""

from __future__ import annotations

from pydantic import Field

from sglang_omni.config.schema import FactoryArgs, StageConfig
from sglang_omni.models.liveavatar.formats import (
    CHUNK_MS,
    HEIGHT,
    RENDER_LEAD_MS,
    WIDTH,
    jpeg_frame_port,
)
from sglang_omni.models.speech_avatar.node import session_ports
from sglang_omni.proto.streams import SessionPorts

LIVEAVATAR_STAGE = "liveavatar"


class LiveAvatarFactoryArgs(FactoryArgs):
    """The Live Avatar workers (examples/speech_avatar_workers/liveavatar_worker.py, five GPUs each) and how the node uses them."""

    worker_urls: list[str] | None = Field(default=None, min_length=1)
    "One http://host:port per worker; each renders the blocks of one Session at a time."
    speech_sample_rate_hz: int | None = Field(default=None, gt=0)
    "The rate of the speech node's pcm_f32le reply audio."
    input_sample_rate_hz: int | None = Field(default=None, gt=0)
    "The rate of the Session input's pcm16 audio, 16 kHz unless the speech node takes another."
    width: int | None = Field(default=None, ge=16)
    height: int | None = Field(default=None, ge=16)
    "The workers' picture, which their /info must match."
    render_lead_ms: float | None = Field(default=None, ge=0)
    "How far past the frame the node has reached a reply starts: the time its first block has to arrive and pass the pipeline before that frame plays."
    tail_ms: float | None = Field(default=None, ge=0)
    "Blocks keep rendering this long past a reply, so the mouth settles before the picture holds."
    prime_chunks: int | None = Field(default=None, ge=0)
    "Silent blocks rendered (and dropped) when a Session starts, so its slow first blocks come before the first reply."
    pipeline_depth: int | None = Field(default=None, ge=1)
    "Blocks of a Session in flight at once; the workers' /info pipeline_depth unless set."
    startup_timeout_s: float | None = Field(default=None, gt=0)
    "How long the stage waits at start-up for every worker to finish loading and compiling."


class LiveAvatarStageConfig(StageConfig):
    factory: LiveAvatarFactoryArgs = Field(default_factory=LiveAvatarFactoryArgs)

    def session_ports(self) -> SessionPorts:
        factory = self.factory
        return session_ports(
            jpeg_frame_port(
                width=factory.width or WIDTH,
                height=factory.height or HEIGHT,
                render_lead_ms=(
                    RENDER_LEAD_MS
                    if factory.render_lead_ms is None
                    else factory.render_lead_ms
                ),
                chunk_ms=CHUNK_MS,
            )
        )


def liveavatar_stage() -> LiveAvatarStageConfig:
    return LiveAvatarStageConfig(
        name=LIVEAVATAR_STAGE,
        process=LIVEAVATAR_STAGE,
        terminal=True,
        factory_path="sglang_omni.models.liveavatar.worker.create_scheduler",
    )


SESSION_PORTS = liveavatar_stage().session_ports()


__all__ = [
    "LIVEAVATAR_STAGE",
    "LiveAvatarStageConfig",
    "SESSION_PORTS",
    "liveavatar_stage",
]
