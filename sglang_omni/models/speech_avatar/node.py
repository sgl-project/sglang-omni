# SPDX-License-Identifier: Apache-2.0
"""The speech-avatar stages and what their nodes of a Session graph accept and publish."""

from __future__ import annotations

from pydantic import Field

from sglang_omni.config.schema import FactoryArgs, StageConfig
from sglang_omni.models.speech_avatar.formats import jpeg_frame_port, smplx_motion_port
from sglang_omni.proto.streams import (
    PCM16,
    SESSION_INPUT_SAMPLE_RATE_HZ,
    SESSION_UNIT_MS,
    AudioPort,
    PluginPort,
    SessionPorts,
)

SPEECH_VIDEO_STAGE = "speech_video"
SPEECH_MOTION_STAGE = "speech_motion"
# The defaults describe the FlashHead Lite worker (24 new frames of 512x512 at 25 fps per chunk) and the StreamTalk worker (15 frames at 30 fps per block).
DEFAULT_VIDEO_FPS = 25
DEFAULT_VIDEO_SIZE = 512
DEFAULT_VIDEO_CHUNK_MS = 960.0
DEFAULT_MOTION_FPS = 30
DEFAULT_MOTION_CHUNK_MS = 500.0
# A reply's chunk is placed this far past the frame the node has reached, the render's budget beside the client's 100 ms playout delay: a 24-frame FlashHead chunk renders in 125-175 ms on an H100.
DEFAULT_RENDER_LEAD_MS = 150.0


def session_ports(port: PluginPort) -> SessionPorts:
    """The node reads the Session input's audio as its clock and publishes one plugin port."""
    return SessionPorts(
        input=(AudioPort(PCM16, SESSION_INPUT_SAMPLE_RATE_HZ),),
        output=None,
        unit_ms=SESSION_UNIT_MS,
        plugin_output=(port,),
    )


class SpeechWorkerFactoryArgs(FactoryArgs):
    worker_urls: list[str] | None = Field(default=None, min_length=1)
    "The model's worker services, one http://host:port each (examples/speech_avatar_workers); each renders one chunk at a time, of any Session."
    speech_sample_rate_hz: int | None = Field(default=None, gt=0)
    "The rate of the speech node's pcm_f32le reply audio."
    input_sample_rate_hz: int | None = Field(default=None, gt=0)
    "The rate of the Session input's pcm16 audio, 16 kHz unless the speech node takes another."
    fps: int | None = Field(default=None, gt=0)
    chunk_ms: float | None = Field(default=None, gt=0)
    "How long the workers' chunk lasts; the stage checks it against their /info and announces it to clients."
    lookahead_ms: float | None = Field(default=None, ge=0)
    "How much of the agent's speech past a chunk its window reads."
    sessions_per_worker: int | None = Field(default=None, ge=1)
    "Sessions one worker keeps at real time; the stage holds this many per worker."
    startup_timeout_s: float | None = Field(default=None, gt=0)
    "How long the stage waits at start-up for every worker to finish warming up."


class SpeechVideoFactoryArgs(SpeechWorkerFactoryArgs):
    """A talking-head model's workers and the frames they encode."""

    width: int | None = Field(default=None, ge=16)
    height: int | None = Field(default=None, ge=16)
    render_lead_ms: float | None = Field(default=None, ge=0)
    "How far past the frame the node has reached a reply starts: the time its first chunk has to render before that frame plays."
    tail_ms: float | None = Field(default=None, ge=0)
    "Chunks keep rendering this long past a reply, so the mouth settles before the picture holds."
    reply_quiet_ms: float | None = Field(default=None, ge=0)
    "A reply shorter than a chunk starts once its audio stopped arriving for this long (its turn.done comes only after it played out)."
    reply_starts_chunk: bool | None = None
    "Send the frames up to a placed reply at once, so the reply's first chunk starts on its first frame: for chunks long against the render lead (LiveAct)."


class SpeechMotionFactoryArgs(SpeechWorkerFactoryArgs):
    """A gesture model's workers."""


class SpeechVideoStageConfig(StageConfig):
    factory: SpeechVideoFactoryArgs = Field(default_factory=SpeechVideoFactoryArgs)

    def session_ports(self) -> SessionPorts:
        factory = self.factory
        return session_ports(
            jpeg_frame_port(
                fps=factory.fps or DEFAULT_VIDEO_FPS,
                width=factory.width or DEFAULT_VIDEO_SIZE,
                height=factory.height or DEFAULT_VIDEO_SIZE,
                chunk_ms=factory.chunk_ms or DEFAULT_VIDEO_CHUNK_MS,
                lookahead_ms=factory.lookahead_ms or 0.0,
                render_lead_ms=(
                    DEFAULT_RENDER_LEAD_MS
                    if factory.render_lead_ms is None
                    else factory.render_lead_ms
                ),
            )
        )


class SpeechMotionStageConfig(StageConfig):
    factory: SpeechMotionFactoryArgs = Field(default_factory=SpeechMotionFactoryArgs)

    def session_ports(self) -> SessionPorts:
        factory = self.factory
        return session_ports(
            smplx_motion_port(
                fps=factory.fps or DEFAULT_MOTION_FPS,
                chunk_ms=factory.chunk_ms or DEFAULT_MOTION_CHUNK_MS,
                lookahead_ms=factory.lookahead_ms or 0.0,
            )
        )


def speech_video_stage() -> SpeechVideoStageConfig:
    return SpeechVideoStageConfig(
        name=SPEECH_VIDEO_STAGE,
        process=SPEECH_VIDEO_STAGE,
        terminal=True,
        factory_path="sglang_omni.models.speech_avatar.worker.create_video_scheduler",
    )


def speech_motion_stage() -> SpeechMotionStageConfig:
    return SpeechMotionStageConfig(
        name=SPEECH_MOTION_STAGE,
        process=SPEECH_MOTION_STAGE,
        terminal=True,
        factory_path="sglang_omni.models.speech_avatar.worker.create_motion_scheduler",
    )


# The ports of the stages at their defaults; a configured stage announces its own (StageGroup.stage_ports).
VIDEO_SESSION_PORTS = speech_video_stage().session_ports()
MOTION_SESSION_PORTS = speech_motion_stage().session_ports()

__all__ = [
    "MOTION_SESSION_PORTS",
    "SPEECH_MOTION_STAGE",
    "SPEECH_VIDEO_STAGE",
    "SpeechMotionStageConfig",
    "SpeechVideoStageConfig",
    "VIDEO_SESSION_PORTS",
    "speech_motion_stage",
    "speech_video_stage",
]
