# SPDX-License-Identifier: Apache-2.0
"""The VAD stage and what its node of a Session graph accepts and publishes."""

from __future__ import annotations

from pydantic import Field

from sglang_omni.config.schema import FactoryArgs, StageConfig
from sglang_omni.proto.streams import (
    COMMIT,
    CONFIGURE,
    PCM16,
    SERVER_VAD,
    SESSION_INPUT_SAMPLE_RATE_HZ,
    SESSION_UNIT_MS,
    AudioPort,
    SessionPorts,
)

VAD_STAGE = "vad"
TURN_DETECTION_FIELDS = ("threshold", "prefix_padding_ms", "silence_duration_ms")
SESSION_PORTS = SessionPorts(
    input=(AudioPort(PCM16, SESSION_INPUT_SAMPLE_RATE_HZ),),
    output=None,
    unit_ms=SESSION_UNIT_MS,
    controls=frozenset({COMMIT, CONFIGURE}),
    hot_settings=frozenset({"turn_detection"}),
    turn_detection_types=(SERVER_VAD,),
)


class VADFactoryArgs(FactoryArgs):
    """Detector defaults of every session of the stage; a session's `turn_detection` overrides them."""

    threshold: float | None = Field(default=None, ge=0, le=1)
    prefix_padding_ms: int | None = Field(default=None, ge=0)
    # Note (ruoyu): A pause after a comma or a short "Sorry." often runs past 500 ms and split one utterance into two turns, which 800 ms avoided in our zh and en runs.
    silence_duration_ms: int | None = Field(default=800, ge=0)
    # note (wenyao): On the Full-Duplex-Bench v1.5 dev split, 6 dB still heard every interruption, backchannel and side remark, and ignored 71 of 75 background voices.
    background_margin_db: float | None = Field(default=6.0, ge=0)
    max_open_sessions: int | None = Field(default=None, ge=1)
    max_concurrency: int | None = Field(default=None, ge=1)


class VADStageConfig(StageConfig):
    factory: VADFactoryArgs = Field(default_factory=VADFactoryArgs)


def vad_stage() -> VADStageConfig:
    return VADStageConfig(
        name=VAD_STAGE,
        process=VAD_STAGE,
        terminal=True,
        factory_path="sglang_omni.models.vad.session.create_scheduler",
    )
