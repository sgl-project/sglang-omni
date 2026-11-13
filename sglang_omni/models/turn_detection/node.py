# SPDX-License-Identifier: Apache-2.0
"""The turn detection stage and what its node of a Session graph accepts and publishes."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from sglang_omni.config.schema import FactoryArgs, StageConfig
from sglang_omni.proto.streams import (
    COMMIT,
    CONFIGURE,
    PCM16,
    SEMANTIC_VAD,
    SESSION_INPUT_SAMPLE_RATE_HZ,
    SESSION_UNIT_MS,
    AudioPort,
    SessionPorts,
)

TURN_DETECTION_STAGE = "turn_detection"
# note (ratish): the stage declares its default model here so that config imports
# stop short of the audio stack, whose silero import sets torch to one thread.
SMART_TURN_MODEL_REPO = "pipecat-ai/smart-turn-v3"
SESSION_PORTS = SessionPorts(
    input=(AudioPort(PCM16, SESSION_INPUT_SAMPLE_RATE_HZ),),
    output=None,
    unit_ms=SESSION_UNIT_MS,
    controls=frozenset({COMMIT, CONFIGURE}),
    hot_settings=frozenset({"turn_detection"}),
    turn_detection_types=(SEMANTIC_VAD,),
)


class TurnDetectionFactoryArgs(FactoryArgs):
    """The end-of-turn model and the eagerness of every session of the stage; a session's `turn_detection` overrides the eagerness."""

    model_path: str = SMART_TURN_MODEL_REPO
    "The Smart Turn ONNX file, its directory, or the Hugging Face repository holding it."
    eagerness: Literal["low", "medium", "high"] = "medium"
    max_open_sessions: int | None = Field(default=None, ge=1)
    max_concurrency: int | None = Field(default=None, ge=1)


class TurnDetectionStageConfig(StageConfig):
    factory: TurnDetectionFactoryArgs = Field(default_factory=TurnDetectionFactoryArgs)


def turn_detection_stage() -> TurnDetectionStageConfig:
    return TurnDetectionStageConfig(
        name=TURN_DETECTION_STAGE,
        process=TURN_DETECTION_STAGE,
        terminal=True,
        factory_path="sglang_omni.models.turn_detection.session.create_scheduler",
    )
