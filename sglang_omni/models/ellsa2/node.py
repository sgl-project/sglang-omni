# SPDX-License-Identifier: Apache-2.0
"""The ELLSA 2 motion stage and what its node of a Session graph accepts and publishes."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from sglang_omni.config.schema import FactoryArgs, StageConfig
from sglang_omni.models.ellsa2.formats import SMPLH_FLAME_MOTION_PORT
from sglang_omni.proto.streams import (
    PCM16,
    SESSION_INPUT_SAMPLE_RATE_HZ,
    SESSION_UNIT_MS,
    AudioPort,
    SessionPorts,
)

MOTION_STAGE = "ellsa2_motion"
CHECKPOINT_REPO = "tsinghua-ee/ELLSA2"
DATA_REPO = "tsinghua-ee/ELLSA2_data"
# The release demo's default live-microphone style reference.
DEFAULT_STYLE = "r2d_00073"
SESSION_PORTS = SessionPorts(
    input=(AudioPort(PCM16, SESSION_INPUT_SAMPLE_RATE_HZ),),
    output=None,
    unit_ms=SESSION_UNIT_MS,
    plugin_output=(SMPLH_FLAME_MOTION_PORT,),
)


class MotionFactoryArgs(FactoryArgs):
    """The checkpoint, the style clip every session of the stage moves like, and the rate of the speech node's reply audio."""

    model_path: str = CHECKPOINT_REPO
    "The ELLSA 2 checkpoint: a Hub repository or a local copy of it."
    data_path: str = DATA_REPO
    "Where the style clip lives: the ELLSA 2 data Hub dataset or a local directory with its testset/style and testset/style_flame layout."
    style: str = DEFAULT_STYLE
    "The style clip's id: testset/style/<style>.npz and testset/style_flame/<style>_flame.npz."
    speech_sample_rate_hz: int | None = Field(default=None, gt=0)
    "The rate of the speech node's pcm_f32le reply audio."
    encoder_attn_implementation: Literal["sdpa", "flash_attention_2"] = "sdpa"
    "The audio encoder's attention: flash_attention_2 (the flash-attn package) is what the reference runs; sdpa needs nothing extra and rounds differently."
    cudnn_attention: bool = False
    "Let the backbone's attention use cuDNN as the reference does: needed to match the reference's codes bit for bit, at 7 Sessions an H100 instead of 8. Applies with kv_batch off."
    kv_batch: bool = True
    "Generate the ready blocks of all Sessions as one batch, each block's six steps decoding from a KV cache: codes that follow the reference's in distribution, not bit for bit. Off, the worker runs the reference's path, one block at a time with every step a full forward pass, for checks against the reference."
    cuda_graphs: bool = True
    "With kv_batch on a GPU, run each batch's five cached steps as one captured CUDA graph per batch size."
    max_open_sessions: int | None = Field(default=None, ge=1)


class MotionStageConfig(StageConfig):
    factory: MotionFactoryArgs = Field(default_factory=MotionFactoryArgs)


def motion_stage() -> MotionStageConfig:
    return MotionStageConfig(
        name=MOTION_STAGE,
        process=MOTION_STAGE,
        gpu=0,
        terminal=True,
        factory_path="sglang_omni.models.ellsa2.session.create_scheduler",
    )
