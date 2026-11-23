# SPDX-License-Identifier: Apache-2.0
"""Stable Audio 3 request options and the state handed to generation."""

import random

from pydantic import BaseModel, ConfigDict, Field

from sglang_omni.models.stable_audio3.model_config import MAX_DURATION_SECONDS
from sglang_omni.models.stable_audio3.payload_types import StableAudio3State
from sglang_omni.proto.request import StagePayload

GENERATION_STAGE = "generation"
DEFAULT_DURATION_SECONDS = 120.0
SEED_LIMIT = 2**63


class GenerationOptions(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)
    duration_seconds: float = Field(
        default=DEFAULT_DURATION_SECONDS, gt=0, le=MAX_DURATION_SECONDS
    )


def build_state(payload: StagePayload) -> StableAudio3State:
    """An unset seed draws a random one, so repeated prompts give different music."""
    params = payload.request.params
    stage_params = params.get("stage_params") or {}
    options = GenerationOptions.model_validate(stage_params.get(GENERATION_STAGE, {}))
    seed = params.get("seed")
    return StableAudio3State(
        prompt=payload.request.inputs,
        duration_seconds=options.duration_seconds,
        seed=random.randrange(SEED_LIMIT) if seed is None else seed,
    )
