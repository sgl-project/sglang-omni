# SPDX-License-Identifier: Apache-2.0
"""Stable Audio 3 request options and the state handed to generation."""

import random

from pydantic import BaseModel, ConfigDict, Field

from sglang_omni.models.stable_audio3.model_config import MAX_DURATION_SECONDS
from sglang_omni.models.stable_audio3.payload_types import StableAudio3State
from sglang_omni.proto.request import StagePayload

GENERATION_STAGE = "generation"
DEFAULT_DURATION_SECONDS = 120.0
# note (ruilin): about one latent frame (4096 samples); without a floor a tiny duration rounds to an empty WAV.
MIN_DURATION_SECONDS = 0.1
# note (ruilin): seeds stay within a signed 64-bit integer, the range every client and torch.Generator.manual_seed accept.
SEED_LIMIT = 2**63


class GenerationOptions(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)
    duration_seconds: float = Field(
        default=DEFAULT_DURATION_SECONDS,
        ge=MIN_DURATION_SECONDS,
        le=MAX_DURATION_SECONDS,
    )


def build_state(payload: StagePayload) -> StableAudio3State:
    """An unset seed draws a random one, so repeated prompts give different music."""
    request = payload.request
    metadata = request.metadata or {}
    # note (ruilin): only the speech endpoint validates the prompt and seed, so a /generate or chat request must not reach a batch.
    if not isinstance(metadata.get("tts_params"), dict):
        raise ValueError("Stable Audio 3 serves /v1/audio/speech requests only")
    else:
        pass
    params = request.params
    stage_params = params.get("stage_params") or {}
    options = GenerationOptions.model_validate(stage_params.get(GENERATION_STAGE, {}))
    seed = params.get("seed")
    return StableAudio3State(
        prompt=request.inputs,
        duration_seconds=options.duration_seconds,
        seed=random.randrange(SEED_LIMIT) if seed is None else seed,
    )
