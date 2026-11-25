# SPDX-License-Identifier: Apache-2.0
"""Stable Audio 3 request options and the state handed to generation."""

import random

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sglang_omni.models.stable_audio3.model_config import (
    MAX_DURATION_SECONDS,
    MIN_DURATION_SECONDS,
)
from sglang_omni.models.stable_audio3.payload_types import StableAudio3State
from sglang_omni.proto.request import StagePayload

GENERATION_STAGE = "generation"
DEFAULT_DURATION_SECONDS = 120.0
# note (ruilin): seeds stay within a signed 64-bit integer, the range every client and torch.Generator.manual_seed accept.
SEED_LIMIT = 2**63
# note (ruilin): the T5Gemma tokenizer keeps 256 tokens of at most 31 characters, so a longer prompt loses text anyway; tokenizing 1 MB holds the generation batch for about 2 s.
MAX_PROMPT_CHARACTERS = 8192
# note (ruilin): the API layer answers 400 to errors that carry this prefix.
REQUEST_ERROR_PREFIX = "Stable Audio 3 request:"


class GenerationOptions(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)
    duration_seconds: float = Field(
        default=DEFAULT_DURATION_SECONDS,
        ge=MIN_DURATION_SECONDS,
        le=MAX_DURATION_SECONDS,
    )


def seed_in_range(seed: object) -> bool:
    return (
        isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed < SEED_LIMIT
    )


def build_state(payload: StagePayload) -> StableAudio3State:
    """An unset seed draws a random one, so repeated prompts give different music."""
    request = payload.request
    metadata = request.metadata or {}
    params = request.params
    prompt = request.inputs
    seed = params.get("seed")
    stage_params = params.get("stage_params") or {}
    # note (ruilin): only the speech endpoint validates its request, and /generate passes the caller's metadata through, so the prompt, seed and options are checked again before they reach a batch.
    if not isinstance(metadata.get("tts_params"), dict):
        raise ValueError(f"{REQUEST_ERROR_PREFIX} only /v1/audio/speech is served")
    elif not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{REQUEST_ERROR_PREFIX} the input must be a text prompt")
    elif len(prompt) > MAX_PROMPT_CHARACTERS:
        raise ValueError(
            f"{REQUEST_ERROR_PREFIX} the prompt is longer than {MAX_PROMPT_CHARACTERS} characters"
        )
    elif seed is not None and not seed_in_range(seed):
        raise ValueError(f"{REQUEST_ERROR_PREFIX} seed must be in [0, {SEED_LIMIT})")
    elif not isinstance(stage_params, dict):
        raise ValueError(f"{REQUEST_ERROR_PREFIX} stage_params must be an object")
    else:
        pass
    try:
        options = GenerationOptions.model_validate(
            stage_params.get(GENERATION_STAGE, {})
        )
    except ValidationError as error:
        detail = error.errors()[0]
        field = ".".join(map(str, detail["loc"]))
        raise ValueError(
            f"{REQUEST_ERROR_PREFIX} stage_params.{GENERATION_STAGE}.{field}: {detail['msg']}"
        ) from error
    return StableAudio3State(
        prompt=prompt,
        duration_seconds=options.duration_seconds,
        seed=random.randrange(SEED_LIMIT) if seed is None else seed,
    )
