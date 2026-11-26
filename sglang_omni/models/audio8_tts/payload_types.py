# SPDX-License-Identifier: Apache-2.0
"""Sampling options and stage payloads for Audio8 TTS."""

from dataclasses import dataclass

import torch
from pydantic import BaseModel, Field, field_validator

from sglang_omni.models.audio8_tts.hf_config import SAMPLE_RATE_HZ
from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase

MAX_REQUEST_SEED = 2**64 - 1


class Audio8SamplingOptions(BaseModel):
    """Shared validation with defaults from the checkpoint's generation config."""

    temperature: float = Field(default=0.7, ge=0, allow_inf_nan=False, strict=True)
    top_p: float = Field(default=0.9, gt=0, le=1, strict=True)
    top_k: int = Field(default=50, ge=-1, strict=True)
    max_new_tokens: int = Field(default=512, ge=1, strict=True)
    seed: int | None = Field(default=None, ge=0, le=MAX_REQUEST_SEED, strict=True)

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, top_k: int) -> int:
        if top_k == 0:
            raise ValueError("top_k must be -1 or positive")
        else:
            return top_k


@dataclass(kw_only=True)
class Audio8State(DeclarativeStateBase):
    sample_rate: int = SAMPLE_RATE_HZ
    prompt_codes: torch.Tensor | None = None
    audio_codes: torch.Tensor | None = None
