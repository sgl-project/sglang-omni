# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 pipeline state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.models.voxcpm2 import constants as C
from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


@dataclass
class VoxCPM2State(DeclarativeStateBase):
    """Request fields shared by preprocessing, reference encode, engine and vocoder."""

    sample_rate: int = wire(C.SAMPLE_RATE, codec="int")
    out_sample_rate: int = wire(C.OUT_SAMPLE_RATE, codec="int")

    prompt_text: str = wire("", codec="str")
    prompt_audio: str = wire("", codec="str")
    reference_audio: str = wire("", codec="str")

    text_token: Any | None = wire(None, codec="typed_tensor")
    target_text_length: int = wire(0, codec="int")

    ref_latents: Any | None = wire(None, codec="typed_tensor")
    prompt_latents: Any | None = wire(None, codec="typed_tensor")

    patch_size: int = wire(C.PATCH_SIZE, codec="int_or")
    feat_dim: int = wire(C.FEAT_DIM, codec="int_or")
    inference_timesteps: int = wire(C.DEFAULT_INFERENCE_TIMESTEPS, codec="int_or")
    cfg_value: float = wire(C.DEFAULT_CFG_VALUE, codec="float")
    min_len: int = wire(C.DEFAULT_MIN_LEN, codec="int_or")
    max_len: int = wire(C.DEFAULT_MAX_LEN, codec="int_or")
    streaming_prefix_len: int = wire(C.DEFAULT_STREAMING_PREFIX_LEN, codec="int_or")
    seed: int | None = wire(None, codec="opt_int")
    stream: bool = wire(False, codec="bool")

    generated_latents: Any | None = wire(None, codec="typed_tensor")
    context_len: int = wire(0, codec="int")
    finish_reason: str | None = None


__all__ = ["VoxCPM2State"]
