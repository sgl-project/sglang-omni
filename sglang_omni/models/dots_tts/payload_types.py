# SPDX-License-Identifier: Apache-2.0
"""Per-request state carried by the dots.tts Omni pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


@dataclass
class DotsTTSState(DeclarativeStateBase):
    """Only cross-stage state; scheduler/runner state stays in Omni."""

    sample_rate: int = wire(48000, codec="int")
    prompt_audio_path: str | None = None
    use_prompt_prefill: bool = wire(False, codec="bool")
    speaker_scale: float = wire(1.5, codec="float")
    ode_method: str = wire("euler", codec="str_or")
    num_steps: int = wire(4, codec="int_or")
    guidance_scale: float = wire(1.2, codec="float")
    seed: int | None = wire(None, codec="opt_int")
    max_new_tokens: int | None = wire(None, codec="opt_int")
    stream: bool = wire(False, codec="bool")
    eos_threshold: float = wire(0.8, codec="float")
    generation_schedule: torch.Tensor | None = wire(None, codec="typed_tensor")
    audio_span_token_ids: list[int] = wire(default_factory=list, codec="list")
    latent_patch_size: int = wire(4, codec="int_or")
    vocab_size: int = wire(0, codec="int")
    prompt_latents: torch.Tensor | None = wire(None, codec="typed_tensor")
    speaker_embedding: torch.Tensor | None = wire(None, codec="typed_tensor")
    generated_latents: torch.Tensor | None = wire(None, codec="typed_tensor")
    finish_reason: str | None = None


def load_dots_tts_state(payload: StagePayload) -> DotsTTSState:
    return DotsTTSState.from_dict(payload.data)


def store_dots_tts_state(payload: StagePayload, state: DotsTTSState) -> StagePayload:
    payload.data = state.to_dict()
    return payload


__all__ = ["DotsTTSState", "load_dots_tts_state", "store_dots_tts_state"]
