# SPDX-License-Identifier: Apache-2.0
"""Per-request state carried by the dots.tts Omni pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    interleaved: bool = wire(False, codec="bool")
    streaming_schedule: dict[str, Any] | None = None
    audio_span_token_ids: list[int] = wire(default_factory=list, codec="list")
    latent_patch_size: int = wire(4, codec="int_or")
    vocab_size: int = wire(0, codec="int")
    prompt_latents: torch.Tensor | None = wire(None, codec="typed_tensor")
    speaker_embedding: torch.Tensor | None = wire(None, codec="typed_tensor")
    generated_latents: torch.Tensor | None = wire(None, codec="typed_tensor")
    finish_reason: str | None = None


def materialize_streaming_schedule(state: DotsTTSState) -> None:
    """Build the paired STTS schedule once prompt patch count is known."""
    spec = state.streaming_schedule
    if spec is None:
        return
    from dots_tts.runtime_double_streaming import build_interleave_token_sequence

    prompt_patches = state.prompt_latents
    if prompt_patches is None:
        raise ValueError("dots.tts STTS requires reference audio with its transcript")
    prompt_span_count = int(prompt_patches.shape[1]) // int(state.latent_patch_size) + 1
    target_span_count = int(spec["max_audio_patches"]) - prompt_span_count
    if target_span_count <= 0:
        raise ValueError(
            "dots.tts STTS max_generate_length must exceed the reference audio "
            f"span count ({prompt_span_count})"
        )
    cadence = {
        "interleave_mode": spec["interleave_mode"],
        "initial_lookahead": spec["initial_lookahead"],
        "ta_per_tta": spec["ta_per_tta"],
        "warmup_ta": spec["warmup_ta"],
    }
    audio_span_id = int(spec["audio_span_id"])
    prompt = build_interleave_token_sequence(
        text_tokens=list(spec["prompt_text_ids"]),
        audio_tokens=[audio_span_id] * prompt_span_count + [int(spec["audio_end_id"])],
        text_cond_end_id=int(spec["text_cond_end_id"]),
        **cadence,
    )
    target = build_interleave_token_sequence(
        text_tokens=list(spec["text_ids"]),
        audio_tokens=[audio_span_id] * target_span_count,
        text_cond_end_id=int(spec["text_cond_end_id"]),
        **cadence,
    )
    schedule = [*spec["prefix_ids"], *prompt, *spec["prefix_ids"], *target]
    if len(schedule) > int(spec["max_sequence_length"]):
        raise ValueError(
            "dots.tts STTS schedule exceeds context length "
            f"{spec['max_sequence_length']}"
        )
    state.generation_schedule = torch.tensor([schedule], dtype=torch.long)


def load_dots_tts_state(payload: StagePayload) -> DotsTTSState:
    return DotsTTSState.from_dict(payload.data)


def store_dots_tts_state(payload: StagePayload, state: DotsTTSState) -> StagePayload:
    payload.data = state.to_dict()
    return payload


__all__ = ["DotsTTSState", "load_dots_tts_state", "store_dots_tts_state"]
