# SPDX-License-Identifier: Apache-2.0
"""Per-request pipeline state for Higgs TTS.

Carried between stages via :class:`sglang_omni.proto.StagePayload.data`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


@dataclass
class HiggsTtsState(DeclarativeStateBase):
    """Per-request state threaded through preprocessing -> audio_encoder ->
    tts_engine -> vocoder. Fields populate lazily so a deserialised state is
    valid at any stage boundary."""

    sample_rate: int = 24000

    # preprocessing / audio_encoder
    prompt_token_ids: list[int] = wire(default_factory=list, codec="list")
    reference_codes_delayed: list[list[int]] | None = None
    target_text: str | None = None
    reference_text: str | None = None
    reference_waveform: Any | None = None  # mono 24 kHz [1, 1, L] torch.Tensor
    reference_code_cache_key: str | None = None
    uploaded_voice_name: str | None = None
    uploaded_voice_created_at: int | None = None

    # voice fusion: when set, this request is conditioned on multiple reference
    # voices blended at the output-distribution layer (see fusion.py). Each entry
    # is ``{"codes_delayed": list[list[int]], "weight": float,
    # "reference_text": str | None}`` — one per reference voice. The request
    # builder fans this out into N sibling requests sharing a fusion group id;
    # ``reference_codes_delayed`` above stays None for the fusion case (each
    # sibling carries its own slice). ``None``/len<2 → ordinary single-voice path.
    fusion_refs: list[dict[str, Any]] | None = None

    # Reference-space fusion build parameters (mode "reference", the default):
    # ``{"cal_text": str, "final_prompt_prefix": list[int],
    # "final_prompt_suffix": list[int]}``. The engine assembles the real
    # request's prompt as ``prefix + [-100] * len(hybrid_ref) + suffix`` once
    # the hybrid reference exists (its frame count is unknown before then).
    # ``None`` in the legacy "logits" sibling-blend mode.
    fusion_build: dict[str, Any] | None = None

    num_codebooks: int = 8
    codebook_size: int = 1026  # 1024 data + <|boc|> + <|eoc|>

    # generation params
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None

    # RL rollout controls
    return_logprob: bool = wire(False, emit="truthy", codec="bool")
    return_omni_rollout: bool = wire(False, emit="truthy", codec="bool")

    # tts_engine
    output_codes_delayed: list[list[int]] | None = None
    omni_rollout: dict[str, Any] | None = None

    # vocoder
    audio_samples: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        if self.audio_samples is None:
            data.pop("sample_rate", None)
        return data


__all__ = ["HiggsTtsState"]
