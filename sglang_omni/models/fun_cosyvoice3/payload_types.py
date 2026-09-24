# SPDX-License-Identifier: Apache-2.0
"""Fun-CosyVoice3 pipeline state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from numpy.typing import ArrayLike

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire

if TYPE_CHECKING:
    import torch


@dataclass
class FunCosyVoice3State(DeclarativeStateBase):
    """Per-request state for Fun-CosyVoice3 generation."""

    sample_rate: int = wire(24000, codec="int")

    text: str = wire("", codec="str")
    language: str = wire("auto", codec="str_or")
    instructions: str | None = None
    ref_audio: object = None
    ref_text: str | None = None
    stream: bool = wire(False, codec="bool")
    speed: float = wire(1.0, codec="float")
    seed: int | None = None
    generation_kwargs: dict[str, Any] = wire(default_factory=dict, codec="dict")
    flow_embedding: ArrayLike | torch.Tensor | None = wire(None, codec="tensor_list")
    flow_prompt_speech_token: ArrayLike | torch.Tensor | None = wire(
        None, codec="tensor_list"
    )
    flow_prompt_speech_feat: ArrayLike | torch.Tensor | None = wire(
        None, codec="tensor_list"
    )
    audio_codes: ArrayLike | torch.Tensor | None = wire(None, codec="tensor_list")
    audio_samples: object = wire(None, codec="tensor_list")
