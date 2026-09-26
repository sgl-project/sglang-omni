# SPDX-License-Identifier: Apache-2.0
"""The state one PersonaPlex request carries from stage to stage."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


@dataclass
class PersonaPlexState(DeclarativeStateBase):
    waveform: torch.Tensor | None = wire(None, codec="typed_tensor")
    num_samples: int = wire(0, codec="int")
    text_prompt_ids: list[int] = wire(default_factory=list, codec="list")
    carries_prompt: bool = wire(False, codec="bool")
    voice_waveform: torch.Tensor | None = wire(None, codec="typed_tensor")
    voice_path: str | None = wire(None, codec="str")
    voice_frames: int = wire(0, codec="int")
    user_codes: torch.Tensor | None = wire(None, codec="typed_tensor")
    voice_codes: torch.Tensor | None = wire(None, codec="typed_tensor")
    text_ids: list[int] = wire(default_factory=list, codec="list")
    codes: torch.Tensor | None = wire(None, codec="typed_tensor")
