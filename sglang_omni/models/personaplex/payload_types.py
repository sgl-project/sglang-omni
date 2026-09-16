# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


@dataclass
class PersonaPlexState(DeclarativeStateBase):
    waveform: Any | None = wire(None, codec="typed_tensor")
    text_prompt_ids: list = wire(default_factory=list, codec="list")
    voice_waveform: Any | None = wire(None, codec="typed_tensor")
    voice_embeddings: Any | None = wire(None, codec="typed_tensor")
    voice_tail_codes: Any | None = wire(None, codec="typed_tensor")
    voice_frames: int = wire(0, codec="int")
    user_codes: Any | None = wire(None, codec="typed_tensor")
    voice_codes: Any | None = wire(None, codec="typed_tensor")
    text_ids: list = wire(default_factory=list, codec="list")
    codes: Any | None = wire(None, codec="typed_tensor")
