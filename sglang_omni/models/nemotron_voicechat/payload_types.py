from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire

INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 22_050


@dataclass
class NemotronVoiceChatState(DeclarativeStateBase):
    # 16 kHZ mono audio
    waveform: Any | None = wire(None, codec="typed_tensor")
    acoustic_frames: Any | None = wire(None, codec="typed_tensor")
    num_frames: int = wire(0, codec="int")
    text_ids: list = wire(default_factory=list, codec="list")
    codes: Any | None = wire(None, codec="typed_tensor")
