# SPDX-License-Identifier: Apache-2.0
"""AuK pipeline state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.models.auk import constants as C
from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


@dataclass
class AuKState(DeclarativeStateBase):
    """Request fields shared by preprocessing, conditioning, sampling and decode."""

    sample_rate: int = wire(C.SAMPLE_RATE, codec="int")

    instruction: str = wire("", codec="str")
    ref_audio: Any | None = None
    qwen_audio: Any | None = None
    ref_seconds: float = wire(0.0, codec="float")

    gen_frames: int = wire(0, codec="int")
    seed: int | None = None
    conditioning: Any | None = wire(None, codec="tensor_cpu")
    text_mask: Any | None = wire(None, codec="tensor_cpu")
    ref_latent: Any | None = wire(None, codec="tensor_cpu")
    ref_length: int = wire(0, codec="int")
    latent: Any | None = wire(None, codec="tensor_cpu")

    @property
    def gen_seconds(self) -> float:
        return self.gen_frames * C.VAE_DOWNSAMPLE_RATE / self.sample_rate
