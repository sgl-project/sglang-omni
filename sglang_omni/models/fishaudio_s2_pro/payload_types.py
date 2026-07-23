# SPDX-License-Identifier: Apache-2.0
"""FishAudio S2-Pro pipeline state definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


@dataclass
class S2ProState(DeclarativeStateBase):
    """Per-request pipeline state for S2-Pro TTS."""

    sample_rate: int = 44100

    # -- From preprocessing ------------------------------------------------
    input_ids: Any = wire(None, codec="tensor_list")  # [seq_len] as list
    vq_mask_tokens: Any | None = wire(None, codec="tensor_list")  # [seq_len] bool
    vq_parts: Any | None = wire(None, codec="tensor_items")  # [num_codebooks, T_i]
    num_codebooks: int = 10
    codebook_size: int = 4096

    # -- Generation params -------------------------------------------------
    max_new_tokens: int = 1024
    temperature: float = 0.8
    top_p: float = 0.8
    top_k: int = 30
    repetition_penalty: float = 1.1
    ras_window: int = 16
    ras_temperature: float = 1.0
    ras_top_p: float = 0.9
    seed: int | None = None

    # -- From TTS engine ---------------------------------------------------
    output_codes: Any | None = wire(None, codec="tensor_restore")  # [nq+1, T]
    finish_reason: str | None = None

    # -- From vocoder ------------------------------------------------------
    audio_samples: Any | None = wire(None, codec="tensor_list")
