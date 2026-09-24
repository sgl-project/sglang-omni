# SPDX-License-Identifier: Apache-2.0
"""FishAudio S2-Pro pipeline state definition."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire

if TYPE_CHECKING:
    import torch


@dataclass
class S2ProState(DeclarativeStateBase):
    """Per-request pipeline state for S2-Pro TTS."""

    sample_rate: int = 44100

    # -- From preprocessing ------------------------------------------------
    input_ids: Any = wire(None, codec="tensor_list")  # [seq_len] as list
    vq_mask_tokens: torch.Tensor | list[bool] | None = wire(
        None, codec="tensor_list"
    )  # [seq_len] bool
    vq_parts: Sequence[torch.Tensor | list[list[int]]] | None = wire(
        None, codec="tensor_items"
    )  # [num_codebooks, T_i]
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
    output_codes: torch.Tensor | None = wire(None, codec="tensor_restore")  # [nq+1, T]
    finish_reason: str | None = None

    # -- From vocoder ------------------------------------------------------
    audio_samples: torch.Tensor | list[float] | None = wire(None, codec="tensor_list")
