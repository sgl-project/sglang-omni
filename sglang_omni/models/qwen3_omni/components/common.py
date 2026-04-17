# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for Qwen3-Omni components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.utils import load_hf_config


def load_thinker_config(model_path: str) -> Any:
    cfg = load_hf_config(model_path, trust_remote_code=True, local_files_only=True)
    return cfg.thinker_config


@dataclass(frozen=True)
class Qwen3OmniSpec:
    """Lightweight spec extracted from the HF config."""

    model_path: str
    audio_token_id: int
    image_token_id: int
    spatial_merge_size: int

    @classmethod
    def from_model_path(cls, model_path: str) -> "Qwen3OmniSpec":
        thinker_cfg = load_thinker_config(model_path)
        vision_cfg = thinker_cfg.vision_config
        return cls(
            model_path=model_path,
            audio_token_id=int(thinker_cfg.audio_token_id),
            image_token_id=int(thinker_cfg.image_token_id),
            spatial_merge_size=int(vision_cfg.spatial_merge_size),
        )
