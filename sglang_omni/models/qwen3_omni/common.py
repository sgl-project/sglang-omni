# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for Qwen3-Omni components."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch.nn as nn
from transformers import AutoConfig
from transformers.modeling_utils import no_init_weights
from transformers.utils.hub import cached_file


@lru_cache(maxsize=4)
def load_thinker_config(model_id: str) -> Any:
    try:
        config_path = cached_file(model_id, "config.json", local_files_only=True)
        cfg = AutoConfig.from_pretrained(
            str(config_path.parent),
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    return getattr(cfg, "thinker_config", cfg)


def instantiate_module(module_cls: type[nn.Module], config: Any) -> nn.Module:
    with no_init_weights():
        if hasattr(module_cls, "_from_config"):
            return module_cls._from_config(config)
        return module_cls(config)


@dataclass(frozen=True)
class Qwen3OmniSpec:
    """Lightweight spec extracted from the HF config."""

    model_id: str
    audio_token_id: int
    image_token_id: int
    spatial_merge_size: int

    @classmethod
    def from_model_id(cls, model_id: str) -> "Qwen3OmniSpec":
        thinker_cfg = load_thinker_config(model_id)
        vision_cfg = thinker_cfg.vision_config
        return cls(
            model_id=model_id,
            audio_token_id=int(thinker_cfg.audio_token_id),
            image_token_id=int(thinker_cfg.image_token_id),
            spatial_merge_size=int(vision_cfg.spatial_merge_size),
        )
