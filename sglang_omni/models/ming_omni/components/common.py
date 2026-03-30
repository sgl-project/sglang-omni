# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for Ming-Omni components."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from sglang_omni.models.ming_omni.hf_config import (
    AudioConfig,
    BailingMoeV2LLMConfig,
    MingOmniConfig,
)
from sglang_omni.models.weight_loader import resolve_model_path

logger = logging.getLogger(__name__)


def load_ming_config(model_path: str) -> MingOmniConfig:
    """Load Ming-Omni configuration from model checkpoint."""
    resolved = resolve_model_path(model_path)
    config_path = Path(resolved) / "config.json"
    with open(config_path) as f:
        raw = json.load(f)
    return MingOmniConfig.from_dict(raw)


def load_llm_config(model_path: str) -> BailingMoeV2LLMConfig:
    """Load just the LLM config from the Ming-Omni checkpoint."""
    config = load_ming_config(model_path)
    return config.llm_config


def load_audio_config(model_path: str) -> AudioConfig:
    """Load just the audio config from the Ming-Omni checkpoint."""
    config = load_ming_config(model_path)
    return config.audio_config


@dataclass(frozen=True)
class MingOmniSpec:
    """Lightweight spec extracted from HF config for component factories."""

    model_path: str
    audio_patch_token_id: int  # <audioPatch> token ID
    hidden_size: int  # LLM hidden size for projection dimensions

    @classmethod
    def from_config(cls, model_path: str, config: MingOmniConfig) -> "MingOmniSpec":
        # The audioPatch token ID needs to be read from the tokenizer
        # For now use a sentinel; will be resolved at preprocessor init time
        return cls(
            model_path=model_path,
            audio_patch_token_id=-1,  # resolved from tokenizer
            hidden_size=config.llm_config.hidden_size,
        )
