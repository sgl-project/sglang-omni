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

# Known subdirectories where Ming repos store tokenizer files.
_TOKENIZER_SUBDIRS = ("talker/llm",)

# Fallback tokenizer source: Ming-flash-omni-Preview has tokenizer files
# at the repo root, used only as a last resort.
_TOKENIZER_FALLBACK = "inclusionAI/Ming-flash-omni-Preview"


def load_ming_tokenizer(model_path: str):
    """Load the Ming tokenizer, searching subdirectories before falling back.

    Ming HF repos may store tokenizer files in subdirectories (e.g.
    ``talker/llm/``) rather than at the repo root.  We try:
    1. AutoTokenizer from the root path (with trust_remote_code)
    2. PreTrainedTokenizerFast from the root path
    3. Known subdirectories (talker/llm) that ship tokenizer files
    4. Fallback to Ming-flash-omni-Preview repo (same vocab)
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    # Strategy 1: standard AutoTokenizer at root
    try:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except (OSError, ValueError):
        pass

    # Strategy 2: direct PreTrainedTokenizerFast at root
    try:
        return PreTrainedTokenizerFast.from_pretrained(model_path)
    except Exception:
        pass

    # Strategy 3: check known subdirectories (local snapshots only)
    resolved = resolve_model_path(model_path)
    for subdir in _TOKENIZER_SUBDIRS:
        subpath = Path(resolved) / subdir
        if (subpath / "tokenizer.json").is_file() or (
            subpath / "tokenizer_config.json"
        ).is_file():
            try:
                return AutoTokenizer.from_pretrained(
                    str(subpath), trust_remote_code=True
                )
            except (OSError, ValueError):
                pass
            try:
                return PreTrainedTokenizerFast.from_pretrained(str(subpath))
            except Exception:
                pass

    # Strategy 4: fallback repo with matching vocab
    logger.warning(
        "Tokenizer not found in %s, falling back to %s",
        model_path,
        _TOKENIZER_FALLBACK,
    )
    return PreTrainedTokenizerFast.from_pretrained(_TOKENIZER_FALLBACK)


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
