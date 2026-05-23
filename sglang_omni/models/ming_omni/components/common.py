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

_TOKENIZER_LOAD_ERRORS = (OSError, ValueError, KeyError)
# Fallback tokenizer source: Ming-flash-omni-Preview has tokenizer files
# at the repo root.  Ming-flash-omni-2.0 also ships a talker/llm tokenizer, but
# that is Qwen2-format and does not match the Bailing thinker vocabulary.
_TOKENIZER_FALLBACK = "inclusionAI/Ming-flash-omni-Preview"


def load_ming_tokenizer(model_path: str):
    """（wenyao）Load the Ming thinker tokenizer with a same-vocab fallback.

    Ming-flash-omni thinker uses the Bailing tokenizer.  Some Ming HF repos
    (e.g. Ming-flash-omni-2.0) omit thinker tokenizer files at the root, so we
    try:
    1. AutoTokenizer from the root path (with trust_remote_code)
    2. PreTrainedTokenizerFast from the root path
    3. Fallback to Ming-flash-omni-Preview repo (same thinker vocab)
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    # Strategy 1: standard AutoTokenizer at root
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return _attach_ming_tokenizer_compat(tokenizer)
    except _TOKENIZER_LOAD_ERRORS:
        pass

    # Strategy 2: direct PreTrainedTokenizerFast at root
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        return _attach_ming_tokenizer_compat(tokenizer)
    except Exception:
        pass

    # Strategy 3: fallback repo with matching thinker vocab
    logger.warning(
        "Tokenizer not found in %s, falling back to %s",
        model_path,
        _TOKENIZER_FALLBACK,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            _TOKENIZER_FALLBACK, trust_remote_code=True
        )
        return _attach_ming_tokenizer_compat(tokenizer)
    except _TOKENIZER_LOAD_ERRORS:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(_TOKENIZER_FALLBACK)
        return _attach_ming_tokenizer_compat(tokenizer)


def _attach_ming_tokenizer_compat(tokenizer):
    """（wenyao）Patch tokenizer fields assumed by SGLang's scheduler.

    Ming V0 relies on ``Req.eos_token_ids`` only.  Newer upstream SGLang also
    probes ``tokenizer.additional_stop_token_ids`` during finish checks, so keep
    the attribute present without adding extra stop ids that would change Ming's
    known-good stop behavior.
    """
    try:
        tokenizer.additional_stop_token_ids = None
    except Exception:
        pass
    return tokenizer


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
