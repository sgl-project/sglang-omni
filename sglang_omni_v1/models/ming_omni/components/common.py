# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for Ming-Omni components."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from sglang_omni_v1.models.ming_omni.hf_config import (
    AudioConfig,
    BailingMoeV2LLMConfig,
    MingOmniConfig,
)
from sglang_omni_v1.models.weight_loader import resolve_model_path

logger = logging.getLogger(__name__)

# Known subdirectories where Ming repos store tokenizer files.
_TOKENIZER_SUBDIRS = ("talker/llm",)
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")

# Fallback tokenizer source: Ming-flash-omni-Preview has tokenizer files
# at the repo root, used only as a last resort.
_TOKENIZER_FALLBACK = "inclusionAI/Ming-flash-omni-Preview"


def _resolve_local_tokenizer_subdir(model_path: str) -> Path | None:
    base = Path(model_path)
    if not base.exists():
        return None
    for subdir in _TOKENIZER_SUBDIRS:
        subpath = base / subdir
        if any((subpath / filename).is_file() for filename in _TOKENIZER_FILES):
            return subpath
    return None


def _download_tokenizer_subdir_from_hub(model_path: str) -> Path | None:
    for subdir in _TOKENIZER_SUBDIRS:
        for filename in _TOKENIZER_FILES:
            try:
                cached = hf_hub_download(
                    repo_id=model_path,
                    filename=filename,
                    subfolder=subdir,
                )
            except Exception:
                continue
            return Path(cached).parent
    return None


def load_ming_tokenizer(model_path: str):
    """Load the Ming tokenizer, searching subdirectories before falling back.

    Ming HF repos may store tokenizer files in subdirectories (e.g.
    ``talker/llm/``) rather than at the repo root.  We try:
    1. AutoTokenizer from the root path (with trust_remote_code)
    2. PreTrainedTokenizerFast from the root path
    3. Known subdirectories (talker/llm), local first then selective Hub probe
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

    # Strategy 3a: check known subdirectories for local paths
    local_subpath = _resolve_local_tokenizer_subdir(model_path)
    if local_subpath is not None:
        try:
            return AutoTokenizer.from_pretrained(
                str(local_subpath), trust_remote_code=True
            )
        except (OSError, ValueError):
            pass
        try:
            return PreTrainedTokenizerFast.from_pretrained(str(local_subpath))
        except Exception:
            pass

    # Strategy 3b: probe known subdirectories on Hub by downloading only
    # tokenizer files, avoiding a full snapshot download.
    remote_subpath = _download_tokenizer_subdir_from_hub(model_path)
    if remote_subpath is not None:
        try:
            return AutoTokenizer.from_pretrained(
                str(remote_subpath), trust_remote_code=True
            )
        except (OSError, ValueError):
            pass
        try:
            return PreTrainedTokenizerFast.from_pretrained(str(remote_subpath))
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
