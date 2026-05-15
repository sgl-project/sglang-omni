# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for LLaDA2-Uni components."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.utils import load_hf_config

logger = logging.getLogger(__name__)


def resolve_local_model_dir(model_path: str) -> str:
    """Resolve a local model directory without eagerly hydrating full snapshots."""
    path = Path(model_path)
    if path.exists():
        return str(path)

    try:
        return str(resolve_model_path(model_path, local_files_only=True))
    except (FileNotFoundError, OSError) as exc:
        logger.warning(
            "Local-only model resolution failed for %s; falling back to hub id",
            model_path,
            exc_info=exc,
        )
        return model_path


def load_llada2_tokenizer(model_path: str):
    """Load LLaDA2 tokenizer from model checkpoint."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    try:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except (OSError, ValueError):
        pass

    return PreTrainedTokenizerFast.from_pretrained(model_path)


def load_llada2_config(model_path: str) -> Any:
    """Load LLaDA2 configuration from model checkpoint."""
    return load_hf_config(model_path, trust_remote_code=True, local_files_only=True)
