# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for LLaDA2-Uni components."""

from __future__ import annotations

from typing import Any

from sglang_omni.utils import load_hf_config


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
