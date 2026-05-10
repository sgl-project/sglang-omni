# SPDX-License-Identifier: Apache-2.0
"""Hugging Face helper utilities."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch.nn as nn
from transformers import AutoConfig

try:
    from transformers.initialization import no_init_weights
except ImportError:
    from transformers.modeling_utils import no_init_weights

from transformers.utils.hub import cached_file

# ---------------------------------------------------------------------------
# Architecture resolution helpers
# ---------------------------------------------------------------------------

_CONFIG_MODEL_TYPE_TO_ARCH = {
    "voxtral_tts": "VoxtralTTSForConditionalGeneration",
    "fish_qwen3_omni": "FishQwen3OmniForCausalLM",
}


def architecture_from_hf_config(hf_config: Any) -> str | None:
    """Prefer HF ``architectures``; fall back to ``model_type`` when needed."""
    archs = getattr(hf_config, "architectures", None)
    if archs:
        for a in archs:
            if a:
                return a
    mt = getattr(hf_config, "model_type", None)
    if mt and mt in _CONFIG_MODEL_TYPE_TO_ARCH:
        return _CONFIG_MODEL_TYPE_TO_ARCH[mt]
    return None


def load_mistral_params_json(model_path: str) -> dict | None:
    """Load Mistral-format ``params.json`` from a local dir or Hugging Face hub id.
    Official Voxtral TTS checkpoints ship without ``config.json``; architecture is
    only indicated by ``model_type`` inside ``params.json`` (see Hub repo files).
    """
    params_path = os.path.join(model_path, "params.json")
    if os.path.isfile(params_path):
        with open(params_path) as f:
            return json.load(f)
    # Local directory without params — do not treat as hub repo id
    if os.path.isdir(model_path):
        return None
    try:
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(repo_id=model_path, filename="params.json")
        with open(cached) as f:
            return json.load(f)
    except Exception:
        return None


def try_resolve_arch_from_mistral_config(model_path: str) -> str | None:
    """Resolve architecture from Mistral-format params.json (local or Hub)."""
    params = load_mistral_params_json(model_path)
    if params is None:
        return None
    model_type = params.get("model_type", "")
    return _CONFIG_MODEL_TYPE_TO_ARCH.get(model_type)


def try_resolve_arch_from_raw_config(model_path: str) -> str | None:
    """Resolve architecture by reading raw ``config.json`` as plain JSON.

    This is useful when ``AutoConfig.from_pretrained`` fails (e.g. because the
    model requires ``trust_remote_code=True`` and the custom Python config
    module is unavailable).  We parse the JSON directly to extract
    ``architectures`` or map ``model_type``.
    """
    raw: dict | None = None

    # Try local path first
    local_config = os.path.join(model_path, "config.json")
    if os.path.isfile(local_config):
        with open(local_config) as f:
            raw = json.load(f)
    elif not os.path.isdir(model_path):
        # Treat as a Hub repo id — download config.json
        try:
            from huggingface_hub import hf_hub_download

            cached = hf_hub_download(repo_id=model_path, filename="config.json")
            with open(cached) as f:
                raw = json.load(f)
        except Exception:
            return None

    if raw is None:
        return None

    # Prefer architectures list
    archs = raw.get("architectures")
    if archs:
        for a in archs:
            if a:
                return a

    # Fall back to model_type mapping
    mt = raw.get("model_type")
    if mt and mt in _CONFIG_MODEL_TYPE_TO_ARCH:
        return _CONFIG_MODEL_TYPE_TO_ARCH[mt]

    return None


# ---------------------------------------------------------------------------
# HF config loading
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def load_hf_config(
    model_path: str,
    *,
    trust_remote_code: bool = True,
    local_files_only: bool = True,
) -> Any:
    """Load the HF config, preferring the local cache."""
    try:
        config_path = cached_file(
            model_path, "config.json", local_files_only=local_files_only
        )
        cfg = AutoConfig.from_pretrained(
            str(Path(config_path).parent),
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    except Exception:
        cfg = AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
    return cfg


def instantiate_module(module_cls: type[nn.Module], config: Any) -> nn.Module:
    """Instantiate a module without allocating its parameters."""
    with no_init_weights():
        if hasattr(module_cls, "_from_config"):
            return module_cls._from_config(config)
        return module_cls(config)
