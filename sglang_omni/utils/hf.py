# SPDX-License-Identifier: Apache-2.0
"""Hugging Face helper utilities."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoConfig

try:
    from transformers.initialization import no_init_weights
except ImportError:
    from transformers.modeling_utils import no_init_weights

from transformers.utils.hub import cached_file

_CONFIG_MODEL_TYPE_TO_ARCH = {
    "fish_qwen3_omni": "FishQwen3OmniForCausalLM",
    "moss_tts_delay": "MossTTSDelayModel",
    "moss_tts_delay_with_codec": "MossTTSDelayWithCodec",
    "moss_tts_local": "MossTTSLocalModel",
    "dots_tts": "DotsTTSForConditionalGeneration",
    "qwen3_tts": "Qwen3TTSForConditionalGeneration",
    "voxtral_tts": "VoxtralTTSForConditionalGeneration",
    "zonos2": "Zonos2ForCausalLM",
}

_COSYVOICE3_LAYOUT_MARKER = "cosyvoice3.yaml"
_COSYVOICE3_ARCHITECTURE = "FunCosyVoice3SGLangModel"
_AUK_ARCHITECTURE = "AuKForConditionalGeneration"
_AUK_CONFIG_NAMES = ("config.yaml", "config.yml")
_AUK_MODEL_NAMES = frozenset({"auk", "auk-flash"})
_AUK_WEIGHT_MARKERS = ("auk_base.safetensors", "auk_flash.safetensors")


def architecture_from_hf_config(hf_config: Any) -> str | None:
    """Prefer HF architectures; fall back to architecture/model_type."""
    archs = getattr(hf_config, "architectures", None)
    if archs:
        for a in archs:
            if a:
                return a
    arch = getattr(hf_config, "architecture", None)
    if arch:
        return arch
    mt = getattr(hf_config, "model_type", None)
    if mt and mt in _CONFIG_MODEL_TYPE_TO_ARCH:
        return _CONFIG_MODEL_TYPE_TO_ARCH[mt]
    return None


def load_mistral_params_json(
    model_path: str, revision: str | None = None
) -> dict | None:
    """Load Mistral-format ``params.json`` from a local dir or Hugging Face hub id.
    Official Voxtral TTS checkpoints ship without ``config.json``; architecture is
    only indicated by ``model_type`` inside ``params.json`` (see Hub repo files).
    """
    params_path = os.path.join(model_path, "params.json")
    if os.path.isfile(params_path):
        with open(params_path) as f:
            return json.load(f)
    if os.path.isdir(model_path):
        return None
    try:
        cached = hf_hub_download(
            repo_id=model_path, filename="params.json", revision=revision
        )
        with open(cached) as f:
            return json.load(f)
    except Exception:
        return None


def try_resolve_arch_from_mistral_config(
    model_path: str, revision: str | None = None
) -> str | None:
    """Resolve architecture from Mistral-format params.json (local or Hub)."""
    params = load_mistral_params_json(model_path, revision=revision)
    if params is None:
        return None
    model_type = params.get("model_type", "")
    return _CONFIG_MODEL_TYPE_TO_ARCH.get(model_type)


def try_resolve_arch_from_raw_config(
    model_path: str, revision: str | None = None
) -> str | None:
    """Resolve architecture by reading raw ``config.json`` as plain JSON.

    This is useful when ``AutoConfig.from_pretrained`` fails (e.g. because the
    model requires ``trust_remote_code=True`` and the custom Python config
    module is unavailable).  We parse the JSON directly to extract
    ``architectures`` or map ``model_type``.
    """
    raw: dict | None = None

    local_config = os.path.join(model_path, "config.json")
    if os.path.isfile(local_config):
        with open(local_config) as f:
            raw = json.load(f)
    elif not os.path.isdir(model_path):
        try:
            cached = hf_hub_download(
                repo_id=model_path, filename="config.json", revision=revision
            )
            with open(cached) as f:
                raw = json.load(f)
        except Exception:
            return None

    if raw is None:
        return None

    archs = raw.get("architectures")
    if archs:
        for a in archs:
            if a:
                return a
    arch = raw.get("architecture")
    if arch:
        return arch

    mt = raw.get("model_type")
    if mt and mt in _CONFIG_MODEL_TYPE_TO_ARCH:
        return _CONFIG_MODEL_TYPE_TO_ARCH[mt]

    return None


def try_resolve_arch_from_cosyvoice3_layout(
    model_path: str, revision: str | None = None
) -> str | None:
    """Resolve Fun-CosyVoice3 from the official checkpoint layout."""
    marker_path = os.path.join(model_path, _COSYVOICE3_LAYOUT_MARKER)
    if os.path.isfile(marker_path):
        return _COSYVOICE3_ARCHITECTURE
    if os.path.isdir(model_path):
        return None
    try:
        hf_hub_download(
            repo_id=model_path,
            filename=_COSYVOICE3_LAYOUT_MARKER,
            revision=revision,
        )
    except Exception:
        return None
    return _COSYVOICE3_ARCHITECTURE


def _auk_architecture_from_config(path: str) -> str | None:
    """Return the AuK architecture if the file names AuK or AuK-Flash."""
    import yaml

    try:
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(raw, dict):
        return None
    model = raw.get("model") if isinstance(raw.get("model"), dict) else raw
    if not isinstance(model, dict):
        return None
    name = model.get("name") or raw.get("name")
    if not isinstance(name, str):
        return None
    if name.lower() in _AUK_MODEL_NAMES:
        return _AUK_ARCHITECTURE
    return None


def try_resolve_arch_from_auk_layout(
    model_path: str, revision: str | None = None
) -> str | None:
    """Resolve AuK from the released OmegaConf layout."""
    for filename in _AUK_CONFIG_NAMES:
        local = os.path.join(model_path, filename)
        if os.path.isfile(local):
            return _auk_architecture_from_config(local)
    if os.path.isdir(model_path):
        for marker in _AUK_WEIGHT_MARKERS:
            if os.path.isfile(os.path.join(model_path, marker)):
                return _AUK_ARCHITECTURE
        return None
    for filename in _AUK_CONFIG_NAMES:
        try:
            cached = hf_hub_download(
                repo_id=model_path, filename=filename, revision=revision
            )
        except Exception:
            continue
        architecture = _auk_architecture_from_config(cached)
        if architecture is not None:
            return architecture
    return None


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
