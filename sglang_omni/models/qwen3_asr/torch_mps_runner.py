# SPDX-License-Identifier: Apache-2.0
"""Torch MPS runner for Qwen3-ASR."""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers import Qwen3Config, Qwen3ForCausalLM

from sglang_omni.model_runner.audio_torch_mps import AudioTorchMpsModelRunner

logger = logging.getLogger(__name__)

_MROPE_ONLY_KEYS = frozenset({"interleaved", "mrope_interleaved", "mrope_section"})


def _resolve_model_path(model_path: str) -> Path:
    path = Path(model_path).expanduser()
    if path.is_dir():
        return path.resolve()

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_path))


def _text_config(model_path: Path) -> Qwen3Config:
    root_config = json.loads((model_path / "config.json").read_text())
    text_config = dict(root_config["thinker_config"]["text_config"])
    for field in ("rope_parameters", "rope_scaling"):
        rope_config = text_config.get(field)
        if isinstance(rope_config, dict):
            text_config[field] = {
                key: value
                for key, value in rope_config.items()
                if key not in _MROPE_ONLY_KEYS
            }
    return Qwen3Config(**text_config)


def _load_language_weights(
    language_model: Qwen3ForCausalLM,
    model_path: Path,
) -> None:
    expected = set(language_model.state_dict())
    state_dict = {}
    weight_files = sorted(model_path.glob("*.safetensors"))
    if not weight_files:
        raise ValueError(f"Qwen3-ASR checkpoint has no safetensors in {model_path}")
    for weight_file in weight_files:
        with safe_open(weight_file, framework="pt", device="cpu") as f:
            for checkpoint_name in f.keys():
                if checkpoint_name.startswith(
                    "thinker.model."
                ) or checkpoint_name.startswith("thinker.lm_head."):
                    model_name = checkpoint_name.removeprefix("thinker.")
                    if model_name in expected:
                        state_dict[model_name] = f.get_tensor(checkpoint_name)

    missing = expected - set(state_dict)
    if missing:
        raise ValueError(
            "Qwen3-ASR Torch MPS language checkpoint is incomplete: "
            f"{sorted(missing)[:10]}"
        )
    language_model.load_state_dict(state_dict, strict=True, assign=True)

    rotary = language_model.model.rotary_emb
    rope_config = language_model.config.rope_parameters
    if not isinstance(rope_config, dict):
        rope_config = language_model.config.rope_scaling
    rope_theta = float(rope_config["rope_theta"])
    head_dim = int(language_model.config.head_dim)
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    rotary.inv_freq = inv_freq
    rotary.original_inv_freq = inv_freq.clone()


def install_torch_mps_language_model(model: Any, model_path: str) -> None:
    """Replace SGLang's Torch-native LM with the pinned HF Torch implementation."""
    checkpoint = _resolve_model_path(model_path)
    old_parameter = next(model.language_model.parameters())
    device = old_parameter.device
    dtype = old_parameter.dtype

    del model.language_model
    gc.collect()
    torch.mps.empty_cache()

    with torch.device("meta"):
        language_model = Qwen3ForCausalLM(_text_config(checkpoint))
    _load_language_weights(language_model, checkpoint)
    model.language_model = language_model.eval().to(device=device, dtype=dtype)
    logger.info("Installed Qwen3-ASR Hugging Face Torch LM on %s (%s)", device, dtype)


class Qwen3ASRTorchMpsModelRunner(AudioTorchMpsModelRunner):
    """Torch audio and Hugging Face Qwen3 decoding for Qwen3-ASR."""

    model_name = "Qwen3-ASR"


__all__ = [
    "Qwen3ASRTorchMpsModelRunner",
    "install_torch_mps_language_model",
]
