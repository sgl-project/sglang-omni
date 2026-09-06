# SPDX-License-Identifier: Apache-2.0
"""Fun-ASR's Torch/MPS compatibility runner and HF decoder installation."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers import Qwen3Config, Qwen3ForCausalLM

from sglang_omni.model_runner.audio_torch_mps import AudioTorchMpsModelRunner


class FunASRTorchMpsModelRunner(AudioTorchMpsModelRunner):
    model_name = "Fun-ASR"


def load_language_model(checkpoint: Path) -> Qwen3ForCausalLM:
    """Load only the HF text decoder, including the checkpoint's tied head."""
    config = Qwen3Config(
        **json.loads((checkpoint / "config.json").read_text())["text_config"]
    )
    # Build on CPU: rotary buffers must be materialized even when absent in weights.
    model = Qwen3ForCausalLM(config)
    weights = {}
    for weight_file in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(weight_file, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if name.startswith("model.language_model."):
                    weights[name.replace("model.language_model.", "model.", 1)] = (
                        reader.get_tensor(name)
                    )
                elif name == "lm_head.weight":
                    weights[name] = reader.get_tensor(name)
    if config.tie_word_embeddings and "model.embed_tokens.weight" in weights:
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    model.load_state_dict(weights, strict=True, assign=True)
    model.tie_weights()
    return model.eval()


def install_torch_mps_language_model(model: Any, model_path: str) -> None:
    from huggingface_hub import snapshot_download

    checkpoint = Path(model_path).expanduser()
    if not checkpoint.is_dir():
        checkpoint = Path(snapshot_download(model_path))
    parameter = next(model.language_model.parameters())
    device, dtype = parameter.device, parameter.dtype
    del parameter
    del model.language_model
    gc.collect()
    torch.mps.empty_cache()
    model.language_model = load_language_model(checkpoint).to(
        device=device, dtype=dtype
    )
