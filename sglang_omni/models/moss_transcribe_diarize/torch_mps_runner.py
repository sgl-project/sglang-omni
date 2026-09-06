# SPDX-License-Identifier: Apache-2.0
"""Torch/MPS compatibility runner for MOSS-Transcribe-Diarize."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from safetensors import safe_open
from transformers import Qwen3Config, Qwen3ForCausalLM

from sglang_omni.model_runner.audio_torch_mps import AudioTorchMpsModelRunner


def load_language_model(checkpoint: Path) -> Qwen3ForCausalLM:
    """Load the checkpoint's Hugging Face Qwen3 decoder."""
    config = Qwen3Config(
        **json.loads((checkpoint / "config.json").read_text())["text_config"]
    )
    model = Qwen3ForCausalLM(config)
    weights = {}
    for weight_file in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(weight_file, framework="pt", device="cpu") as reader:
            weight_names = reader.keys()
            for name in weight_names:
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


class MossTranscribeDiarizeTorchMpsModelRunner(AudioTorchMpsModelRunner):
    model_name = "MOSS-Transcribe-Diarize"
    encoder_window_batch_size = 8
    prefill_chunk_size = 4096

    def _validate_audio_positions(self, audio_positions: list[int]) -> None:
        del audio_positions

    def _get_audio_feature(self, item: Any, forward_batch: Any) -> torch.Tensor:
        feature_lengths = item.audio_feature_lengths
        outputs = []
        for start in range(0, item.feature.shape[0], self.encoder_window_batch_size):
            end = start + self.encoder_window_batch_size
            batch_features = item.feature[start:end]
            batch_item = SimpleNamespace(
                feature=batch_features,
                audio_feature_lengths=feature_lengths[start:end],
                audio_chunk_mapping=torch.zeros(
                    batch_features.shape[0], dtype=torch.long
                ),
            )
            outputs.append(
                self.model._get_audio_feature_uncached([batch_item], forward_batch)
            )
        return torch.cat(outputs, dim=0)

    def _assign_audio_features(
        self,
        input_embeddings: torch.Tensor,
        audio_features: torch.Tensor,
        audio_positions: list[int],
    ) -> None:
        positions = torch.tensor(audio_positions, device=input_embeddings.device)
        input_embeddings[0, positions, :] = audio_features[0]


__all__ = [
    "MossTranscribeDiarizeTorchMpsModelRunner",
    "install_torch_mps_language_model",
    "load_language_model",
]
