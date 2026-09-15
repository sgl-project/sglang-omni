# SPDX-License-Identifier: Apache-2.0
"""Torch/MPS compatibility runner for MOSS-Transcribe-Diarize."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from safetensors import safe_open
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from transformers import Qwen3Config, Qwen3ForCausalLM

from sglang_omni.model_runner.audio_torch_mps import AudioTorchMpsModelRunner
from sglang_omni.models.moss_transcribe_diarize.sglang_model import (
    MossTranscribeDiarizeForConditionalGeneration,
)


def load_language_model(checkpoint: Path) -> Qwen3ForCausalLM:
    """Load the checkpoint's Hugging Face Qwen3 decoder."""
    config = Qwen3Config(
        **json.loads((checkpoint / "config.json").read_text())["text_config"]
    )
    model = Qwen3ForCausalLM(config)
    weights: dict[str, torch.Tensor] = {}
    for weight_file in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(weight_file, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if name.startswith("model.language_model."):
                    weights[name.replace("model.language_model.", "model.", 1)] = (
                        reader.get_tensor(name)
                    )
                elif name == "lm_head.weight":
                    weights[name] = reader.get_tensor(name)
                else:
                    pass
    if config.tie_word_embeddings:
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    else:
        pass
    model.load_state_dict(weights, strict=True, assign=True)
    model.tie_weights()
    return model.eval()


def install_torch_mps_language_model(
    model: MossTranscribeDiarizeForConditionalGeneration, checkpoint_dir: str
) -> None:
    """Replace the decoder using the builder's already resolved checkpoint."""
    parameter = next(model.language_model.parameters())
    device, dtype = parameter.device, parameter.dtype
    del parameter
    del model.language_model
    gc.collect()
    torch.mps.empty_cache()
    model.language_model = load_language_model(Path(checkpoint_dir)).to(
        device=device, dtype=dtype
    )


class MossTranscribeDiarizeTorchMpsModelRunner(AudioTorchMpsModelRunner):
    model_name = "MOSS-Transcribe-Diarize"
    encoder_window_batch_size = 8
    prefill_chunk_size = 4096
    requires_contiguous_audio_positions = False

    def get_audio_feature(
        self, audio_item: MultimodalDataItem, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        """Encode long audio in bounded window batches."""
        feature_lengths = audio_item.audio_feature_lengths
        outputs: list[torch.Tensor] = []
        for start in range(
            0, audio_item.feature.shape[0], self.encoder_window_batch_size
        ):
            end = start + self.encoder_window_batch_size
            batch_features = audio_item.feature[start:end]
            batch_item = MultimodalDataItem(
                modality=Modality.AUDIO,
                feature=batch_features,
                model_specific_data={
                    "audio_feature_lengths": feature_lengths[start:end]
                },
            )
            outputs.append(
                self.model.get_audio_feature_uncached([batch_item], forward_batch)
            )
        return torch.cat(outputs, dim=0)

    def assign_audio_features(
        self,
        input_embeddings: torch.Tensor,
        audio_features: torch.Tensor,
        audio_positions: list[int],
    ) -> None:
        """Scatter embeddings around interleaved timestamp tokens."""
        positions = torch.tensor(audio_positions, device=input_embeddings.device)
        input_embeddings[0, positions, :] = audio_features[0]


__all__ = [
    "MossTranscribeDiarizeTorchMpsModelRunner",
    "install_torch_mps_language_model",
    "load_language_model",
]
