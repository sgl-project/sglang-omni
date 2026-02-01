# SPDX-License-Identifier: Apache-2.0
"""Torch-native audio encoder wrapper for Qwen3-Omni."""

from __future__ import annotations

import torch
import torch.nn as nn

from sglang_omni.models.qwen3_omni.components.torch_common import load_config_dict
from sglang_omni.models.qwen3_omni.modeling import Qwen3OmniAudioEncoder as TorchAudio
from sglang_omni.models.weight_loader import load_weights_by_prefix, resolve_dtype


class Qwen3OmniTorchAudioEncoder(nn.Module):
    """Audio encoder using torch-native Qwen3-Omni modules."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self._device = torch.device(device)
        torch_dtype = resolve_dtype(dtype)
        config = load_config_dict(model_path)
        self.audio_tower = TorchAudio(config)
        if torch_dtype is not None:
            self.audio_tower = self.audio_tower.to(dtype=torch_dtype)
        self.audio_tower = self.audio_tower.to(self._device)
        self.audio_tower.eval()

        state_dict = load_weights_by_prefix(
            model_path,
            prefix=("thinker.audio_tower.", "audio_tower."),
        )
        self.audio_tower.load_state_dict(state_dict, strict=True)

    def forward(
        self,
        *,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = (
                input_features.permute(0, 2, 1)[feature_attention_mask.bool()]
                .permute(1, 0)
                .contiguous()
            )
        if audio_feature_lengths is None:
            raise ValueError(
                "audio_feature_lengths or feature_attention_mask is required"
            )

        audio_feature_lengths = audio_feature_lengths.to(self._device, dtype=torch.long)
        input_features = input_features.to(self._device, dtype=self.audio_tower.conv_out.weight.dtype)
        audio_embeds = self.audio_tower(
            input_features=input_features,
            feature_lens=audio_feature_lengths,
        )
        audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(  # pylint: disable=protected-access
            audio_feature_lengths
        )
        return {
            "audio_embeds": audio_embeds,
            "audio_feature_lengths": audio_feature_lengths,
            "audio_output_lengths": audio_output_lengths,
        }
