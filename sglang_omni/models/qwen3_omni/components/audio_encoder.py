# SPDX-License-Identifier: Apache-2.0
"""Audio encoder component for Qwen3-Omni."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

from sglang_omni.models.qwen3_omni.components.common import load_thinker_config
from sglang_omni.models.weight_loader import load_module, resolve_dtype
from sglang_omni.utils import instantiate_module

AUDIO_TOWER_PREFIX = ("thinker.audio_tower.", "audio_tower.")
AUDIO_TOWER_CLASS = hf_modeling.Qwen3OmniMoeAudioEncoder


def _build_audio_tower(
    model_id: str,
    *,
    thinker_cfg: object,
    torch_dtype: torch.dtype | None,
    device: str,
) -> nn.Module:
    audio_cfg = thinker_cfg.audio_config
    audio_tower = instantiate_module(AUDIO_TOWER_CLASS, audio_cfg)
    return load_module(
        audio_tower,
        model_id,
        prefix=AUDIO_TOWER_PREFIX,
        dtype=torch_dtype,
        device=device,
        strict=True,
    )


class Qwen3OmniAudioEncoder(nn.Module):
    """Audio tower extracted from the HF thinker."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
    ) -> None:
        super().__init__()
        torch_dtype = resolve_dtype(dtype)
        thinker_cfg = load_thinker_config(model_id)
        self._device = torch.device(device)
        self.audio_tower = _build_audio_tower(
            model_id,
            thinker_cfg=thinker_cfg,
            torch_dtype=torch_dtype,
            device=device,
        )
        self._downsample_lengths = hf_modeling._get_feat_extract_output_lengths

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
        outputs = self.audio_tower(
            input_features.to(device=self._device, dtype=self.audio_tower.dtype),
            feature_lens=audio_feature_lengths,
        )
        audio_embeds = outputs.last_hidden_state
        audio_output_lengths = self._downsample_lengths(audio_feature_lengths)
        return {
            "audio_embeds": audio_embeds,
            "audio_feature_lengths": audio_feature_lengths,
            "audio_output_lengths": audio_output_lengths,
        }
