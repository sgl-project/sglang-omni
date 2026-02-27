# SPDX-License-Identifier: Apache-2.0
"""Audio encoder component for Qwen3-ASR."""

from __future__ import annotations

import torch
import torch.nn as nn

from sglang_omni.models.qwen3_asr.components.common import load_thinker_config
from sglang_omni.models.qwen3_asr.modeling import modeling_qwen3_asr as hf_modeling
from sglang_omni.models.utils.hf import instantiate_module
from sglang_omni.models.weight_loader import load_module, resolve_dtype

AUDIO_TOWER_PREFIX = ("thinker.audio_tower.", "audio_tower.")
AUDIO_TOWER_CLASS = hf_modeling.Qwen3ASRAudioEncoder


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


class Qwen3ASRAudioEncoder(nn.Module):
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
            # In Qwen3-ASR, input_features might be (batch, mel, time)
            # hf_modeling.Qwen3ASRAudioEncoder.forward expects input_features, feature_lens
            # We need to handle the case where input_features is padded and we want to pass only active parts
            # or pass the whole thing with lengths.
            # Qwen3-ASR audio_tower forward: forward(self, input_features, feature_lens=None, aftercnn_lens=None)

        if audio_feature_lengths is None:
            raise ValueError(
                "audio_feature_lengths or feature_attention_mask is required"
            )

        audio_feature_lengths = audio_feature_lengths.to(self._device, dtype=torch.long)
        # Ensure input_features is on the right device and dtype
        input_features = input_features.to(
            device=self._device, dtype=self.audio_tower.dtype
        )

        # Qwen3-ASR audio encoder does not support batch inference to keep precision
        # We loop over batches as in the original HF implementation
        all_audio_embeds = []
        for i in range(input_features.shape[0]):
            feat = input_features[i, :, : audio_feature_lengths[i]]
            outputs = self.audio_tower(
                feat,
                feature_lens=audio_feature_lengths[i].unsqueeze(0),
            )
            all_audio_embeds.append(outputs.last_hidden_state)

        # Concatenate all features into a single sequence (flat list of tokens)
        # sglang-omni expects (num_total_tokens, hidden_size) or (batch, seq, hidden)
        # but the merge_embeddings logic in thinker.py expects them to be cat'd or handled correctly.
        # Usually sglang-omni components return a dict that pipeline stages use.

        audio_embeds = torch.cat(all_audio_embeds, dim=0)
        audio_output_lengths = self._downsample_lengths(audio_feature_lengths)

        return {
            "audio_embeds": audio_embeds,
            "audio_feature_lengths": audio_feature_lengths,
            "audio_output_lengths": audio_output_lengths,
        }
