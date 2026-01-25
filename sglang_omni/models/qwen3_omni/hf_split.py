# SPDX-License-Identifier: Apache-2.0
"""Split-model helpers for Qwen3-Omni (HF-based)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

from sglang_omni.models.qwen3_omni.registry import get_shared_thinker


@dataclass(frozen=True)
class Qwen3OmniSpec:
    """Lightweight spec extracted from the HF config."""

    model_id: str
    audio_token_id: int
    image_token_id: int
    spatial_merge_size: int

    @classmethod
    def from_model_id(cls, model_id: str) -> "Qwen3OmniSpec":
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        thinker_cfg = getattr(cfg, "thinker_config", cfg)
        vision_cfg = thinker_cfg.vision_config
        return cls(
            model_id=model_id,
            audio_token_id=int(thinker_cfg.audio_token_id),
            image_token_id=int(thinker_cfg.image_token_id),
            spatial_merge_size=int(vision_cfg.spatial_merge_size),
        )


def _concat_features(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        tensors = [v for v in value if isinstance(v, torch.Tensor)]
        if not tensors:
            return None
        return torch.cat(tensors, dim=0)
    return None


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
        thinker = get_shared_thinker(model_id, dtype=dtype)
        self._device = torch.device(device)
        self.audio_tower = thinker.audio_tower.to(self._device)
        self.audio_tower.eval()
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
            raise ValueError("audio_feature_lengths or feature_attention_mask is required")

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


class Qwen3OmniImageEncoder(nn.Module):
    """Vision tower extracted from the HF thinker."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
    ) -> None:
        super().__init__()
        thinker = get_shared_thinker(model_id, dtype=dtype)
        self._device = torch.device(device)
        self.visual = thinker.visual.to(self._device)
        self.visual.eval()
        self.spatial_merge_size = int(thinker.config.vision_config.spatial_merge_size)

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        image_grid_thw = image_grid_thw.to(self._device, dtype=torch.long)
        pixel_values = pixel_values.to(device=self._device, dtype=self.visual.dtype)
        image_embeds, _ = self.visual(pixel_values, grid_thw=image_grid_thw)
        merge = self.spatial_merge_size**2
        image_token_counts = image_grid_thw.prod(-1) // merge
        return {
            "image_embeds": image_embeds,
            "image_grid_thw": image_grid_thw,
            "image_token_counts": image_token_counts.to(device=self._device),
        }


class Qwen3OmniSplitThinker(nn.Module):
    """Thinker wrapper that accepts precomputed encoder embeddings."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self._device = torch.device(device)
        self.thinker = get_shared_thinker(model_id, dtype=dtype)
        # Move only the text model and LM head to the thinker device.
        self.thinker.model = self.thinker.model.to(self._device)
        self.thinker.lm_head = self.thinker.lm_head.to(self._device)

    def _merge_embeddings(
        self,
        *,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_embeds: torch.Tensor | None,
        audio_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        image_mask, _, audio_mask = self.thinker.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )

        if image_embeds is not None:
            image_embeds = image_embeds.to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if audio_embeds is not None:
            audio_token_count = int((input_ids == self.thinker.config.audio_token_id).sum().item())
            if audio_token_count != int(audio_embeds.shape[0]):
                raise ValueError(
                    "Audio placeholder count mismatch: "
                    f"tokens={audio_token_count} embeds={audio_embeds.shape[0]}"
                )
            audio_embeds = audio_embeds.to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        image_embeds: torch.Tensor | list[torch.Tensor] | None = None,
        audio_embeds: torch.Tensor | list[torch.Tensor] | None = None,
        **kwargs: Any,
    ):
        image_embeds_t = _concat_features(image_embeds)
        audio_embeds_t = _concat_features(audio_embeds)

        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(self._device)

        if inputs_embeds is None and (image_embeds_t is not None or audio_embeds_t is not None):
            inputs_embeds = self.thinker.get_input_embeddings()(input_ids.to(self._device))

        if inputs_embeds is not None and (image_embeds_t is not None or audio_embeds_t is not None):
            image_embeds_t = image_embeds_t.to(self._device) if image_embeds_t is not None else None
            audio_embeds_t = audio_embeds_t.to(self._device) if audio_embeds_t is not None else None
            inputs_embeds = self._merge_embeddings(
                input_ids=input_ids.to(self._device),
                inputs_embeds=inputs_embeds,
                image_embeds=image_embeds_t,
                audio_embeds=audio_embeds_t,
            )

        return self.thinker(
            input_ids=input_ids.to(self._device),
            attention_mask=attention_mask.to(self._device) if isinstance(attention_mask, torch.Tensor) else None,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
