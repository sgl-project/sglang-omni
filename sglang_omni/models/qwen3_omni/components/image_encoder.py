# SPDX-License-Identifier: Apache-2.0
"""Image encoder component for Qwen3-Omni."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

from sglang_omni.models.qwen3_omni.components.common import load_thinker_config
from sglang_omni.models.weight_loader import load_module, resolve_dtype
from sglang_omni.utils import instantiate_module

VISUAL_PREFIX = ("thinker.visual.", "visual.")
VISUAL_CLASS = hf_modeling.Qwen3OmniMoeVisionEncoder


def _build_visual(
    model_id: str,
    *,
    thinker_cfg: object,
    torch_dtype: torch.dtype | None,
    device: str,
) -> nn.Module:
    vision_cfg = thinker_cfg.vision_config
    visual = instantiate_module(VISUAL_CLASS, vision_cfg)
    return load_module(
        visual,
        model_id,
        prefix=VISUAL_PREFIX,
        dtype=torch_dtype,
        device=device,
        strict=True,
    )


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
        torch_dtype = resolve_dtype(dtype)
        thinker_cfg = load_thinker_config(model_id)
        self._device = torch.device(device)
        self.visual = _build_visual(
            model_id,
            thinker_cfg=thinker_cfg,
            torch_dtype=torch_dtype,
            device=device,
        )
        self.spatial_merge_size = int(thinker_cfg.vision_config.spatial_merge_size)

    def forward(
        self,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        merge = self.spatial_merge_size**2

        if isinstance(pixel_values, torch.Tensor) and isinstance(
            image_grid_thw, torch.Tensor
        ):
            image_grid_thw = image_grid_thw.to(self._device, dtype=torch.long)
            pixel_values = pixel_values.to(device=self._device, dtype=self.visual.dtype)
            image_embeds, image_embeds_multiscale = self.visual(
                pixel_values, grid_thw=image_grid_thw
            )
            image_token_counts = image_grid_thw.prod(-1) // merge
            outputs.update(
                {
                    "image_embeds": image_embeds,
                    "image_grid_thw": image_grid_thw,
                    "image_token_counts": image_token_counts.to(device=self._device),
                    "deepstack_visual_embeds_image": image_embeds_multiscale,
                }
            )

        if isinstance(pixel_values_videos, torch.Tensor) and isinstance(
            video_grid_thw, torch.Tensor
        ):
            video_grid_thw = video_grid_thw.to(self._device, dtype=torch.long)
            pixel_values_videos = pixel_values_videos.to(
                device=self._device, dtype=self.visual.dtype
            )
            video_embeds, video_embeds_multiscale = self.visual(
                pixel_values_videos, grid_thw=video_grid_thw
            )
            video_token_counts = video_grid_thw.prod(-1) // merge
            outputs.update(
                {
                    "video_embeds": video_embeds,
                    "video_grid_thw": video_grid_thw,
                    "video_token_counts": video_token_counts.to(device=self._device),
                    "deepstack_visual_embeds_video": video_embeds_multiscale,
                }
            )

        return outputs
