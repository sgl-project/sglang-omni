# SPDX-License-Identifier: Apache-2.0
"""Torch-native image encoder wrapper for Qwen3-Omni."""

from __future__ import annotations

import torch
import torch.nn as nn

from sglang_omni.models.qwen3_omni.components.torch_common import load_config_dict
from sglang_omni.models.qwen3_omni.modeling import Qwen3OmniVisionEncoder as TorchVision
from sglang_omni.models.weight_loader import load_weights_by_prefixes, resolve_dtype

VISUAL_PREFIX = ("thinker.visual.", "visual.")


class Qwen3OmniTorchImageEncoder(nn.Module):
    """Vision encoder using torch-native Qwen3-Omni modules."""

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
        thinker_cfg = config.get("thinker_config", config)
        vision_cfg = thinker_cfg.get("vision_config", thinker_cfg)

        self.visual = TorchVision(vision_cfg)
        state_dict = load_weights_by_prefixes(
            model_path,
            prefixes=VISUAL_PREFIX,
            device=str(self._device),
            dtype=torch_dtype,
        )
        self.visual.load_state_dict(state_dict, strict=True, assign=True)
        if torch_dtype is not None:
            self.visual = self.visual.to(dtype=torch_dtype)
        self.visual = self.visual.to(self._device)
        self.visual.eval()
        self.spatial_merge_size = int(vision_cfg.get("spatial_merge_size", 1))

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        image_grid_thw = image_grid_thw.to(self._device, dtype=torch.long)
        dtype = next(self.visual.parameters()).dtype
        pixel_values = pixel_values.to(device=self._device, dtype=dtype)
        image_embeds, image_embeds_multiscale = self.visual(
            pixel_values, grid_thw=image_grid_thw
        )
        merge = self.spatial_merge_size**2
        image_token_counts = image_grid_thw.prod(-1) // merge
        return {
            "image_embeds": image_embeds,
            "image_grid_thw": image_grid_thw,
            "image_token_counts": image_token_counts.to(device=self._device),
            "deepstack_visual_embeds": image_embeds_multiscale,
        }
