# SPDX-License-Identifier: Apache-2.0
"""Image encoder component for Qwen3-Omni."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

from sglang_omni.models.qwen3_omni.components.common import load_thinker_config
from sglang_omni.models.utils.hf import instantiate_module
from sglang_omni.models.weight_loader import load_module, resolve_dtype

VISUAL_PREFIX = "thinker.visual."
VISUAL_CLASS = hf_modeling.Qwen3OmniMoeVisionEncoder


def _build_visual(
    model_path: str,
    *,
    thinker_cfg: object,
    torch_dtype: torch.dtype | None,
    device: str,
) -> nn.Module:
    vision_cfg = thinker_cfg.vision_config
    visual = instantiate_module(VISUAL_CLASS, vision_cfg)
    return load_module(
        visual,
        model_path,
        prefix=VISUAL_PREFIX,
        dtype=torch_dtype,
        device=device,
        strict=True,
    )


class Qwen3OmniImageEncoder(nn.Module):
    """Vision tower extracted from the HF thinker."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
    ) -> None:
        super().__init__()
        torch_dtype = resolve_dtype(dtype)
        thinker_cfg = load_thinker_config(model_path)
        self._device = torch.device(device)
        self.visual = _build_visual(
            model_path,
            thinker_cfg=thinker_cfg,
            torch_dtype=torch_dtype,
            device=device,
        )
        self.spatial_merge_size = int(thinker_cfg.vision_config.spatial_merge_size)

    def forward(
        self,
        *,
        pixel_values: torch.Tensor | list | None = None,
        image_grid_thw: torch.Tensor | list | None = None,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(pixel_values, torch.Tensor):
            return {}
        if not isinstance(image_grid_thw, torch.Tensor):
            return {}
        image_grid_thw = image_grid_thw.to(self._device, dtype=torch.long)
        pixel_values = pixel_values.to(device=self._device, dtype=self.visual.dtype)
        output = self.visual(pixel_values, grid_thw=image_grid_thw)
        # Newer transformers returns a ModelOutput dataclass; older versions
        # return a plain tuple (hidden_states, deepstack_feature_lists).
        if isinstance(output, tuple):
            image_embeds, image_embeds_multiscale = output
        else:
            image_embeds = output.pooler_output
            image_embeds_multiscale = output.deepstack_features
        merge = self.spatial_merge_size**2
        image_token_counts = image_grid_thw.prod(-1) // merge
        return {
            "image_embeds": image_embeds,
            "image_grid_thw": image_grid_thw,
            "image_token_counts": image_token_counts.to(device=self._device),
            "deepstack_visual_embeds": image_embeds_multiscale,
        }
