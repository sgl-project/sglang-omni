# SPDX-License-Identifier: Apache-2.0
"""Image encoder component for Qwen3-Omni."""

from __future__ import annotations

import logging
import types

import torch
import torch.nn as nn
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

from sglang_omni.models.qwen3_omni.components.common import load_thinker_config
from sglang_omni.models.weight_loader import load_module, resolve_dtype
from sglang_omni.utils import instantiate_module

logger = logging.getLogger(__name__)

VISUAL_PREFIX = ("thinker.visual.", "visual.")
VISUAL_CLASS = hf_modeling.Qwen3OmniMoeVisionEncoder


def _patch_embed_forward(self: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Optimized PatchEmbed forward using Linear instead of Conv3d."""
    return self.linear(hidden_states.to(dtype=self.linear.weight.dtype))


def _optimize_patch_embed(visual: nn.Module) -> None:
    """Replace Conv3d with Linear in PatchEmbed for ~7-15× speedup.

    The Conv3d kernel does not slide (kernel_size == stride), so it is
    equivalent to a reshape + linear. We load weights via Conv3d for
    checkpoint compatibility, then copy them to a Linear layer.

    Reference: https://github.com/sgl-project/sglang/pull/19788
    """
    patch_embed = getattr(visual, "patch_embed", None)
    if patch_embed is None:
        return
    conv = getattr(patch_embed, "proj", None)
    if conv is None or not isinstance(conv, nn.Conv3d):
        return

    if list(conv.kernel_size) != list(conv.stride):
        logger.debug(
            "PatchEmbed Conv3d kernel_size=%s != stride=%s, skipping optimization",
            conv.kernel_size,
            conv.stride,
        )
        return

    if conv.padding != (0, 0, 0) or conv.dilation != (1, 1, 1) or conv.groups != 1:
        logger.debug(
            "PatchEmbed Conv3d has non-trivial padding/dilation/groups, skipping"
        )
        return

    embed_dim = conv.out_channels
    in_features = (
        conv.in_channels
        * conv.kernel_size[0]
        * conv.kernel_size[1]
        * conv.kernel_size[2]
    )

    linear = nn.Linear(
        in_features,
        embed_dim,
        bias=True,
        dtype=conv.weight.dtype,
        device=conv.weight.device,
    )
    with torch.no_grad():
        linear.weight.copy_(conv.weight.view(embed_dim, -1))
        linear.bias.copy_(conv.bias)

    del patch_embed.proj
    patch_embed.linear = linear
    patch_embed.forward = types.MethodType(_patch_embed_forward, patch_embed)
    logger.info(
        "PatchEmbed optimized: Conv3d(%d→%d) replaced with Linear(%d→%d)",
        conv.in_channels,
        embed_dim,
        in_features,
        embed_dim,
    )


def _build_visual(
    model_path: str,
    *,
    thinker_cfg: object,
    torch_dtype: torch.dtype | None,
    device: str,
) -> nn.Module:
    vision_cfg = thinker_cfg.vision_config
    visual = instantiate_module(VISUAL_CLASS, vision_cfg)
    visual = load_module(
        visual,
        model_path,
        prefix=VISUAL_PREFIX,
        dtype=torch_dtype,
        device=device,
        strict=True,
    )
    _optimize_patch_embed(visual)
    return visual


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
        vision_cfg = thinker_cfg.vision_config
        self._device = torch.device(device)
        self.visual = _build_visual(
            model_path,
            thinker_cfg=thinker_cfg,
            torch_dtype=torch_dtype,
            device=device,
        )
        self.spatial_merge_size = int(vision_cfg.spatial_merge_size)
        self.out_hidden_size = int(vision_cfg.out_hidden_size)
        self.deepstack_layers = len(vision_cfg.deepstack_visual_indexes)
        # Avoid constructing a visual output tensor just to learn dtype width.
        self.visual_dtype_bytes = torch.empty(
            (), dtype=self.visual.dtype
        ).element_size()

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
