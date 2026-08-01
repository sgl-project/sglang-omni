# SPDX-License-Identifier: Apache-2.0
"""Standalone Qwen3-VL vision encoder for Cosmos3-Nano."""

from __future__ import annotations

import logging
import types
from typing import Any

import torch
import torch.nn as nn
from transformers import Qwen3VLVisionModel
from transformers.models.qwen3_vl import modeling_qwen3_vl as hf_modeling
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from sglang_omni.models.cosmos3.bootstrap import resolve_vision_encoder_path
from sglang_omni.models.weight_loader import load_module, resolve_dtype
from sglang_omni.utils import instantiate_module, load_hf_config

logger = logging.getLogger(__name__)


class Cosmos3Qwen3VLVisionModelCompat(Qwen3VLVisionModel):
    """Qwen3-VL tower with the interpolation arithmetic used by Nano's export."""

    def _legacy_pos_embed_interpolate(
        self,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        grid_thw_list = grid_thw.tolist()
        grid_ts = [row[0] for row in grid_thw_list]
        grid_hs = [row[1] for row in grid_thw_list]
        grid_ws = [row[2] for row in grid_thw_list]
        device = self.pos_embed.weight.device

        index_lists = [[] for _ in range(4)]
        weight_lists = [[] for _ in range(4)]
        for _, height, width in grid_thw_list:
            height_indices = torch.linspace(0, self.num_grid_per_side - 1, height)
            width_indices = torch.linspace(0, self.num_grid_per_side - 1, width)
            height_floor = height_indices.int()
            width_floor = width_indices.int()
            height_ceil = (height_floor + 1).clip(max=self.num_grid_per_side - 1)
            width_ceil = (width_floor + 1).clip(max=self.num_grid_per_side - 1)
            delta_height = height_indices - height_floor
            delta_width = width_indices - width_floor
            base_height = height_floor * self.num_grid_per_side
            base_height_ceil = height_ceil * self.num_grid_per_side

            indices = [
                (base_height[:, None] + width_floor[None]).flatten(),
                (base_height[:, None] + width_ceil[None]).flatten(),
                (base_height_ceil[:, None] + width_floor[None]).flatten(),
                (base_height_ceil[:, None] + width_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - delta_height)[:, None] * (1 - delta_width)[None]).flatten(),
                ((1 - delta_height)[:, None] * delta_width[None]).flatten(),
                (delta_height[:, None] * (1 - delta_width)[None]).flatten(),
                (delta_height[:, None] * delta_width[None]).flatten(),
            ]
            for corner in range(4):
                index_lists[corner].extend(indices[corner].tolist())
                weight_lists[corner].extend(weights[corner].tolist())

        index_tensor = torch.tensor(index_lists, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(
            weight_lists,
            dtype=self.pos_embed.weight.dtype,
            device=device,
        )
        corners = self.pos_embed(index_tensor) * weight_tensor[:, :, None]
        position_embeds = corners.sum(dim=0)
        split_embeds = position_embeds.split(
            [height * width for height, width in zip(grid_hs, grid_ws)]
        )

        reordered = []
        merge_size = self.config.spatial_merge_size
        for position_embed, grid_t, grid_h, grid_w in zip(
            split_embeds,
            grid_ts,
            grid_hs,
            grid_ws,
        ):
            position_embed = position_embed.repeat(grid_t, 1)
            position_embed = (
                position_embed.view(
                    grid_t,
                    grid_h // merge_size,
                    merge_size,
                    grid_w // merge_size,
                    merge_size,
                    -1,
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            reordered.append(position_embed)
        return torch.cat(reordered)

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | hf_modeling.BaseModelOutputWithDeepstackFeatures:
        kwargs.pop("bilinear_indices", None)
        kwargs.pop("bilinear_weights", None)
        position_ids = hf_modeling.get_vision_position_ids(
            grid_thw,
            self.spatial_merge_size,
            kwargs=kwargs,
        )
        cu_seqlens = hf_modeling.get_vision_cu_seqlens(grid_thw, kwargs=kwargs)

        hidden_states = self.patch_embed(hidden_states)
        position_embeds = self._legacy_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + position_embeds.to(hidden_states.dtype)
        rotary_pos_emb = self.rotary_pos_emb(position_ids)
        rotary_pos_emb = rotary_pos_emb.reshape(hidden_states.shape[0], -1)
        rotary_pos_emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (rotary_pos_emb.cos(), rotary_pos_emb.sin())

        deepstack_features = []
        for layer_index, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_index in self.deepstack_visual_indexes:
                merger_index = self.deepstack_visual_indexes.index(layer_index)
                deepstack_features.append(
                    self.deepstack_merger_list[merger_index](hidden_states)
                )

        return hf_modeling.BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=self.merger(hidden_states),
            deepstack_features=deepstack_features,
        )


def _linear_patch_embed_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    return self.linear(hidden_states.to(dtype=self.linear.weight.dtype))


def _optimize_patch_embed(visual: nn.Module) -> None:
    """Replace the non-overlapping Conv3d patch projection with a Linear."""

    patch_embed = getattr(visual, "patch_embed", None)
    conv = getattr(patch_embed, "proj", None)
    if not isinstance(conv, nn.Conv3d):
        return
    if list(conv.kernel_size) != list(conv.stride):
        return
    if conv.padding != (0, 0, 0) or conv.dilation != (1, 1, 1) or conv.groups != 1:
        return

    in_features = (
        conv.in_channels
        * conv.kernel_size[0]
        * conv.kernel_size[1]
        * conv.kernel_size[2]
    )
    linear = nn.Linear(
        in_features,
        conv.out_channels,
        bias=conv.bias is not None,
        dtype=conv.weight.dtype,
        device=conv.weight.device,
    )
    with torch.no_grad():
        linear.weight.copy_(conv.weight.view(conv.out_channels, -1))
        if conv.bias is not None:
            linear.bias.copy_(conv.bias)

    del patch_embed.proj
    patch_embed.linear = linear
    patch_embed.forward = types.MethodType(_linear_patch_embed_forward, patch_embed)
    logger.info("Cosmos3 vision patch projection replaced with Linear")


class Cosmos3VisionEncoder(nn.Module):
    """Vision tower split from the unified Cosmos3 checkpoint."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
    ) -> None:
        super().__init__()
        root_config = load_hf_config(model_path, local_files_only=True)
        vision_config = root_config.vision_config
        torch_dtype = resolve_dtype(dtype)
        visual = instantiate_module(Cosmos3Qwen3VLVisionModelCompat, vision_config)
        self.visual = load_module(
            visual,
            resolve_vision_encoder_path(model_path),
            prefix="",
            dtype=torch_dtype,
            device=device,
            strict=True,
        )
        _optimize_patch_embed(self.visual)

        self._device = torch.device(device)
        self.spatial_merge_size = int(vision_config.spatial_merge_size)
        self.out_hidden_size = int(vision_config.out_hidden_size)
        self.deepstack_layers = len(vision_config.deepstack_visual_indexes)
        self.visual_dtype_bytes = torch.empty(
            (), dtype=self.visual.dtype
        ).element_size()

    def _encode(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        grid_thw = grid_thw.to(self._device, dtype=torch.long)
        pixel_values = pixel_values.to(
            device=self._device,
            dtype=self.visual.dtype,
        )
        output = self.visual(pixel_values, grid_thw=grid_thw, return_dict=True)
        return output.pooler_output, list(output.deepstack_features)

    def forward(
        self,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        merge = self.spatial_merge_size**2

        if isinstance(pixel_values, torch.Tensor) and isinstance(
            image_grid_thw, torch.Tensor
        ):
            image_grid_thw = image_grid_thw.to(self._device, dtype=torch.long)
            image_embeds, image_deepstack = self._encode(
                pixel_values,
                image_grid_thw,
            )
            outputs.update(
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
                image_token_counts=image_grid_thw.prod(-1) // merge,
                deepstack_visual_embeds_image=image_deepstack,
            )

        if isinstance(pixel_values_videos, torch.Tensor) and isinstance(
            video_grid_thw, torch.Tensor
        ):
            video_grid_thw = video_grid_thw.to(self._device, dtype=torch.long)
            video_embeds, video_deepstack = self._encode(
                pixel_values_videos,
                video_grid_thw,
            )
            outputs.update(
                video_embeds=video_embeds,
                video_grid_thw=video_grid_thw,
                video_token_counts=video_grid_thw.prod(-1) // merge,
                deepstack_visual_embeds_video=video_deepstack,
            )

        return outputs


__all__ = ["Cosmos3Qwen3VLVisionModelCompat", "Cosmos3VisionEncoder"]
