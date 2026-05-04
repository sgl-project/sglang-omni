# SPDX-License-Identifier: Apache-2.0
"""Ming-Omni vision encoder composing sglang's native ViT sub-modules.

Builds the same ViT architecture as sglang's ``Qwen3OmniMoeVisionEncoder``
(Qwen3-style ViT with deepstack mergers) but without the top-level
``Qwen3VLMoeVisionModel`` runtime dependencies (PP group, server_args,
CUDA graph runners).

Keeps:
- TP via ``ColumnParallelLinear`` / ``RowParallelLinear`` / ``VocabParallelEmbedding``
- Flash attention via ``VisionAttention``
- Deepstack multi-scale feature support

Skips (can be added later):
- Pipeline parallelism (always single PP rank)
- CUDA graph capture for vision forward
- flashinfer_cudnn attention backend
"""

from __future__ import annotations

import logging
import re
from functools import partial
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from sglang_omni_v1.models.weight_loader import default_weight_loader

logger = logging.getLogger(__name__)


def _extract_vision_dict(vision_config) -> dict:
    """Extract a plain dict from a vision config object."""
    if hasattr(vision_config, "to_dict"):
        d = vision_config.to_dict()
    elif hasattr(vision_config, "__dataclass_fields__"):
        from dataclasses import asdict

        d = asdict(vision_config)
    else:
        d = {k: v for k, v in vars(vision_config).items() if not k.startswith("_")}
    # Remove keys that Qwen3OmniMoeVisionEncoderConfig doesn't accept
    for key in ("model_type", "transformers_version", "torch_dtype"):
        d.pop(key, None)
    return d


_WEIGHT_SUBSTR_MAPPINGS = {
    "deepstack_merger_list.": "merger_list.",
    "merger.norm.": "merger.ln_q.",
    "merger.linear_fc1.": "merger.mlp.0.",
    "merger.linear_fc2.": "merger.mlp.2.",
}


def _remap_ming_vision_weight(name: str) -> str:
    """Remap Ming checkpoint vision weight name to sglang model name."""
    # 1. Merger / deepstack naming
    for old, new in _WEIGHT_SUBSTR_MAPPINGS.items():
        name = name.replace(old, new)
    # Also handle merger_list inner layers (after deepstack rename)
    name = re.sub(r"(merger_list\.\d+)\.norm\.", r"\1.ln_q.", name)
    name = re.sub(r"(merger_list\.\d+)\.linear_fc1\.", r"\1.mlp.0.", name)
    name = re.sub(r"(merger_list\.\d+)\.linear_fc2\.", r"\1.mlp.2.", name)

    # 2. VisionAttention naming
    name = name.replace("attn.qkv.", "attn.qkv_proj.")
    name = name.replace("attn.out_proj.", "attn.proj.")

    return name


class MingOmniVisionEncoder(nn.Module):
    """ViT for Ming-Omni, composed from sglang sub-modules.

    Architecture:
      PatchEmbed(Conv3d) -> pos_embed -> 27x VisionBlock -> merger
                                          + deepstack mergers at [8,16,24]

    Weight checkpoint mapping (Ming -> this model):
      deepstack_merger_list.N.* -> merger_list.N.*
      merger.norm.*             -> merger.ln_q.*
      merger.linear_fc1.*       -> merger.mlp.0.*
      merger.linear_fc2.*       -> merger.mlp.2.*
      attn.qkv.*               -> attn.qkv_proj.*
      attn.out_proj.*           -> attn.proj.*
    """

    def __init__(
        self,
        vision_config,
        quant_config: Optional[object] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        from sglang.srt.layers.rotary_embedding import get_rope
        from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
        from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeVisionPatchMerger
        from sglang.srt.models.qwen3_vl import (
            Qwen3_VisionBlock,
            Qwen3VLVisionPatchEmbed,
        )
        from sglang.srt.models.utils import RotaryPosMixin
        from sglang.srt.utils import add_prefix

        # Store mixin method as a static helper
        self._rot_pos_ids = RotaryPosMixin.rot_pos_ids

        # --- config ---
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.num_position_embeddings = vision_config.num_position_embeddings
        self.num_grid_per_side = int(self.num_position_embeddings**0.5)
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.deepstack_visual_indexes = vision_config.deepstack_visual_indexes or []

        self.image_emb_dim = vision_config.out_hidden_size
        self.use_deepstack = bool(self.deepstack_visual_indexes)
        self.out_hidden_size = vision_config.out_hidden_size * (
            1 + len(self.deepstack_visual_indexes)
        )

        # --- patch embedding (Conv3d, no deps) ---
        self.patch_embed = Qwen3VLVisionPatchEmbed(config=vision_config)

        # --- position embedding (TP-sharded) ---
        self.pos_embed = VocabParallelEmbedding(
            self.num_position_embeddings,
            self.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("pos_embed", prefix),
        )

        # --- rotary pos embedding for vision blocks ---
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = get_rope(
            head_size=head_dim,
            rotary_dim=head_dim // 2,
            max_position=8192,
            base=10000.0,
            is_neox_style=True,
        )

        # --- vision transformer blocks ---
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList(
            [
                Qwen3_VisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    intermediate_dim=vision_config.intermediate_size,
                    hidden_act=vision_config.hidden_act,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{layer_idx}", prefix),
                )
                for layer_idx in range(vision_config.depth)
            ]
        )

        # --- merger (final spatial merge, uses Qwen3-Omni style with ln_q) ---
        self.merger = Qwen3OmniMoeVisionPatchMerger(
            dim=vision_config.out_hidden_size,
            context_dim=self.hidden_size,
            spatial_merge_size=self.spatial_merge_size,
            quant_config=quant_config,
            use_postshuffle_norm=False,
            prefix=add_prefix("merger", prefix),
        )

        # --- deepstack mergers ---
        self.merger_list = nn.ModuleList(
            [
                Qwen3OmniMoeVisionPatchMerger(
                    dim=vision_config.out_hidden_size,
                    context_dim=self.hidden_size,
                    spatial_merge_size=self.spatial_merge_size,
                    use_postshuffle_norm=True,
                    quant_config=quant_config,
                    prefix=add_prefix(f"merger_list.{i}", prefix),
                )
                for i in range(len(self.deepstack_visual_indexes))
            ]
        )

    # --- properties ---

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    # --- position embedding helpers (adapted from Qwen3VLMoeVisionModel) ---

    def _rot_pos_emb(
        self, grid_thw: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute rotary position embeddings for vision blocks."""
        pos_ids = []
        for t, h, w in grid_thw:
            base = self._rot_pos_ids(h, w, self.spatial_merge_size)
            pos_ids.append(base if t == 1 else base.repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0).to(self.device, non_blocking=True)
        max_grid_size = max(max(h, w) for _, h, w in grid_thw)

        cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)
        cos_combined = cos[pos_ids].flatten(1)
        sin_combined = sin[pos_ids].flatten(1)
        return cos_combined, sin_combined

    def _interpolate_pos_embed(self, grid_thw: list[list[int]]) -> torch.Tensor:
        """Bilinear interpolation of position embeddings.

        Adapted from ``Qwen3VLMoeVisionModel.fast_pos_embed_interpolate_from_list``.
        Uses ``align_corners=True`` (linspace from 0 to N-1).
        """
        num_grid_per_side = self.num_grid_per_side
        m_size = self.spatial_merge_size
        hidden_dim = self.pos_embed.embedding_dim

        outputs = []
        for t, h, w in grid_thw:
            h_idxs = torch.linspace(
                0, num_grid_per_side - 1, h, dtype=torch.float32, device=self.device
            )
            w_idxs = torch.linspace(
                0, num_grid_per_side - 1, w, dtype=torch.float32, device=self.device
            )

            h_floor = h_idxs.to(torch.long)
            w_floor = w_idxs.to(torch.long)
            h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
            w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
            h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
            h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

            w11 = dh_grid * dw_grid
            w10 = dh_grid - w11
            w01 = dw_grid - w11
            w00 = 1 - dh_grid - w01

            h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
            w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
            indices = (h_grid * num_grid_per_side + w_grid).reshape(4, -1)
            weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)
            weights = weights.to(dtype=self.dtype)

            embeds = self.pos_embed(indices)
            embeds *= weights
            combined = embeds.sum(dim=0)

            combined = combined.reshape(
                h // m_size, m_size, w // m_size, m_size, hidden_dim
            )
            combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
            repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
            outputs.append(repeated)

        return torch.cat(outputs, dim=0)

    # --- forward ---

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the vision encoder.

        Adapted from ``Qwen3VLMoeVisionModel.forward()`` without CUDA graph
        and flashinfer_cudnn paths.

        Args:
            pixel_values: Flattened pixel values, shape [total_patches, C*Tp*Pp*Pp].
            grid_thw: [num_images, 3] tensor of (t, h, w) grid dimensions.

        Returns:
            If deepstack: [seq_len, out_hidden_size * (1 + num_deepstack)].
            Otherwise: [seq_len, out_hidden_size].
        """
        x = pixel_values.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        # Convert grid_thw to list for iteration
        if isinstance(grid_thw, torch.Tensor):
            grid_thw_list = grid_thw.tolist()
            grid_thw_np = grid_thw.cpu().numpy()
        else:
            grid_thw_list = grid_thw
            grid_thw_np = np.array(grid_thw, dtype=np.int32)

        # Position embeddings (bilinear interpolation)
        pos_embeds = self._interpolate_pos_embed(grid_thw_list)
        x += pos_embeds

        # Rotary position embeddings for attention
        rotary_pos_emb_cos, rotary_pos_emb_sin = self._rot_pos_emb(grid_thw_list)

        # Build cu_seqlens for variable-length flash attention
        token_cu_seqlens = np.repeat(
            grid_thw_np[:, 1] * grid_thw_np[:, 2], grid_thw_np[:, 0]
        ).cumsum(axis=0, dtype=np.int32)
        token_cu_seqlens = np.concatenate(
            [np.zeros(1, dtype=np.int32), token_cu_seqlens]
        )
        cu_seqlens = torch.from_numpy(token_cu_seqlens).to(
            self.device, non_blocking=True
        )

        # VisionBlock expects shape [seq_len, 1, hidden_size]
        x = x.unsqueeze(1)

        # Run vision transformer blocks
        deepstack_feature_lists = []
        num_deepstack_captured = 0

        for layer_num, blk in enumerate(self.blocks):
            x = blk(
                x,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
            )

            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.merger_list[num_deepstack_captured](x)
                deepstack_feature_lists.append(deepstack_feature)
                num_deepstack_captured += 1

        # Final merger
        x = self.merger(x)

        # Concat deepstack features
        hidden_states = torch.cat(
            [x] + deepstack_feature_lists, dim=1
        )  # [seq_len, out_hidden_size * (1 + num_deepstack)]

        return hidden_states

    # --- weight loading ---

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with Ming -> sglang name remapping."""
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            name = _remap_ming_vision_weight(name)

            if name not in params_dict:
                logger.debug("Skipping unknown vision weight: %s", name)
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params
