# SPDX-License-Identifier: Apache-2.0
"""Native MLX Qwen3-Omni vision encoder."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mlx.utils import tree_flatten

from sglang_omni.models.qwen3_omni.mlx.common import load_qwen3_omni_mlx_component
from sglang_omni.models.qwen3_omni.mlx.config import Qwen3OmniMlxConfig, VisionConfig
from sglang_omni.models.qwen3_omni.mlx.runner import read_qwen3_omni_component_weights
from sglang_omni.models.qwen3_omni.mlx.tensor_utils import (
    mlx_to_torch as _mlx_to_torch,
)
from sglang_omni.models.qwen3_omni.mlx.tensor_utils import (
    torch_to_mlx as _torch_to_mlx,
)

_VISION_PREFIXES = ("thinker.visual.", "thinker.vision_tower.")
_VISION_LOCAL_PREFIXES = (
    "patch_embed.",
    "pos_embed.",
    "blocks.",
    "merger.",
    "merger_list.",
    "deepstack_merger_list.",
)
# MLX SDPA can materialize its score matrix, so cap each Metal temporary at 1 GiB.
_VISION_ATTENTION_SCORE_BUDGET_BYTES = 1 << 30
_VISION_ATTENTION_SCORE_BYTES_PER_ELEMENT = 4


def sanitize_vision_weights(
    weights: Mapping[str, mx.array],
    *,
    expected_shapes: Mapping[str, tuple[int, ...]],
) -> dict[str, mx.array]:
    """Map official and converted vision weights onto the native MLX model."""

    sanitized: dict[str, mx.array] = {}
    for source_key, value in weights.items():
        key = source_key
        for prefix in _VISION_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break

        if key.startswith("merger_list."):
            key = f"deepstack_merger_list.{key[len('merger_list.') :]}"
        if key.startswith(("merger.", "deepstack_merger_list.")):
            key = key.replace(".ln_q.", ".norm.")
            key = key.replace(".mlp.0.", ".linear_fc1.")
            key = key.replace(".mlp.2.", ".linear_fc2.")

        if key == "patch_embed.proj.weight":
            expected_shape = expected_shapes.get(key)
            source_shape = tuple(value.shape)
            if expected_shape is None:
                raise ValueError(
                    "Qwen3-Omni vision model has no target shape for "
                    "'patch_embed.proj.weight'"
                )
            if source_shape != expected_shape:
                candidate = value.transpose(0, 2, 3, 4, 1) if value.ndim == 5 else None
                if candidate is None or tuple(candidate.shape) != expected_shape:
                    raise ValueError(
                        f"Qwen3-Omni vision tensor {source_key!r} has shape "
                        f"{source_shape}; expected MLX shape {expected_shape} "
                        "or a Torch Conv3D layout that transposes to it"
                    )
                value = candidate

        if key in sanitized:
            raise ValueError(
                f"Qwen3-Omni vision weights {source_key!r} and another source "
                f"both map to {key!r}"
            )
        sanitized[key] = value
    return sanitized


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate((-x[..., half:], x[..., :half]), axis=-1)


class VisionPatchEmbed(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        kernel = (
            config.temporal_patch_size,
            config.patch_size,
            config.patch_size,
        )
        self.in_channels = config.in_channels
        self.temporal_patch_size = config.temporal_patch_size
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        self.proj = nn.Conv3d(
            config.in_channels,
            config.hidden_size,
            kernel_size=kernel,
            stride=kernel,
            bias=True,
        )

    def __call__(self, pixel_values: mx.array) -> mx.array:
        hidden = pixel_values.reshape(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden = hidden.transpose(0, 2, 3, 4, 1)
        return self.proj(hidden).reshape(-1, self.hidden_size)


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def __call__(
        self,
        hidden: mx.array,
        *,
        rope: tuple[mx.array, mx.array],
        cu_seqlens: tuple[int, ...],
    ) -> mx.array:
        length = hidden.shape[0]
        qkv = self.qkv(hidden).reshape(length, 3, self.num_heads, self.head_dim)
        queries, keys, values = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        cos, sin = rope

        query_dtype = queries.dtype
        key_dtype = keys.dtype
        queries_float = queries.astype(mx.float32)
        keys_float = keys.astype(mx.float32)
        cos = cos[:, None, :].astype(mx.float32)
        sin = sin[:, None, :].astype(mx.float32)
        queries = (queries_float * cos + _rotate_half(queries_float) * sin).astype(
            query_dtype
        )
        keys = (keys_float * cos + _rotate_half(keys_float) * sin).astype(key_dtype)

        chunks: list[mx.array] = []
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
            query = queries[start:end].transpose(1, 0, 2)[None]
            key = keys[start:end].transpose(1, 0, 2)[None]
            value = values[start:end].transpose(1, 0, 2)[None]
            query_chunk_size = min(
                query.shape[2],
                max(
                    1,
                    _VISION_ATTENTION_SCORE_BUDGET_BYTES
                    // (
                        self.num_heads
                        * key.shape[2]
                        * _VISION_ATTENTION_SCORE_BYTES_PER_ELEMENT
                    ),
                ),
            )
            is_chunked = query_chunk_size < query.shape[2]
            attended_chunks: list[mx.array] = []
            for query_start in range(0, query.shape[2], query_chunk_size):
                query_end = min(query_start + query_chunk_size, query.shape[2])
                attended_chunk = mx.fast.scaled_dot_product_attention(
                    query[:, :, query_start:query_end],
                    key,
                    value,
                    scale=self.scale,
                )
                if is_chunked:
                    mx.eval(attended_chunk)
                attended_chunks.append(attended_chunk)
            attended = (
                attended_chunks[0]
                if len(attended_chunks) == 1
                else mx.concatenate(attended_chunks, axis=2)
            )
            chunks.append(attended[0].transpose(1, 0, 2))

        attended = mx.concatenate(chunks, axis=0).reshape(length, -1)
        return self.proj(attended)


class VisionMlp(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.hidden_act = config.hidden_act
        self.linear_fc1 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=True
        )
        self.linear_fc2 = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=True
        )

    def __call__(self, hidden: mx.array) -> mx.array:
        hidden = self.linear_fc1(hidden)
        if self.hidden_act in ("gelu_pytorch_tanh", "gelu_new"):
            hidden = nn.gelu_approx(hidden)
        elif self.hidden_act == "gelu":
            hidden = nn.gelu(hidden)
        else:
            raise ValueError(
                f"unsupported Qwen3-Omni vision activation {self.hidden_act!r}"
            )
        return self.linear_fc2(hidden)


class VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = VisionAttention(config)
        self.mlp = VisionMlp(config)

    def __call__(
        self,
        hidden: mx.array,
        *,
        rope: tuple[mx.array, mx.array],
        cu_seqlens: tuple[int, ...],
    ) -> mx.array:
        hidden = hidden + self.attn(
            self.norm1(hidden),
            rope=rope,
            cu_seqlens=cu_seqlens,
        )
        return hidden + self.mlp(self.norm2(hidden))


class VisionPatchMerger(nn.Module):
    def __init__(self, config: VisionConfig, *, use_postshuffle_norm: bool):
        super().__init__()
        self.merged_hidden_size = config.hidden_size * config.spatial_merge_size**2
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_size = (
            self.merged_hidden_size if use_postshuffle_norm else config.hidden_size
        )
        self.norm = nn.LayerNorm(norm_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(
            self.merged_hidden_size, self.merged_hidden_size, bias=True
        )
        self.linear_fc2 = nn.Linear(
            self.merged_hidden_size, config.out_hidden_size, bias=True
        )

    def __call__(self, hidden: mx.array) -> mx.array:
        if self.use_postshuffle_norm:
            hidden = self.norm(hidden.reshape(-1, self.merged_hidden_size))
        else:
            hidden = self.norm(hidden).reshape(-1, self.merged_hidden_size)
        return self.linear_fc2(nn.gelu(self.linear_fc1(hidden)))


class Qwen3OmniMlxVisionEncoder(nn.Module):
    """Qwen3-Omni ViT with legacy interpolation and frame-segmented attention."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        if config.hidden_size % config.num_heads:
            raise ValueError("vision hidden_size must be divisible by num_heads")
        num_grid_per_side = math.isqrt(config.num_position_embeddings)
        if num_grid_per_side**2 != config.num_position_embeddings:
            raise ValueError("vision num_position_embeddings must be a square")

        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.num_grid_per_side = num_grid_per_side
        self.patch_embed = VisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(
            config.num_position_embeddings, config.hidden_size
        )
        self.blocks = [VisionBlock(config) for _ in range(config.depth)]
        self.merger = VisionPatchMerger(config, use_postshuffle_norm=False)
        self.deepstack_merger_list = [
            VisionPatchMerger(config, use_postshuffle_norm=True)
            for _ in config.deepstack_visual_indexes
        ]
        self.deepstack_visual_indexes = config.deepstack_visual_indexes

    def _validated_grid(
        self, grid_thw: mx.array, *, expected_rows: int | None = None
    ) -> tuple[tuple[int, int, int], ...]:
        raw = np.asarray(grid_thw)
        if raw.ndim != 2 or raw.shape[1] != 3:
            raise ValueError("grid_thw must have shape (num_images_or_videos, 3)")
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("grid_thw must contain integers")
        grid = tuple(tuple(int(value) for value in row) for row in raw.tolist())
        if any(value <= 0 for row in grid for value in row):
            raise ValueError("grid_thw values must be positive")
        merge = self.spatial_merge_size
        if any(h % merge or w % merge for _, h, w in grid):
            raise ValueError(
                "grid height and width must be divisible by spatial_merge_size"
            )
        rows = sum(t * h * w for t, h, w in grid)
        if expected_rows is not None and rows != expected_rows:
            raise ValueError(
                f"grid_thw describes {rows} patches but pixel_values has "
                f"{expected_rows} rows"
            )
        return grid

    def interpolate_position_embeddings(self, grid_thw: mx.array) -> mx.array:
        grid = self._validated_grid(grid_thw)
        outputs: list[mx.array] = []
        table_dtype = self.pos_embed.weight.dtype
        side = self.num_grid_per_side
        merge = self.spatial_merge_size

        for temporal, height, width in grid:
            h_indexes = np.linspace(0, side - 1, height, dtype=np.float32)
            w_indexes = np.linspace(0, side - 1, width, dtype=np.float32)
            h_floor = h_indexes.astype(np.int32)
            w_floor = w_indexes.astype(np.int32)
            h_ceil = np.minimum(h_floor + 1, side - 1)
            w_ceil = np.minimum(w_floor + 1, side - 1)
            dh = h_indexes - h_floor
            dw = w_indexes - w_floor

            indices = np.stack(
                [
                    (h_floor[:, None] * side + w_floor[None, :]).reshape(-1),
                    (h_floor[:, None] * side + w_ceil[None, :]).reshape(-1),
                    (h_ceil[:, None] * side + w_floor[None, :]).reshape(-1),
                    (h_ceil[:, None] * side + w_ceil[None, :]).reshape(-1),
                ]
            )
            weights = np.stack(
                [
                    ((1 - dh)[:, None] * (1 - dw)[None, :]).reshape(-1),
                    ((1 - dh)[:, None] * dw[None, :]).reshape(-1),
                    (dh[:, None] * (1 - dw)[None, :]).reshape(-1),
                    (dh[:, None] * dw[None, :]).reshape(-1),
                ]
            )
            corners = self.pos_embed(mx.array(indices, dtype=mx.int32))
            spatial = mx.sum(
                corners * mx.array(weights, dtype=table_dtype)[:, :, None], axis=0
            )
            spatial = spatial.reshape(
                height // merge,
                merge,
                width // merge,
                merge,
                self.config.hidden_size,
            ).transpose(0, 2, 1, 3, 4)
            spatial = spatial.reshape(-1, self.config.hidden_size)
            outputs.append(mx.tile(spatial, (temporal, 1)))
        return mx.concatenate(outputs, axis=0)

    def build_rotary_positions(self, grid_thw: mx.array) -> tuple[mx.array, mx.array]:
        grid = self._validated_grid(grid_thw)
        merge = self.spatial_merge_size
        positions: list[np.ndarray] = []
        for temporal, height, width in grid:
            rows = np.broadcast_to(
                np.arange(height, dtype=np.int32)[:, None], (height, width)
            )
            columns = np.broadcast_to(
                np.arange(width, dtype=np.int32)[None, :], (height, width)
            )
            rows = (
                rows.reshape(height // merge, merge, width // merge, merge)
                .transpose(0, 2, 1, 3)
                .reshape(-1)
            )
            columns = (
                columns.reshape(height // merge, merge, width // merge, merge)
                .transpose(0, 2, 1, 3)
                .reshape(-1)
            )
            frame_positions = np.stack((rows, columns), axis=-1)
            positions.append(np.tile(frame_positions, (temporal, 1)))

        position_ids = mx.array(np.concatenate(positions), dtype=mx.float32)
        head_dim = self.config.hidden_size // self.config.num_heads
        rotary_dim = head_dim // 2
        inv_freq = 1.0 / (
            10000.0
            ** (mx.arange(0, rotary_dim, 2, dtype=mx.float32) / float(rotary_dim))
        )
        frequencies = (position_ids[:, :, None] * inv_freq[None, None, :]).reshape(
            position_ids.shape[0], -1
        )
        embedding = mx.concatenate((frequencies, frequencies), axis=-1)
        return mx.cos(embedding), mx.sin(embedding)

    def frame_cu_seqlens(self, grid_thw: mx.array) -> tuple[int, ...]:
        grid = self._validated_grid(grid_thw)
        cumulative = [0]
        for temporal, height, width in grid:
            for _ in range(temporal):
                cumulative.append(cumulative[-1] + height * width)
        return tuple(cumulative)

    def __call__(
        self, pixel_values: mx.array, grid_thw: mx.array
    ) -> tuple[mx.array, list[mx.array]]:
        self._validated_grid(grid_thw, expected_rows=pixel_values.shape[0])
        hidden = self.patch_embed(pixel_values)
        hidden = hidden + self.interpolate_position_embeddings(grid_thw).astype(
            hidden.dtype
        )
        rope = self.build_rotary_positions(grid_thw)
        cu_seqlens = self.frame_cu_seqlens(grid_thw)

        deepstack: list[mx.array] = []
        for layer_index, block in enumerate(self.blocks):
            hidden = block(hidden, rope=rope, cu_seqlens=cu_seqlens)
            if layer_index in self.deepstack_visual_indexes:
                index = self.deepstack_visual_indexes.index(layer_index)
                deepstack.append(self.deepstack_merger_list[index](hidden))

        primary = self.merger(hidden)
        return primary, deepstack


def load_qwen3_omni_mlx_vision(model_path: str) -> Qwen3OmniMlxVisionEncoder:
    """Load the vision component from an official or converted checkpoint."""

    from sglang.srt.hardware_backend.mlx.remote_code_gate import (
        ensure_remote_code_allowed,
        resolve_model_directory,
    )

    directory = Path(resolve_model_directory(model_path))
    ensure_remote_code_allowed(directory, False)
    raw = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    root_config = Qwen3OmniMlxConfig.from_dict(raw)
    model = Qwen3OmniMlxVisionEncoder(root_config.vision)
    expected_shapes = {
        key: tuple(value.shape) for key, value in tree_flatten(model.parameters())
    }
    weights = read_qwen3_omni_component_weights(
        directory,
        component="vision",
        official_prefixes=_VISION_PREFIXES,
        local_prefixes=_VISION_LOCAL_PREFIXES,
    )
    return load_qwen3_omni_mlx_component(
        model,
        weights,
        sanitizer=lambda raw_weights: sanitize_vision_weights(
            raw_weights,
            expected_shapes=expected_shapes,
        ),
        quantization=root_config.quantization,
    )


class Qwen3OmniMlxImageEncoder:
    """Torch-compatible stage adapter around the native MLX vision encoder."""

    def __init__(self, model_path: str) -> None:
        self.visual = load_qwen3_omni_mlx_vision(model_path)
        self.spatial_merge_size = self.visual.config.spatial_merge_size
        self.out_hidden_size = self.visual.config.out_hidden_size
        self.deepstack_layers = len(self.visual.config.deepstack_visual_indexes)
        self.visual_dtype_bytes = int(self.visual.patch_embed.proj.weight.dtype.size)

    def _encode(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        primary, deepstack = self.visual(
            _torch_to_mlx(pixel_values),
            _torch_to_mlx(grid_thw),
        )
        mx.eval(primary, deepstack)
        return _mlx_to_torch(primary), [_mlx_to_torch(layer) for layer in deepstack]

    def __call__(self, **inputs: Any) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        merge = self.spatial_merge_size**2

        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        if isinstance(pixel_values, torch.Tensor) and isinstance(
            image_grid_thw, torch.Tensor
        ):
            image_grid_thw = image_grid_thw.detach().cpu().to(dtype=torch.long)
            image_embeds, deepstack = self._encode(pixel_values, image_grid_thw)
            outputs.update(
                {
                    "image_embeds": image_embeds,
                    "image_grid_thw": image_grid_thw,
                    "image_token_counts": image_grid_thw.prod(-1) // merge,
                    "deepstack_visual_embeds_image": deepstack,
                }
            )

        pixel_values_videos = inputs.get("pixel_values_videos")
        video_grid_thw = inputs.get("video_grid_thw")
        if isinstance(pixel_values_videos, torch.Tensor) and isinstance(
            video_grid_thw, torch.Tensor
        ):
            video_grid_thw = video_grid_thw.detach().cpu().to(dtype=torch.long)
            video_embeds, deepstack = self._encode(pixel_values_videos, video_grid_thw)
            outputs.update(
                {
                    "video_embeds": video_embeds,
                    "video_grid_thw": video_grid_thw,
                    "video_token_counts": video_grid_thw.prod(-1) // merge,
                    "deepstack_visual_embeds_video": deepstack,
                }
            )

        return outputs
