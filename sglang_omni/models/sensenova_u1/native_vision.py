# SPDX-License-Identifier: Apache-2.0
"""Native SenseNova U1 vision embedding path used by M6 probes.

This mirrors the small public ``NEOVisionModel`` tower without importing the
official HF modeling modules.  The tower is intentionally narrow: it covers the
native-resolution patch/dense embedding path required to compose VQA prefill
embeddings for the SGLang-native language model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch import nn

from sglang_omni.models.weight_loader import resolve_dtype


def _to_namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def _llm_hidden_size(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)


def load_u1_vision_config(model_path: str | Path) -> SimpleNamespace:
    config_path = Path(model_path) / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    vision = dict(raw["vision_config"])
    vision.setdefault("patch_size", raw.get("patch_size", 16))
    vision.setdefault("downsample_ratio", raw.get("downsample_ratio", 0.5))
    vision.setdefault("torch_dtype", raw.get("torch_dtype", "bfloat16"))
    return _to_namespace(vision)


@dataclass(slots=True)
class SenseNovaU1VisionLoadReport:
    loaded_keys: list[str] = field(default_factory=list)
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_keys and not self.unexpected_keys and not self.errors

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            {
                "loaded_count": len(self.loaded_keys),
                "missing_count": len(self.missing_keys),
                "unexpected_count": len(self.unexpected_keys),
                "error_count": len(self.errors),
                "ok": self.ok,
            }
        )
        return data


def precompute_rope_freqs_sincos(
    dim: int,
    max_position: int,
    *,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_position, device=device).type_as(inv_freq)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def build_abs_positions_from_grid_hw(
    grid_hw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = grid_hw.device
    grid_hw = grid_hw.to(device=device, dtype=torch.long)
    heights = grid_hw[:, 0]
    widths = grid_hw[:, 1]
    patch_counts = heights * widths
    total = int(patch_counts.sum().item())
    patch_to_sample = torch.repeat_interleave(
        torch.arange(grid_hw.shape[0], device=device),
        patch_counts,
    )
    patch_id = torch.arange(total, device=device)
    offsets = torch.cumsum(
        torch.cat([torch.zeros(1, device=device, dtype=torch.long), patch_counts[:-1]]),
        dim=0,
    )
    within = patch_id - offsets[patch_to_sample]
    width_per_patch = widths[patch_to_sample]
    abs_x = within % width_per_patch
    abs_y = within // width_per_patch
    return abs_x, abs_y


def apply_rotary_emb_1d(
    x: torch.Tensor,
    cos_cached: torch.Tensor,
    sin_cached: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    cos = cos_cached[positions]
    sin = sin_cached[positions]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


def apply_2d_rotary_pos_emb(
    x: torch.Tensor,
    cos_cached_x: torch.Tensor,
    sin_cached_x: torch.Tensor,
    cos_cached_y: torch.Tensor,
    sin_cached_y: torch.Tensor,
    abs_positions_x: torch.Tensor,
    abs_positions_y: torch.Tensor,
) -> torch.Tensor:
    dim_half = x.shape[-1] // 2
    x_part_1 = x[..., :dim_half]
    x_part_2 = x[..., dim_half:]
    rotated_1 = apply_rotary_emb_1d(
        x_part_1,
        cos_cached_x,
        sin_cached_x,
        abs_positions_x,
    )
    rotated_2 = apply_rotary_emb_1d(
        x_part_2,
        cos_cached_y,
        sin_cached_y,
        abs_positions_y,
    )
    return torch.cat((rotated_1, rotated_2), dim=-1)


def _rope_cache_matches(
    cos_x: torch.Tensor,
    sin_x: torch.Tensor,
    cos_y: torch.Tensor,
    sin_y: torch.Tensor,
    *,
    device: torch.device,
    max_position: int,
    dim: int,
) -> bool:
    cache_shape = (max_position, dim // 2)
    return (
        cos_x.device == device
        and sin_x.device == device
        and cos_y.device == device
        and sin_y.device == device
        and cos_x.dtype == torch.float32
        and sin_x.dtype == torch.float32
        and cos_y.dtype == torch.float32
        and sin_y.dtype == torch.float32
        and tuple(cos_x.shape) == cache_shape
        and tuple(sin_x.shape) == cache_shape
        and tuple(cos_y.shape) == cache_shape
        and tuple(sin_y.shape) == cache_shape
    )


class SenseNovaU1NativeVisionModel(nn.Module):
    expected_weight_keys = {
        "patch_embedding.weight",
        "patch_embedding.bias",
        "dense_embedding.weight",
        "dense_embedding.bias",
    }

    def __init__(
        self,
        config: Any,
        *,
        params_dtype: torch.dtype | str | None = None,
    ) -> None:
        super().__init__()
        self.config = _to_namespace(config)
        dtype = resolve_dtype(params_dtype or getattr(self.config, "torch_dtype", None))
        self.embed_dim = int(self.config.hidden_size)
        self.llm_embed_dim = _llm_hidden_size(self.config.llm_hidden_size)
        self.patch_size = int(self.config.patch_size)
        self.downsample_factor = int(1 / float(self.config.downsample_ratio))

        self.patch_embedding = nn.Conv2d(
            in_channels=int(self.config.num_channels),
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            dtype=dtype,
        )
        self.dense_embedding = nn.Conv2d(
            in_channels=self.embed_dim,
            out_channels=self.llm_embed_dim,
            kernel_size=self.downsample_factor,
            stride=self.downsample_factor,
            dtype=dtype,
        )
        self.gelu = nn.GELU()

        rope_dim_part = self.embed_dim // 2
        self.rope_dim_part = rope_dim_part
        cos_x, sin_x = precompute_rope_freqs_sincos(
            rope_dim_part,
            int(self.config.max_position_embeddings_vision),
            base=float(self.config.rope_theta_vision),
        )
        cos_y, sin_y = precompute_rope_freqs_sincos(
            rope_dim_part,
            int(self.config.max_position_embeddings_vision),
            base=float(self.config.rope_theta_vision),
        )
        self.register_buffer("cos_cached_x", cos_x, persistent=False)
        self.register_buffer("sin_cached_x", sin_x, persistent=False)
        self.register_buffer("cos_cached_y", cos_y, persistent=False)
        self.register_buffer("sin_cached_y", sin_y, persistent=False)

    @classmethod
    def from_model_path(
        cls,
        model_path: str | Path,
        *,
        params_dtype: torch.dtype | str | None = None,
    ) -> "SenseNovaU1NativeVisionModel":
        return cls(load_u1_vision_config(model_path), params_dtype=params_dtype)

    def load_weights(self, model_path: str | Path) -> SenseNovaU1VisionLoadReport:
        model_path = Path(model_path)
        index_file = model_path / "model.safetensors.index.json"
        with index_file.open("r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]

        shards: dict[str, list[str]] = {}
        for key, shard_name in weight_map.items():
            if key.startswith("vision_model."):
                shards.setdefault(shard_name, []).append(key)

        loaded: set[str] = set()
        errors: list[str] = []
        from safetensors import safe_open

        params = dict(self.named_parameters())
        for shard_name in sorted(shards):
            with safe_open(str(model_path / shard_name), framework="pt", device="cpu") as f:
                for full_key in sorted(shards[shard_name]):
                    name = full_key[len("vision_model.") :]
                    if name.startswith("embeddings."):
                        name = name[len("embeddings.") :]
                    tensor = f.get_tensor(full_key)
                    param = params.get(name)
                    if param is None:
                        errors.append(f"unexpected parameter target {name}")
                        continue
                    if tuple(param.shape) != tuple(tensor.shape):
                        errors.append(
                            f"shape mismatch for {name}: param={tuple(param.shape)} "
                            f"checkpoint={tuple(tensor.shape)}"
                        )
                        continue
                    with torch.no_grad():
                        param.copy_(tensor.to(device=param.device, dtype=param.dtype))
                    loaded.add(name)

        unexpected = sorted(loaded - self.expected_weight_keys)
        missing = sorted(self.expected_weight_keys - loaded)
        return SenseNovaU1VisionLoadReport(
            loaded_keys=sorted(loaded),
            missing_keys=missing,
            unexpected_keys=unexpected,
            errors=errors,
        )

    def _ensure_fp32_rope_cache(self, device: torch.device) -> None:
        """Keep vision RoPE cache aligned with full HF NEOChatModel.

        The HF runner loads weights in bf16, then calls ``model.to(device)``
        without a dtype, so non-persistent vision RoPE caches stay fp32. Native
        serving moves this module with ``dtype=bf16`` for weights; refresh the
        caches in fp32 before RoPE so high-frequency spatial rotations match.
        """

        max_position = int(self.config.max_position_embeddings_vision)
        if _rope_cache_matches(
            self.cos_cached_x,
            self.sin_cached_x,
            self.cos_cached_y,
            self.sin_cached_y,
            device=device,
            max_position=max_position,
            dim=self.rope_dim_part,
        ):
            return
        cos_x, sin_x = precompute_rope_freqs_sincos(
            self.rope_dim_part,
            max_position,
            base=float(self.config.rope_theta_vision),
            device=device,
        )
        cos_y, sin_y = precompute_rope_freqs_sincos(
            self.rope_dim_part,
            max_position,
            base=float(self.config.rope_theta_vision),
            device=device,
        )
        self.cos_cached_x = cos_x
        self.sin_cached_x = sin_x
        self.cos_cached_y = cos_y
        self.sin_cached_y = sin_y

    def _apply_2d_rotary_pos_emb(
        self,
        patch_embeds: torch.Tensor,
        grid_hw: torch.Tensor,
    ) -> torch.Tensor:
        self._ensure_fp32_rope_cache(patch_embeds.device)
        abs_pos_x, abs_pos_y = build_abs_positions_from_grid_hw(grid_hw)
        embeddings = apply_2d_rotary_pos_emb(
            patch_embeds.float(),
            self.cos_cached_x,
            self.sin_cached_x,
            self.cos_cached_y,
            self.sin_cached_y,
            abs_pos_x,
            abs_pos_y,
        )
        return embeddings.to(self.patch_embedding.weight.dtype)

    def forward(self, pixel_values: torch.Tensor, grid_hw: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 2:
            raise ValueError(
                f"pixel_values must be 2D native-resolution patches, got {pixel_values.ndim}D"
            )
        grid_hw = grid_hw.to(device=pixel_values.device, dtype=torch.long)
        pixels = pixel_values.reshape(-1, 3, self.patch_size, self.patch_size)
        patch_embeds = self.gelu(self.patch_embedding(pixels)).reshape(-1, self.embed_dim)
        patch_embeds = self._apply_2d_rotary_pos_emb(patch_embeds, grid_hw)
        expected_patches = int((grid_hw[:, 0] * grid_hw[:, 1]).sum().item())
        if expected_patches != int(patch_embeds.shape[0]):
            raise ValueError(
                f"grid_hw patch count {expected_patches} != patch embeddings {patch_embeds.shape[0]}"
            )

        patches: list[torch.Tensor] = []
        cur = 0
        for row in grid_hw:
            h = int(row[0].item())
            w = int(row[1].item())
            img = patch_embeds[cur : cur + h * w].reshape(h, w, -1).unsqueeze(0)
            img = self.dense_embedding(img.permute(0, 3, 1, 2))
            img = img.permute(0, 2, 3, 1)
            patches.append(img.reshape(-1, img.shape[-1]))
            cur += h * w
        embeddings = torch.cat(patches, dim=0)
        expected_tokens = expected_patches // (self.downsample_factor**2)
        if int(embeddings.shape[0]) != expected_tokens:
            raise ValueError(
                f"vision embeddings {embeddings.shape[0]} != expected image tokens {expected_tokens}"
            )
        return embeddings


__all__ = [
    "SenseNovaU1NativeVisionModel",
    "SenseNovaU1VisionLoadReport",
    "build_abs_positions_from_grid_hw",
    "load_u1_vision_config",
]
