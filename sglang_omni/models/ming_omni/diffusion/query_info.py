# SPDX-License-Identifier: Apache-2.0
"""Lightweight image-generation query-token metadata loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


@dataclass(frozen=True)
class ImageGenQueryInfo:
    query_tokens: torch.Tensor
    image_patch_token_id: int


def load_image_gen_query_info(model_path: str) -> ImageGenQueryInfo:
    local_path = _resolve_model_path(model_path)
    query_path = local_path / "mlp" / "model.safetensors"
    if not query_path.exists():
        raise FileNotFoundError(
            f"Ming image generation query token file not found: {query_path}"
        )

    tensors = load_file(str(query_path), device="cpu")
    query_tokens = _load_query_tokens(local_path, query_path, tensors)

    return ImageGenQueryInfo(
        query_tokens=query_tokens.to(dtype=torch.bfloat16, device="cpu"),
        image_patch_token_id=_load_image_patch_token_id(local_path),
    )


def _resolve_model_path(model_path: str) -> Path:
    path = Path(model_path)
    if path.exists():
        return path

    from sglang_omni.models.weight_loader import resolve_model_path

    return Path(resolve_model_path(model_path))


def _load_image_patch_token_id(local_path: Path) -> int:
    config_path = local_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Ming config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    token_id = _nested_get(config, ("llm_config", "image_patch_token"))
    if token_id is None:
        token_id = config.get("image_patch_token")
    if token_id is None:
        raise KeyError(f"image_patch_token missing from {config_path}")
    return int(token_id)


def _load_query_tokens(
    local_path: Path,
    query_path: Path,
    tensors: dict[str, torch.Tensor],
) -> torch.Tensor:
    direct_query_tokens = tensors.get("query_tokens")
    if direct_query_tokens is not None:
        return direct_query_tokens

    scales = _load_img_gen_scales(local_path)
    query_tokens = []
    missing = []
    for scale in scales:
        key = f"query_tokens_dict.{scale}x{scale}"
        tensor = tensors.get(key)
        if tensor is None:
            missing.append(key)
        else:
            query_tokens.append(tensor)

    if missing:
        available = sorted(tensors)
        raise KeyError(
            "Ming image generation query token tensors missing from "
            f"{query_path}: {missing}; available tensors={available}"
        )
    return torch.cat(query_tokens, dim=0)


def _load_img_gen_scales(local_path: Path) -> list[int]:
    config_path = local_path / "mlp" / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            "Ming image generation query token tensor 'query_tokens' missing and "
            f"real-layout config not found: {config_path}"
        )

    mlp_config = json.loads(config_path.read_text(encoding="utf-8"))
    scales = mlp_config.get("img_gen_scales")
    if scales is None:
        raise KeyError(f"img_gen_scales missing from {config_path}")
    if not isinstance(scales, list) or not scales:
        raise ValueError(f"img_gen_scales in {config_path} must be a non-empty list")

    normalized = []
    for scale in scales:
        if isinstance(scale, bool) or not isinstance(scale, int) or scale <= 0:
            raise ValueError(
                f"img_gen_scales in {config_path} must contain positive integers; "
                f"got {scales!r}"
            )
        normalized.append(scale)
    return normalized


def _nested_get(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current
