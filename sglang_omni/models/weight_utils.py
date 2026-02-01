# SPDX-License-Identifier: Apache-2.0
"""Weight loading utilities."""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Iterator

import torch


def get_weight_files(model_path: str) -> list[str]:
    """Get all weight files from model path."""
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    if safetensors_files:
        return sorted(safetensors_files)
    return []


def safetensors_weights_iterator(
    files: list[str],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Iterate over weights from safetensors files."""
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("safetensors is required for loading safetensors files")

    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def weights_iterator(
    model_path: str,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Iterate over weights from model path."""
    files = get_weight_files(model_path)
    if not files:
        raise ValueError(f"No weight files found in {model_path}")
    yield from safetensors_weights_iterator(files)


def filter_weights(
    weights: Iterator[tuple[str, torch.Tensor]],
    prefix: str,
    remove_prefix: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Filter weights by prefix.

    Args:
        weights: Weight iterator
        prefix: Prefix to filter by (e.g., "thinker.", "talker.")
        remove_prefix: Whether to remove the prefix from names

    Yields:
        Filtered (name, tensor) pairs
    """
    for name, tensor in weights:
        if name.startswith(prefix):
            if remove_prefix:
                name = name[len(prefix):]
            yield name, tensor


def default_weight_loader(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
) -> None:
    """Default weight loader - simple copy."""
    if param.numel() == 1 and loaded_weight.numel() == 1:
        param.data.fill_(loaded_weight.item())
    else:
        assert param.size() == loaded_weight.size(), (
            f"Parameter size mismatch: {param.size()} vs {loaded_weight.size()}"
        )
        param.data.copy_(loaded_weight)


def load_config(model_path: str) -> dict[str, Any]:
    """Load model config from model path."""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)
