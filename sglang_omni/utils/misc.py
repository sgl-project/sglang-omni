# SPDX-License-Identifier: Apache-2.0
"""Import helpers for config-driven wiring."""

from __future__ import annotations

import importlib
import re
from typing import Any


def import_string(path: str) -> Any:
    if not path or not isinstance(path, str):
        raise ValueError("Import path must be a non-empty string")

    module_path, _, attr = path.rpartition(".")
    if not module_path or not attr:
        raise ValueError(f"Invalid import path: {path!r}")

    module = importlib.import_module(module_path)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(f"Module {module_path!r} has no attribute {attr!r}") from exc


def get_layer_id(weight_name):
    # example weight name: model.layers.10.self_attn.qkv_proj.weight
    match = re.search(r"layers\.(\d+)\.", weight_name)
    if match:
        return int(match.group(1))
    return None


def add_prefix(name: str, prefix: str) -> str:
    """Add a weight path prefix to a module name.

    Args:
        name: base module name.
        prefix: weight prefix str to added to the front of `name` concatenated with `.`.

    Returns:
        The string `prefix.name` if prefix is non-empty, otherwise just `name`.
    """
    return name if not prefix else f"{prefix}.{name}"
