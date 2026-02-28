# SPDX-License-Identifier: Apache-2.0
"""Import helpers for config-driven wiring."""

from __future__ import annotations

import importlib
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
