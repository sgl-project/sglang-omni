# SPDX-License-Identifier: Apache-2.0
"""Lightweight import helpers."""

from __future__ import annotations

import importlib
from typing import Any


def import_string(path: str) -> Any:
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise ImportError(f"Invalid import path: {path}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)
