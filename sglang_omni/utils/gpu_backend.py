# SPDX-License-Identifier: Apache-2.0
"""Detect the available GPU backend without importing the serving platform stack."""

from __future__ import annotations

from types import ModuleType
from typing import Literal

try:
    import torch
except ImportError:
    torch = None


def gpu_device_type(torch_module: ModuleType | None = None) -> Literal["cuda", "xpu"]:
    """Use XPU when available; otherwise retain CUDA's best-effort telemetry path."""
    torch_module = torch if torch_module is None else torch_module
    xpu_available = (
        torch_module is not None
        and hasattr(torch_module, "xpu")
        and torch_module.xpu.is_available()
    )
    if xpu_available:
        return "xpu"
    else:
        return "cuda"
