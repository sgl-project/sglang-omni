# SPDX-License-Identifier: Apache-2.0
"""SGLang-style lazy platform discovery for SGLang-Omni."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.platforms.cuda_platform import (
    CudaDeviceMixin,
    CudaPlatform,
    RocmDeviceMixin,
    RocmPlatform,
)
from sglang_omni.platforms.interface import (
    CpuDeviceMixin,
    CpuPlatform,
    DeviceMixin,
    Platform,
    PlatformEnum,
)

_current_platform: Platform | None = None


def _is_cuda_available(torch_api: Any) -> bool:
    runtime = getattr(torch_api, "cuda", None)
    return runtime is not None and bool(runtime.is_available())


def resolve_current_platform(torch_module: Any = torch) -> Platform:
    """Resolve one usable platform, falling back to CPU."""

    if _is_cuda_available(torch_module):
        if getattr(getattr(torch_module, "version", None), "hip", None):
            return RocmPlatform(torch_module)
        return CudaPlatform(torch_module)
    return CpuPlatform()


current_platform: Platform


def __getattr__(name: str) -> Platform:
    """Initialize ``current_platform`` on first access."""

    if name == "current_platform":
        global _current_platform
        if _current_platform is None:
            _current_platform = resolve_current_platform()
        return _current_platform
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CpuDeviceMixin",
    "CpuPlatform",
    "CudaDeviceMixin",
    "CudaPlatform",
    "DeviceMixin",
    "Platform",
    "PlatformEnum",
    "RocmDeviceMixin",
    "RocmPlatform",
    "current_platform",
    "resolve_current_platform",
]
