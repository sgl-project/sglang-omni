# SPDX-License-Identifier: Apache-2.0
"""SGLang-Omni platform abstraction."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.platforms.cuda_platform import CudaDeviceMixin, CudaOmniPlatform
from sglang_omni.platforms.interface import (
    CpuDeviceMixin,
    CpuOmniPlatform,
    DeviceMixin,
    OmniPlatform,
    PlatformEnum,
    ResolvedPlatformSpec,
    TransferPolicy,
)

_current_platform: OmniPlatform | None = None


def resolve_current_platform(torch_module: Any = torch) -> OmniPlatform:
    return OmniPlatform.detect(torch_module)


def __getattr__(name: str) -> OmniPlatform:
    if name == "current_platform":
        global _current_platform
        if _current_platform is None:
            _current_platform = OmniPlatform.detect()
        return _current_platform
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CpuDeviceMixin",
    "CpuOmniPlatform",
    "CudaDeviceMixin",
    "CudaOmniPlatform",
    "DeviceMixin",
    "OmniPlatform",
    "PlatformEnum",
    "ResolvedPlatformSpec",
    "TransferPolicy",
    "current_platform",
    "resolve_current_platform",
]
