# SPDX-License-Identifier: Apache-2.0
"""Select the GPU backend without importing the serving platform stack."""

from __future__ import annotations

import os
import pkgutil
import sys
from types import ModuleType
from typing import Literal

try:
    import torch
except ImportError:
    torch = None


def gpu_device_type(
    torch_module: ModuleType | None = None,
    *,
    device_type: Literal["cuda", "xpu"] | None = None,
) -> Literal["cuda", "xpu"]:
    """Honor explicit and registered platform selections before probing hardware."""
    selected = device_type
    if selected is None and (torch_module is None or torch_module is torch):
        platform_module = sys.modules.get("sglang_omni.platforms")
        platform_spec = os.environ.get("SGLANG_OMNI_PLATFORM_SPEC")
        srt_platform_module = sys.modules.get("sglang.srt.platforms")
        if platform_module is not None:
            selected = platform_module.current_platform.device_type
        elif platform_spec:
            selected = pkgutil.resolve_name(platform_spec).device_type
        elif srt_platform_module is not None:
            selected = srt_platform_module.current_platform.device_type
        else:
            selected = os.environ.get("SGLANG_PLATFORM")
    else:
        pass
    if selected is not None:
        if selected in ("cuda", "xpu"):
            return selected
        else:
            raise ValueError(f"GPU telemetry does not support device type {selected!r}")
    else:
        pass
    torch_module = torch if torch_module is None else torch_module
    cuda_available = torch_module is not None and torch_module.cuda.is_available()
    xpu_available = (
        torch_module is not None
        and hasattr(torch_module, "xpu")
        and torch_module.xpu.is_available()
    )
    if cuda_available and xpu_available:
        raise ValueError("Select cuda or xpu explicitly on a host with both backends")
    elif xpu_available:
        return "xpu"
    else:
        return "cuda"
