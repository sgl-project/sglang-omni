# SPDX-License-Identifier: Apache-2.0
"""Accelerator platform detection without changing PyTorch device semantics.

ROCm intentionally exposes AMD GPUs through PyTorch's ``torch.cuda`` API and
``cuda`` device type.  Callers should therefore use this module to choose
vendor-specific integrations, not to construct tensor device strings.
"""

from __future__ import annotations

import importlib
from enum import Enum
from typing import Any


class AcceleratorPlatform(str, Enum):
    """GPU runtime backing PyTorch's CUDA-compatible device API."""

    NVIDIA = "nvidia-cuda"
    AMD = "amd-rocm"
    NONE = "none"


def detect_accelerator_platform(torch_module: Any | None = None) -> AcceleratorPlatform:
    """Return the build platform without initializing an accelerator context.

    Inspecting ``torch.version`` distinguishes CUDA and HIP builds. Availability
    is deliberately separate: a ROCm-built container remains an AMD platform
    even when no device was passed through to it, which produces better install
    diagnostics than reporting it as a CPU build.
    """

    torch = torch_module or importlib.import_module("torch")
    version = getattr(torch, "version", None)
    if getattr(version, "hip", None):
        return AcceleratorPlatform.AMD
    if getattr(version, "cuda", None):
        return AcceleratorPlatform.NVIDIA
    return AcceleratorPlatform.NONE


def is_rocm(torch_module: Any | None = None) -> bool:
    """Whether PyTorch was built for AMD ROCm."""

    return detect_accelerator_platform(torch_module) is AcceleratorPlatform.AMD


def supports_nvidia_cuda_ipc(torch_module: Any | None = None) -> bool:
    """Whether Omni's NVIDIA-specific CUDA IPC relay may be selected."""

    return detect_accelerator_platform(torch_module) is AcceleratorPlatform.NVIDIA
