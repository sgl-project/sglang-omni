# SPDX-License-Identifier: Apache-2.0
"""Focused compatibility patches for Hugging Face Transformers."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import torch

_patched_is_tracing: Callable[[Any | None], bool] | None = None


def _is_accelerator_stream_capturing(tensor: Any | None) -> bool:
    """Query the tensor's device module without assuming CUDA."""

    try:
        if tensor is None:
            device_module = torch.get_device_module()
        else:
            device = tensor.device
            if device.type == "cpu":
                return False
            device_module = torch.get_device_module(device)
        return bool(device_module.is_current_stream_capturing())
    except (AttributeError, RuntimeError, ValueError):
        # Match Transformers' probing behavior: unavailable accelerator APIs
        # are not evidence that tracing is active.
        return False


def patch_transformers_stream_capture_detection() -> None:
    """Make ``is_tracing(tensor)`` recognize non-CUDA graph capture.

    Transformers imports ``is_tracing`` directly into ``masking_utils``, so
    both the source symbol and an already-imported binding must be updated.
    The wrapper preserves every upstream tracing check and only adds a generic
    device-module capture query when those checks return false.
    """

    from transformers.utils import import_utils

    global _patched_is_tracing

    current: Callable[[Any | None], bool] = import_utils.is_tracing
    if current is _patched_is_tracing:
        return

    def accelerator_capture_aware_is_tracing(tensor: Any | None = None) -> bool:
        return bool(current(tensor) or _is_accelerator_stream_capturing(tensor))

    _patched_is_tracing = accelerator_capture_aware_is_tracing
    import_utils.is_tracing = accelerator_capture_aware_is_tracing

    masking_utils = sys.modules.get("transformers.masking_utils")
    if masking_utils is not None and masking_utils.is_tracing is current:
        masking_utils.is_tracing = accelerator_capture_aware_is_tracing


__all__ = ["patch_transformers_stream_capture_detection"]
