# SPDX-License-Identifier: Apache-2.0
"""Focused compatibility patches for Hugging Face Transformers."""

from __future__ import annotations

import sys
from typing import Any, Callable

import torch

_NPU_CAPTURE_PATCH_MARKER = "_sglang_omni_accelerator_capture_aware"


def _is_accelerator_stream_capturing(tensor: Any | None) -> bool:
    """Query the tensor's device module without assuming CUDA."""

    try:
        if tensor is None:
            device_module = torch.get_device_module()
        else:
            device = getattr(tensor, "device", None)
            if device is None or getattr(device, "type", None) == "cpu":
                return False
            device_module = torch.get_device_module(device)
        query = getattr(device_module, "is_current_stream_capturing", None)
        return bool(query is not None and query())
    except Exception:
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

    original: Callable[[Any | None], bool] = import_utils.is_tracing
    if getattr(original, _NPU_CAPTURE_PATCH_MARKER, False):
        return

    def accelerator_capture_aware_is_tracing(tensor: Any | None = None) -> bool:
        return bool(original(tensor) or _is_accelerator_stream_capturing(tensor))

    setattr(accelerator_capture_aware_is_tracing, _NPU_CAPTURE_PATCH_MARKER, True)
    import_utils.is_tracing = accelerator_capture_aware_is_tracing

    masking_utils = sys.modules.get("transformers.masking_utils")
    if (
        masking_utils is not None
        and getattr(masking_utils, "is_tracing", None) is original
    ):
        masking_utils.is_tracing = accelerator_capture_aware_is_tracing


__all__ = ["patch_transformers_stream_capture_detection"]
