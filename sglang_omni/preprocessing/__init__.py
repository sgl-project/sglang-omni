# SPDX-License-Identifier: Apache-2.0
"""High-level preprocessing utilities (model-agnostic)."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from sglang_omni.preprocessing.base import MediaIO
from sglang_omni.preprocessing.resource_connector import (
    MultiModalResourceConnector,
    get_global_resource_connector,
)

_LAZY_EXPORTS = {
    "append_modality_placeholders": "sglang_omni.preprocessing.text",
    "apply_chat_template": "sglang_omni.preprocessing.text",
    "AudioMediaIO": "sglang_omni.preprocessing.audio",
    "build_audio_mm_inputs": "sglang_omni.preprocessing.audio",
    "build_image_mm_inputs": "sglang_omni.preprocessing.image",
    "build_video_mm_inputs": "sglang_omni.preprocessing.video",
    "compute_audio_cache_key": "sglang_omni.preprocessing.audio",
    "compute_image_cache_key": "sglang_omni.preprocessing.image",
    "compute_video_cache_key": "sglang_omni.preprocessing.video",
    "ensure_audio_list_async": "sglang_omni.preprocessing.audio",
    "ensure_chat_template": "sglang_omni.preprocessing.text",
    "ensure_image_list_async": "sglang_omni.preprocessing.image",
    "ensure_video_list_async": "sglang_omni.preprocessing.video",
    "ImageMediaIO": "sglang_omni.preprocessing.image",
    "load_chat_template": "sglang_omni.preprocessing.text",
    "normalize_messages": "sglang_omni.preprocessing.text",
    "VideoMediaIO": "sglang_omni.preprocessing.video",
}

__all__ = [
    "append_modality_placeholders",
    "apply_chat_template",
    "AudioMediaIO",
    "build_audio_mm_inputs",
    "build_image_mm_inputs",
    "build_video_mm_inputs",
    "compute_audio_cache_key",
    "compute_image_cache_key",
    "compute_video_cache_key",
    "ensure_audio_list_async",
    "ensure_chat_template",
    "ensure_image_list_async",
    "ensure_video_list_async",
    "get_global_resource_connector",
    "ImageMediaIO",
    "load_chat_template",
    "MultiModalResourceConnector",
    "MediaIO",
    "normalize_messages",
    "VideoMediaIO",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
