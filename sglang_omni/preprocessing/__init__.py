# SPDX-License-Identifier: Apache-2.0
"""High-level preprocessing utilities (model-agnostic)."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from sglang_omni.preprocessing.base import MediaIO
from sglang_omni.preprocessing.resource_connector import (
    MultiModalResourceConnector,
    get_global_resource_connector,
)

if TYPE_CHECKING:
    from sglang_omni.preprocessing.audio import (
        AudioMediaIO,
        build_audio_mm_inputs,
        compute_audio_cache_key,
        ensure_audio_list_async,
    )
    from sglang_omni.preprocessing.image import (
        ImageMediaIO,
        build_image_mm_inputs,
        compute_image_cache_key,
        ensure_image_list_async,
    )
    from sglang_omni.preprocessing.text import (
        append_modality_placeholders,
        apply_chat_template,
        ensure_chat_template,
        load_chat_template,
        normalize_messages,
    )
    from sglang_omni.preprocessing.transcription import (
        PreparedAudio,
        prepare_audio,
        resolve_audio_source,
    )
    from sglang_omni.preprocessing.video import (
        VideoMediaIO,
        build_video_mm_inputs,
        compute_video_cache_key,
        ensure_video_list_async,
    )
else:
    pass

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
    "prepare_audio": "sglang_omni.preprocessing.transcription",
    "PreparedAudio": "sglang_omni.preprocessing.transcription",
    "resolve_audio_source": "sglang_omni.preprocessing.transcription",
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
    "prepare_audio",
    "PreparedAudio",
    "resolve_audio_source",
    "VideoMediaIO",
]


def __getattr__(name: str) -> object:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    else:
        pass
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
