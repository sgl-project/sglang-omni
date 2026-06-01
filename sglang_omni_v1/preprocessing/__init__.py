# SPDX-License-Identifier: Apache-2.0
"""High-level preprocessing utilities (model-agnostic)."""

from sglang_omni_v1.preprocessing.audio import (
    AudioMediaIO,
    build_audio_mm_inputs,
    compute_audio_cache_key,
    ensure_audio_list_async,
)
from sglang_omni_v1.preprocessing.base import MediaIO
from sglang_omni_v1.preprocessing.image import (
    ImageMediaIO,
    build_image_mm_inputs,
    compute_image_cache_key,
    ensure_image_list_async,
)
from sglang_omni_v1.preprocessing.resource_connector import (
    MultiModalResourceConnector,
    get_global_resource_connector,
)
from sglang_omni_v1.preprocessing.text import (
    append_modality_placeholders,
    apply_chat_template,
    ensure_chat_template,
    load_chat_template,
    normalize_messages,
)
from sglang_omni_v1.preprocessing.video import (
    VideoMediaIO,
    build_video_mm_inputs,
    compute_video_cache_key,
    ensure_video_list_async,
)

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
