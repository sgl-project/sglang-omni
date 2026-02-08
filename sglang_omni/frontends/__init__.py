# SPDX-License-Identifier: Apache-2.0
"""High-level frontend utilities (model-agnostic)."""

from sglang_omni.frontends.audio import (
    AudioMediaIO,
    build_audio_mm_inputs,
    compute_audio_cache_key,
    ensure_audio_list,
)
from sglang_omni.frontends.base import MediaIO
from sglang_omni.frontends.image import (
    ImageMediaIO,
    build_image_mm_inputs,
    compute_image_cache_key,
    ensure_image_list,
)
from sglang_omni.frontends.media_connector import (
    MediaConnector,
    get_global_media_connector,
)
from sglang_omni.frontends.text import (
    append_modality_placeholders,
    apply_chat_template,
    ensure_chat_template,
    load_chat_template,
    normalize_messages,
)
from sglang_omni.frontends.video import (
    VideoMediaIO,
    build_video_mm_inputs,
    compute_video_cache_key,
    ensure_video_list,
    extract_audio_from_video_inputs,
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
    "extract_audio_from_video_inputs",
    "ensure_audio_list",
    "ensure_chat_template",
    "ensure_image_list",
    "ensure_video_list",
    "get_global_media_connector",
    "ImageMediaIO",
    "load_chat_template",
    "MediaConnector",
    "MediaIO",
    "normalize_messages",
    "VideoMediaIO",
]
