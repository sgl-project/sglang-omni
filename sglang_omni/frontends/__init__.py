# SPDX-License-Identifier: Apache-2.0
"""High-level frontend utilities (model-agnostic)."""

from sglang_omni.frontends.audio import (
    build_audio_mm_inputs,
    compute_audio_cache_key,
    ensure_audio_list,
)
from sglang_omni.frontends.image import (
    build_image_mm_inputs,
    compute_image_cache_key,
    ensure_image_list,
)
from sglang_omni.frontends.text import (
    append_modality_placeholders,
    apply_chat_template,
    ensure_chat_template,
    load_chat_template,
    normalize_messages,
)
from sglang_omni.frontends.video import (
    build_video_mm_inputs,
    compute_video_cache_key,
    ensure_video_list,
)

__all__ = [
    "append_modality_placeholders",
    "apply_chat_template",
    "build_audio_mm_inputs",
    "build_image_mm_inputs",
    "build_video_mm_inputs",
    "compute_audio_cache_key",
    "compute_image_cache_key",
    "compute_video_cache_key",
    "ensure_audio_list",
    "ensure_chat_template",
    "ensure_image_list",
    "ensure_video_list",
    "load_chat_template",
    "normalize_messages",
]
