# SPDX-License-Identifier: Apache-2.0
"""Compatibility exports for shared TTS streaming helpers."""

from sglang_omni.models.tts_common import streaming as _streaming

INITIAL_CODEC_CHUNK_FRAMES_PARAM = _streaming.INITIAL_CODEC_CHUNK_FRAMES_PARAM
resolve_initial_codec_chunk_frames = _streaming.resolve_initial_codec_chunk_frames
