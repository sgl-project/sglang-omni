# SPDX-License-Identifier: Apache-2.0
"""Prompt template constants for dots.tts serving (inference-only)."""

from __future__ import annotations

# note (chenyang): copied from upstream dots.tts; training pipeline classes omitted.
TTS_TEXT_PREFIX = "[文本]"
TTS_AUDIO_PREFIX = "[文本对应语音]"
TTS_INSTRUCTION_TEXT_PREFIX = "[带指令文本]"
TTA_TEXT_PREFIX = "[声音描述]"
TTA_AUDIO_PREFIX = "[描述对应声音]"
TTS_INTERLEAVE_PREFIX = "[流式语音合成]"
DEFAULT_TRAIN_TEMPLATE = f"{TTS_TEXT_PREFIX}{{text}}{TTS_AUDIO_PREFIX}{{audio}}"
DEFAULT_INSTRUCTION_TTS_TEMPLATE = (
    f"{TTS_INSTRUCTION_TEXT_PREFIX}{{text}}{TTS_AUDIO_PREFIX}{{audio}}"
)
DEFAULT_TEXT_TO_AUDIO_TEMPLATE = f"{TTA_TEXT_PREFIX}{{text}}{TTA_AUDIO_PREFIX}{{audio}}"
DEFAULT_INTERLEAVE_TRAIN_TEMPLATE = f"{TTS_INTERLEAVE_PREFIX}{{interleave}}"

__all__ = [
    "DEFAULT_INSTRUCTION_TTS_TEMPLATE",
    "DEFAULT_INTERLEAVE_TRAIN_TEMPLATE",
    "DEFAULT_TEXT_TO_AUDIO_TEMPLATE",
    "DEFAULT_TRAIN_TEMPLATE",
    "TTA_AUDIO_PREFIX",
    "TTA_TEXT_PREFIX",
    "TTS_AUDIO_PREFIX",
    "TTS_INSTRUCTION_TEXT_PREFIX",
    "TTS_INTERLEAVE_PREFIX",
    "TTS_TEXT_PREFIX",
]
