# SPDX-License-Identifier: Apache-2.0
"""Minimal Ming 3 speech-generator factories for CPU orchestration tests."""

from sglang_omni.models.ming3_audio.talker import Ming3SpeechGenerator


def make_ming3_speech_generator(
    *,
    enable_text_segmentation: bool,
    max_decode_steps: int = 1,
    max_semantic_length: int = 2,
    max_fragment_characters: int = 4,
    max_fragments: int = 3,
    max_input_characters: int = 256,
    max_total_decode_steps: int = 128,
) -> Ming3SpeechGenerator:
    speech_generator = Ming3SpeechGenerator.__new__(Ming3SpeechGenerator)
    speech_generator.max_decode_steps = max_decode_steps
    speech_generator.enable_text_segmentation = enable_text_segmentation
    speech_generator.max_semantic_length = max_semantic_length
    speech_generator.max_fragment_characters = max_fragment_characters
    speech_generator.max_fragments = max_fragments
    speech_generator.max_input_characters = max_input_characters
    speech_generator.max_total_decode_steps = max_total_decode_steps
    return speech_generator
