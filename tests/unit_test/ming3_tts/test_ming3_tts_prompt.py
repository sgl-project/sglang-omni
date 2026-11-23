# SPDX-License-Identifier: Apache-2.0
"""Prompt layout and checkpoint contract validation for Ming 3 TTS."""

import pytest
from pydantic import JsonValue, ValidationError

from sglang_omni.models.ming3_tts.stages import (
    ASSISTANT_PREFIX,
    SpeechConfig,
    build_speech_prompt,
)

TINY_SPEECH_CONFIG: dict[str, JsonValue] = {
    "prompt_format": "legacy_antmmf",
    "with_rml": True,
    "conditioning_mode": "speaker_only",
    "decode_alignment": "old_shift",
    "patch_size": 1,
    "history_size": 1,
    "latent_dim": 64,
    "speaker_embedding_dim": 192,
    "sampling_temperature": 1.5,
    "stop_threshold": 0.5,
    "minimum_decode_steps": 13,
    "maximum_decode_steps": 250,
    "audio_patch_token_id": 157164,
    "audio_start_token_id": 157165,
    "audio_end_token_id": 157166,
    "speaker_start_token_id": 157175,
    "speaker_end_token_id": 157176,
}


class CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]


def test_prompt_places_speaker_slot_between_speaker_markers() -> None:
    speech_config = SpeechConfig.model_validate(TINY_SPEECH_CONFIG)
    input_ids, speaker_position = build_speech_prompt(
        CharacterTokenizer(), speech_config, "hi"
    )
    assert input_ids[speaker_position - 1 : speaker_position + 2] == [
        speech_config.speaker_start_token_id,
        speech_config.audio_patch_token_id,
        speech_config.speaker_end_token_id,
    ]
    assert input_ids.count(speech_config.audio_patch_token_id) == 1


def test_prompt_ends_with_target_text_assistant_turn_and_audio_start() -> None:
    speech_config = SpeechConfig.model_validate(TINY_SPEECH_CONFIG)
    input_ids, _ = build_speech_prompt(CharacterTokenizer(), speech_config, "hi")
    turn_ids = [ord(character) for character in "hi" + ASSISTANT_PREFIX]
    expected_suffix = turn_ids + [speech_config.audio_start_token_id]
    assert input_ids[-len(expected_suffix) :] == expected_suffix


@pytest.mark.parametrize(
    ("field_name", "unsupported_value"),
    [
        ("prompt_format", "thinking_v1"),
        ("with_rml", False),
        ("decode_alignment", "unshifted"),
        ("patch_size", 4),
    ],
)
def test_speech_config_rejects_unsupported_checkpoint_contracts(
    field_name: str, unsupported_value: JsonValue
) -> None:
    with pytest.raises(ValidationError):
        SpeechConfig.model_validate(
            {**TINY_SPEECH_CONFIG, field_name: unsupported_value}
        )
