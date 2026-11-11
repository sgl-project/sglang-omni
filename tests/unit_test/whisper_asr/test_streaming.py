# SPDX-License-Identifier: Apache-2.0
"""Whisper refreshes replace hypotheses without conditioning on partial text."""

import pytest

from sglang_omni.models.whisper_asr.streaming import WhisperStreamingStrategy


@pytest.mark.parametrize("language", ["en", "zh"])
def test_refresh_and_final_redecode_all_audio_without_previous_hypothesis(
    language: str | None,
) -> None:
    strategy = WhisperStreamingStrategy()
    state = strategy.create_state(model_name="whisper", language=language)
    first_request = strategy.build_decode_request(
        audio=b"first audio", state=state, is_final=False, request_id="partial-1"
    )
    assert first_request.prompt["audio_bytes"] == b"first audio"
    assert first_request.extra_params == {
        "task": "transcribe",
        "language": language or "en",
    }

    first_text = "识别错误" if language == "zh" else "a wrong word"
    final_text = "修订结果" if language == "zh" else "the corrected sentence"
    assert (
        strategy.update_hypothesis(
            generated_text=first_text, language=language, state=state
        )
        == first_text
    )
    for is_final in (False, True):
        request = strategy.build_decode_request(
            audio=b"first audio plus new audio",
            state=state,
            is_final=is_final,
            request_id="next",
        )
        assert request.prompt["audio_bytes"] == b"first audio plus new audio"
        assert request.extra_params == first_request.extra_params
        assert request.sampling.temperature == 0.0
        assert request.stream is False
        assert (
            strategy.update_hypothesis(
                generated_text=final_text, language=language, state=state
            )
            == final_text
        )


def test_reported_language_is_local_to_one_segment_state() -> None:
    strategy = WhisperStreamingStrategy()
    first_state = strategy.create_state(model_name="first", language="en")
    second_state = strategy.create_state(model_name="second", language="zh")
    strategy.update_hypothesis(
        generated_text="revised", language="english", state=first_state
    )
    for state, model_name, language in (
        (first_state, "first", "english"),
        (second_state, "second", "zh"),
    ):
        request = strategy.build_decode_request(
            audio=b"audio", state=state, is_final=True, request_id="final"
        )
        assert request.model == model_name
        assert request.extra_params["language"] == language


@pytest.mark.parametrize("is_final", [False, True])
def test_missing_language_detects_once_before_transcription(is_final: bool) -> None:
    strategy = WhisperStreamingStrategy()
    state = strategy.create_state(model_name="whisper", language=None)
    audio = b"current segment"
    detection = strategy.build_decode_request(
        audio=audio, state=state, is_final=is_final, request_id="detect"
    )
    assert detection.extra_params == {"task": "transcribe", "detect_language": True}
    assert detection.sampling.max_new_tokens == 1
    assert detection.prompt["audio_bytes"] == audio
    assert (
        strategy.update_hypothesis(generated_text="zh", language="english", state=state)
        is None
    )
    for final in (False, True):
        request = strategy.build_decode_request(
            audio=audio, state=state, is_final=final, request_id="transcribe"
        )
        assert request.extra_params == {"task": "transcribe", "language": "zh"}
        assert request.prompt["audio_bytes"] == audio
        assert (
            strategy.update_hypothesis(
                generated_text="你好", language="zh", state=state
            )
            == "你好"
        )


def test_empty_language_detection_does_not_fall_back_to_english() -> None:
    strategy = WhisperStreamingStrategy()
    state = strategy.create_state(model_name="whisper", language=None)
    with pytest.raises(ValueError, match="returned no language"):
        strategy.update_hypothesis(generated_text="", language="english", state=state)
    assert state.language is None
