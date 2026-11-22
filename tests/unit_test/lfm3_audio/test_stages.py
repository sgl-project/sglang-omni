# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the LFM3-Audio prompt helpers and pipeline registration."""

from __future__ import annotations

import pytest

from sglang_omni.models.lfm3_audio import stages
from sglang_omni.models.lfm3_audio.config import Lfm3AudioPipelineConfig


def test_tts_prompt_uses_voice_and_language() -> None:
    assert (
        stages.tts_system_prompt({})
        == "Perform TTS in American English using a female voice."
    )
    assert (
        stages.tts_system_prompt({"voice": "Male", "language": "British English"})
        == "Perform TTS in British English using a male voice."
    )


def test_tts_instructions_replace_the_prompt() -> None:
    assert (
        stages.tts_system_prompt({"instructions": " Speak slowly. "}) == "Speak slowly."
    )


def test_tts_rejects_unknown_voices() -> None:
    with pytest.raises(ValueError, match="voices"):
        stages.tts_system_prompt({"voice": "alloy"})


def test_tts_sampling_fields_drive_the_speech_frames() -> None:
    params = {"temperature": 0.0, "top_k": 64, "repetition_penalty": 1.2}
    assert stages.audio_sampling(params, is_tts=True) == {
        "audio_temperature": 0.0,
        "audio_top_k": 64,
        "audio_repetition_penalty": 1.2,
    }
    assert stages.audio_sampling(params, is_tts=False) == {
        "audio_temperature": None,
        "audio_top_k": None,
        "audio_repetition_penalty": None,
    }


def test_chat_talker_fields_drive_the_speech_frames() -> None:
    params = {"temperature": 0.7, "talker_temperature": 0.0, "talker_top_k": 32}
    assert stages.audio_sampling(params, is_tts=False) == {
        "audio_temperature": 0.0,
        "audio_top_k": 32,
        "audio_repetition_penalty": None,
    }


def test_image_and_video_parts() -> None:
    assert (
        stages.media_source(
            {"type": "image_url", "image_url": {"url": "a.png"}}, "image"
        )
        == "a.png"
    )
    assert (
        stages.media_source({"type": "video_url", "video_url": "b.mp4"}, "video")
        == "b.mp4"
    )
    assert stages.media_source({"type": "image", "image": "c.png"}, "image") == "c.png"


def test_video_data_url_becomes_a_temp_file() -> None:
    import base64
    import os

    path = stages.video_file(
        "data:video/mp4;base64," + base64.b64encode(b"abc").decode()
    )
    try:
        assert path.endswith(".mp4")
        with open(path, "rb") as handle:
            assert handle.read() == b"abc"
    finally:
        os.unlink(path)
    assert stages.video_file("https://example.com/v.mp4") == "https://example.com/v.mp4"


def test_input_audio_part_becomes_a_data_url() -> None:
    part = {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}
    assert stages.audio_source(part) == "data:audio/wav;base64,AAAA"


def test_pipeline_stages() -> None:
    config = Lfm3AudioPipelineConfig(model_path="LiquidAI/LFM3-Audio-2B")
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "generation",
        "decode",
    ]
    assert config.stages[-1].terminal
