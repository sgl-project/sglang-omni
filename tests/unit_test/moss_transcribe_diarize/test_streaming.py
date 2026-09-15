# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from sglang_omni.models.moss_transcribe_diarize.streaming import (
    MossTranscribeDiarizeStreamingStrategy,
)


def test_moss_streaming_builds_cumulative_audio_request() -> None:
    strategy = MossTranscribeDiarizeStreamingStrategy()
    state = strategy.create_state(
        model_name="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        language="zh",
    )

    request = strategy.build_decode_request(
        audio=b"wav-refresh",
        state=state,
        is_final=False,
        request_id="session:0:1",
    )

    assert request.model == "OpenMOSS-Team/MOSS-Transcribe-Diarize"
    assert request.prompt == {
        "audio_bytes": b"wav-refresh",
        "filename": "realtime-segment.wav",
        "content_type": "audio/wav",
    }
    assert request.extra_params == {"task": "transcribe", "language": "zh"}
    assert request.stream is False
    assert request.output_modalities == ["text"]


def test_moss_streaming_updates_language_and_hypothesis() -> None:
    strategy = MossTranscribeDiarizeStreamingStrategy()
    state = strategy.create_state(model_name="moss", language=None)

    first = strategy.update_hypothesis(
        generated_text="[0.00][S01] 你好[1.20]",
        language="Chinese",
        state=state,
    )
    second = strategy.update_hypothesis(
        generated_text="[0.00][S01] 你好，世界。[2.10]",
        language=None,
        state=state,
    )

    assert first == "[0.00][S01] 你好[1.20]"
    assert second == "[0.00][S01] 你好，世界。[2.10]"
    assert state.transcript == second
    assert state.language == "Chinese"
    assert state.chunk_id == 2


def test_moss_streaming_states_are_isolated() -> None:
    strategy = MossTranscribeDiarizeStreamingStrategy()
    first = strategy.create_state(model_name="moss", language="en")
    second = strategy.create_state(model_name="moss", language="zh")

    strategy.update_hypothesis(
        generated_text="first",
        language="English",
        state=first,
    )

    assert first.transcript == "first"
    assert first.language == "English"
    assert second.transcript == ""
    assert second.language == "zh"
