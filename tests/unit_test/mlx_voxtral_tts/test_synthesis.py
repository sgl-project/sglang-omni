# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from threading import Event

import pytest

mx = pytest.importorskip("mlx.core")

from sglang_omni_mlx.voxtral_tts.acoustic_head import END_AUDIO_TOKEN_ID  # noqa: E402
from sglang_omni_mlx.voxtral_tts.synthesis import (  # noqa: E402
    SpeechGenerationCancelledError,
    VoxtralSynthesizer,
)


class FakeTextModel:
    def embed_tokens(self, input_ids: mx.array) -> mx.array:
        return mx.zeros((input_ids.shape[0], input_ids.shape[1], 4))

    def __call__(self, hidden_states: mx.array, caches: list[int]) -> mx.array:
        return hidden_states


class FakeAcousticHead:
    acoustic_codebook_count: int = 2

    def __init__(self, semantic_codes: list[int]) -> None:
        self.semantic_codes: Iterator[int] = iter(semantic_codes)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        semantic_code = next(self.semantic_codes)
        return mx.array([[[semantic_code, 4, 5]]], dtype=mx.int32)


class FakeModel:
    def __init__(self, semantic_codes: list[int]) -> None:
        self.text_model: FakeTextModel = FakeTextModel()
        self.acoustic_head: FakeAcousticHead = FakeAcousticHead(semantic_codes)

    def new_caches(self) -> list[int]:
        return []

    def embed_audio_codes(self, audio_codes: mx.array) -> mx.array:
        return mx.zeros((1, 4))


def test_generate_codes_reports_length_when_end_token_is_not_generated() -> None:
    synthesizer = VoxtralSynthesizer(
        FakeModel([2, 3]), audio_token_id=24, max_audio_frames=2
    )

    result = synthesizer.generate_codes([10], None, Event(), seed=0)

    assert result.finish_reason == "length"
    assert result.audio_codes.tolist() == [[2, 4, 5], [3, 4, 5]]


def test_generate_codes_reports_stop_when_end_token_is_generated() -> None:
    synthesizer = VoxtralSynthesizer(
        FakeModel([END_AUDIO_TOKEN_ID]), audio_token_id=24, max_audio_frames=2
    )

    result = synthesizer.generate_codes([10], None, Event(), seed=0)

    assert result.finish_reason == "stop"
    assert result.audio_codes.shape == (0, 3)


def test_generate_codes_stops_when_request_is_cancelled() -> None:
    cancellation_event = Event()
    cancellation_event.set()
    synthesizer = VoxtralSynthesizer(
        FakeModel([2]), audio_token_id=24, max_audio_frames=2
    )

    with pytest.raises(SpeechGenerationCancelledError):
        synthesizer.generate_codes([10], None, cancellation_event, seed=0)
