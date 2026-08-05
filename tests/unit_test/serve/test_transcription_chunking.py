# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from sglang_omni.config import AudioChunkingConfig
from sglang_omni.serve.transcription_chunking import (
    check_total_duration,
    needs_chunking,
)

_ENABLED = AudioChunkingConfig(allow_audio_chunking=True, max_audio_clip_s=60.0)


def test_disabled_config_never_chunks() -> None:
    config = AudioChunkingConfig(max_audio_clip_s=60.0)

    assert config.allow_audio_chunking is False
    assert needs_chunking(600.0, config) is False


def test_unknown_duration_is_left_alone() -> None:
    # 0.0 is what _probe_audio_duration returns when it cannot read the header.
    assert needs_chunking(0.0, _ENABLED) is False
    assert needs_chunking(-1.0, _ENABLED) is False


def test_duration_exactly_at_the_clip_limit_is_not_chunked() -> None:
    assert needs_chunking(60.0, _ENABLED) is False


def test_duration_past_the_clip_limit_is_chunked() -> None:
    assert needs_chunking(60.1, _ENABLED) is True


def test_chunk_samples_floors_and_stays_positive() -> None:
    assert _ENABLED.chunk_samples(16000) == 960_000
    assert AudioChunkingConfig(max_audio_clip_s=0.5).chunk_samples(16000) == 8000
    # Rates so low that the product rounds to zero still yield one sample.
    assert AudioChunkingConfig(max_audio_clip_s=0.5).chunk_samples(1) == 1


def test_total_duration_limit_can_be_disabled() -> None:
    config = AudioChunkingConfig(allow_audio_chunking=True, max_total_audio_s=None)

    check_total_duration(100_000.0, config)


def test_total_duration_at_the_limit_is_accepted() -> None:
    config = AudioChunkingConfig(allow_audio_chunking=True, max_total_audio_s=3600.0)

    check_total_duration(3600.0, config)


def test_total_duration_past_the_limit_is_rejected() -> None:
    config = AudioChunkingConfig(allow_audio_chunking=True, max_total_audio_s=3600.0)

    with pytest.raises(ValueError) as exc_info:
        check_total_duration(3600.5, config)

    message = str(exc_info.value)
    assert "3600" in message
    assert "3600.500" in message
    # Shared wording with the engine-side limit keeps both mapped to HTTP 400.
    assert "accepts audio up to" in message


def test_total_limit_below_clip_limit_is_rejected_at_config_time() -> None:
    with pytest.raises(ValueError, match="max_total_audio_s"):
        AudioChunkingConfig(max_audio_clip_s=60.0, max_total_audio_s=30.0)


@pytest.mark.parametrize(
    "field, value",
    [
        ("max_audio_clip_s", 0.0),
        ("max_audio_clip_s", -1.0),
        ("max_total_audio_s", 0.0),
    ],
)
def test_out_of_range_config_values_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        AudioChunkingConfig(**{field: value})