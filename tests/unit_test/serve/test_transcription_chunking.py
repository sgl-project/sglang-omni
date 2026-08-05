# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from sglang_omni.config import AudioChunkingConfig
from sglang_omni.serve.transcription_chunking import plan_audio_chunks
from sglang_omni.utils.audio import load_audio

_SAMPLE_RATE = 16000


def _wav_bytes(waveform: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    sf.write(
        buffer,
        waveform,
        _SAMPLE_RATE,
        format="WAV",
        subtype="FLOAT",
    )
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("config", "duration_s"),
    [
        (AudioChunkingConfig(), 2.5),
        (
            AudioChunkingConfig(
                allow_audio_chunking=True,
                max_audio_clip_s=1.0,
            ),
            1.0,
        ),
    ],
)
def test_audio_within_policy_does_not_create_a_chunk_plan(
    config: AudioChunkingConfig,
    duration_s: float,
) -> None:
    waveform = np.zeros(int(duration_s * _SAMPLE_RATE), dtype=np.float32)

    assert plan_audio_chunks(_wav_bytes(waveform), config) is None


def test_long_audio_is_split_at_quiet_boundaries_without_loss() -> None:
    rng = np.random.default_rng(7)
    waveform = rng.uniform(-0.5, 0.5, int(2.5 * _SAMPLE_RATE)).astype(np.float32)
    waveform[int(0.8 * _SAMPLE_RATE) : int(0.9 * _SAMPLE_RATE)] = 0.0
    config = AudioChunkingConfig(
        allow_audio_chunking=True,
        max_audio_clip_s=1.0,
    )

    plan = plan_audio_chunks(_wav_bytes(waveform), config)

    assert plan is not None
    assert plan.sample_rate == _SAMPLE_RATE
    assert plan.duration_s == 2.5
    assert (
        int(0.8 * _SAMPLE_RATE) <= plan.spans[0].end_sample <= int(0.9 * _SAMPLE_RATE)
    )
    expected_start = 0
    for span in plan.spans:
        assert span.start_sample == expected_start
        assert 0 < span.end_sample - span.start_sample <= _SAMPLE_RATE
        expected_start = span.end_sample
    assert expected_start == waveform.size


def test_encoded_chunks_round_trip_sample_for_sample() -> None:
    waveform = np.linspace(-0.5, 0.5, int(2.5 * _SAMPLE_RATE), dtype=np.float32)
    config = AudioChunkingConfig(
        allow_audio_chunking=True,
        max_audio_clip_s=1.0,
    )
    plan = plan_audio_chunks(_wav_bytes(waveform), config)

    assert plan is not None
    for span in plan.spans:
        decoded = load_audio(plan.encode(span), target_sample_rate=_SAMPLE_RATE)
        np.testing.assert_array_equal(
            decoded,
            waveform[span.start_sample : span.end_sample],
        )


def test_planner_avoids_a_subminimum_tail_chunk() -> None:
    waveform = np.zeros(int(1.1 * _SAMPLE_RATE), dtype=np.float32)
    config = AudioChunkingConfig(
        allow_audio_chunking=True,
        max_audio_clip_s=1.0,
    )

    plan = plan_audio_chunks(_wav_bytes(waveform), config)

    assert plan is not None
    assert [span.end_sample - span.start_sample for span in plan.spans] == [
        int(0.6 * _SAMPLE_RATE),
        int(0.5 * _SAMPLE_RATE),
    ]
