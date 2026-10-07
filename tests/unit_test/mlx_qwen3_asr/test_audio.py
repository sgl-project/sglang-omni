# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from transformers import WhisperFeatureExtractor  # noqa: E402

from sglang_omni_mlx.qwen3_asr.audio import (  # noqa: E402
    HOP_LENGTH,
    MEL_FILTERS,
    AudioLayout,
    decode_wav,
    log_mel,
    peak_is_silent,
    reference_token_count,
    swift_token_count,
    token_count,
)


def pcm16_wav(
    samples: np.ndarray, sample_rate: int = 16000, channel_count: int = 1
) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channel_count)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes((samples * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def float32_wav(samples: np.ndarray) -> bytes:
    data = samples.astype("<f4").tobytes()
    fmt = (
        b"fmt "
        + (16).to_bytes(4, "little")
        + (3).to_bytes(2, "little")
        + (1).to_bytes(2, "little")
        + (16000).to_bytes(4, "little")
        + (64000).to_bytes(4, "little")
        + (4).to_bytes(2, "little")
        + (32).to_bytes(2, "little")
    )
    body = b"WAVE" + fmt + b"data" + len(data).to_bytes(4, "little") + data
    return b"RIFF" + len(body).to_bytes(4, "little") + body


def speech_like_audio(seconds: float) -> np.ndarray:
    generator = np.random.default_rng(0)
    times = np.arange(int(seconds * 16000)) / 16000
    tone = 0.3 * np.sin(2 * np.pi * 220 * times) * np.sin(2 * np.pi * 3 * times)
    return (tone + 0.02 * generator.standard_normal(times.size)).astype(np.float32)


def test_pcm16_and_float32_wavs_decode_to_the_same_samples() -> None:
    samples = speech_like_audio(0.5)
    from_pcm16 = decode_wav(pcm16_wav(samples))
    from_float32 = decode_wav(float32_wav(samples))
    assert from_pcm16.dtype == np.float32
    np.testing.assert_allclose(from_pcm16, samples, atol=1e-4)
    np.testing.assert_array_equal(from_float32, samples)


@pytest.mark.parametrize(
    "wav_bytes",
    [
        b"not a wav file",
        pcm16_wav(np.zeros(160, dtype=np.float32), sample_rate=44100),
        pcm16_wav(np.zeros(320, dtype=np.float32), channel_count=2),
    ],
)
def test_decode_wav_rejects_audio_other_than_16khz_mono(wav_bytes: bytes) -> None:
    with pytest.raises(ValueError):
        decode_wav(wav_bytes)


def test_mel_filters_match_the_checkpoint_feature_extractor() -> None:
    extractor = WhisperFeatureExtractor(feature_size=128)
    np.testing.assert_allclose(np.array(MEL_FILTERS), extractor.mel_filters, atol=1e-6)


def test_reference_log_mel_matches_the_feature_extractor() -> None:
    samples = speech_like_audio(2.3)
    extractor = WhisperFeatureExtractor(feature_size=128)
    expected = extractor(
        samples,
        sampling_rate=16000,
        return_tensors="np",
        padding="longest",
        truncation=False,
    ).input_features[0]
    actual = np.array(log_mel(samples, AudioLayout.REFERENCE))
    assert actual.shape == expected.shape == (128, len(samples) // HOP_LENGTH)
    np.testing.assert_allclose(actual, expected, atol=5e-4)


def test_swift_layout_keeps_the_final_stft_frame() -> None:
    samples = speech_like_audio(1.0)
    reference = np.array(log_mel(samples, AudioLayout.REFERENCE))
    swift = np.array(log_mel(samples, AudioLayout.VOXT_SWIFT))
    assert swift.shape[1] == reference.shape[1] + 1
    # The extra frame can raise the clip maximum, which moves the floor; the
    # spectrum itself is the same.
    np.testing.assert_allclose(swift[:, :-1], reference, atol=1e-5)


@pytest.mark.parametrize(("frame_count", "expected"), [(679, 98), (553, 78), (306, 40)])
def test_swift_token_count_credits_a_partial_chunk(
    frame_count: int, expected: int
) -> None:
    assert swift_token_count(frame_count) == expected
    assert token_count(frame_count, AudioLayout.VOXT_SWIFT) == expected


@pytest.mark.parametrize(
    ("frame_count", "expected"), [(100, 13), (200, 26), (679, 88), (553, 72), (306, 40)]
)
def test_reference_token_count_follows_the_convolutions(
    frame_count: int, expected: int
) -> None:
    assert reference_token_count(frame_count) == expected
    assert token_count(frame_count, AudioLayout.REFERENCE) == expected


def test_peak_is_silent() -> None:
    assert peak_is_silent(np.zeros(0, dtype=np.float32), 1e-3)
    assert peak_is_silent(np.full(10, 5e-4, dtype=np.float32), 1e-3)
    assert not peak_is_silent(np.array([0.0, -0.01], dtype=np.float32), 1e-3)


@pytest.mark.parametrize("sample_count", [0, 1, 48, 150, 199, 399])
@pytest.mark.parametrize("layout", [AudioLayout.REFERENCE, AudioLayout.VOXT_SWIFT])
def test_audio_shorter_than_one_fft_window_still_has_frames(
    sample_count: int, layout: AudioLayout
) -> None:
    samples = speech_like_audio(1.0)[:sample_count]
    mel = np.array(log_mel(samples, layout))
    assert mel.shape[0] == 128 and mel.shape[1] >= 1
    assert np.isfinite(mel).all()
