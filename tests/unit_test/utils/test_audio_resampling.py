# SPDX-License-Identifier: Apache-2.0
"""Audio resampling regression tests."""

from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import pytest
import soundfile as sf
from numpy.typing import NDArray

from sglang_omni.utils import audio
from sglang_omni.utils.audio import load_audio


def resampling_signal_wav_bytes(
    sample_rate: int = 24000,
) -> tuple[bytes, NDArray[np.float32]]:
    sample_positions = np.arange(sample_rate // 5, dtype=np.float64) / sample_rate
    signal = (
        0.45 * np.sin(2 * np.pi * 431 * sample_positions)
        + 0.25 * np.sin(2 * np.pi * 6173 * sample_positions)
        + 0.1 * np.sin(2 * np.pi * 9101 * sample_positions)
    )
    pcm_samples = np.round(signal * 32767).astype("<i2")
    decoded_samples = pcm_samples.astype(np.float32) / 32768.0
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_samples.tobytes())
    return buffer.getvalue(), decoded_samples


def resampling_stereo_audio_bytes(
    container_format: Literal["wav", "flac"],
) -> tuple[bytes, NDArray[np.float32]]:
    sample_rate = 24000
    sample_positions = np.arange(sample_rate // 5, dtype=np.float64) / sample_rate
    left_channel = 0.45 * np.sin(2 * np.pi * 431 * sample_positions) + 0.25 * np.sin(
        2 * np.pi * 6173 * sample_positions
    )
    right_channel = 0.4 * np.sin(2 * np.pi * 733 * sample_positions) + 0.2 * np.sin(
        2 * np.pi * 8129 * sample_positions
    )
    pcm_samples = np.round(
        np.stack((left_channel, right_channel), axis=1) * 32767
    ).astype("<i2")
    decoded_samples = pcm_samples.astype(np.float32).T / 32768.0
    buffer = io.BytesIO()
    if container_format == "wav":
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm_samples.tobytes())
    elif container_format == "flac":
        sf.write(
            buffer,
            pcm_samples,
            sample_rate,
            format="FLAC",
            subtype="PCM_16",
        )
    else:
        raise ValueError(f"Unsupported test audio format: {container_format}")
    return buffer.getvalue(), decoded_samples


@pytest.mark.parametrize(
    ("container_format", "mono"),
    [("wav", True), ("wav", False), ("flac", True), ("flac", False)],
)
def test_load_audio_soxr_hq_resamples_stereo_audio(
    container_format: Literal["wav", "flac"],
    mono: bool,
) -> None:
    encoded_audio, source_samples = resampling_stereo_audio_bytes(container_format)
    expected_source = source_samples.mean(axis=0) if mono else source_samples
    expected = librosa.resample(
        expected_source,
        orig_sr=24000,
        target_sr=16000,
        res_type="soxr_hq",
        axis=-1,
    )

    actual = load_audio(
        encoded_audio,
        target_sample_rate=16000,
        mono=mono,
        resampling_backend="soxr_hq",
    )

    assert actual.dtype == np.float32
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)


def test_load_audio_soxr_hq_resamples_wav_bytes_like_librosa() -> None:
    wav, source_samples = resampling_signal_wav_bytes()
    expected = librosa.resample(
        source_samples,
        orig_sr=24000,
        target_sr=16000,
        res_type="soxr_hq",
    )

    actual = load_audio(wav, resampling_backend="soxr_hq")

    assert actual.dtype == np.float32
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)


def test_load_audio_soxr_hq_resamples_decoded_wav_path_like_librosa(
    tmp_path: Path,
) -> None:
    wav, source_samples = resampling_signal_wav_bytes()
    path = tmp_path / "resampling-input.wav"
    path.write_bytes(wav)
    expected = librosa.resample(
        source_samples,
        orig_sr=24000,
        target_sr=16000,
        res_type="soxr_hq",
    )

    actual = load_audio(str(path), resampling_backend="soxr_hq")

    assert actual.dtype == np.float32
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)


def test_load_audio_soxr_hq_preserves_audio_at_target_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav, source_samples = resampling_signal_wav_bytes(sample_rate=16000)

    def fail_resample(
        waveform: NDArray[np.float32],
        *,
        orig_sr: int,
        target_sr: int,
        res_type: str,
    ) -> NDArray[np.float32]:
        raise AssertionError(
            f"Unexpected resampling for {waveform.shape} at {orig_sr}->{target_sr} "
            f"with {res_type}"
        )

    monkeypatch.setattr(audio.librosa, "resample", fail_resample)

    actual = load_audio(wav, target_sample_rate=16000, resampling_backend="soxr_hq")

    np.testing.assert_array_equal(actual, source_samples)


def test_load_audio_default_resampling_keeps_scipy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav, source_samples = resampling_signal_wav_bytes()
    expected = audio.resample_with_scipy(source_samples, 24000, 16000)
    monkeypatch.setattr(
        audio.current_platform,
        "supports_torchaudio_resample",
        lambda: False,
    )

    actual = load_audio(wav)

    np.testing.assert_array_equal(actual, expected)
