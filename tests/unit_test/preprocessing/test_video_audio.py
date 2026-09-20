# SPDX-License-Identifier: Apache-2.0
"""Video audio decoding must not depend on librosa's file-format support."""

import logging
from pathlib import Path

import av
import librosa
import numpy as np
import pytest

from sglang_omni.preprocessing.video import extract_audio_from_path


@pytest.mark.parametrize("sample_rate", [16000, 44100])
def test_extract_aac_from_mp4(tmp_path: Path, monkeypatch, sample_rate: int) -> None:
    samples = (
        0.1 * np.sin(2 * np.pi * 440 * np.arange(sample_rate) / sample_rate)
    ).astype(np.float32)
    path = tmp_path / "audio.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("aac", rate=sample_rate)
        frame = av.AudioFrame.from_ndarray(
            np.stack([samples, samples]), format="fltp", layout="stereo"
        )
        frame.sample_rate = sample_rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    def fail_load(*args, **kwargs):
        pytest.fail("Video audio must be decoded by PyAV, not librosa.load")

    monkeypatch.setattr(librosa, "load", fail_load)
    audio = extract_audio_from_path(path, 16000)

    assert audio is not None
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert 16000 <= len(audio) < 18000
    assert np.isfinite(audio).all()
    assert 0.05 < np.max(np.abs(audio)) < 0.2
    frequency = np.argmax(np.abs(np.fft.rfft(audio[:16000])))
    assert frequency == 440


@pytest.mark.parametrize(
    "codec,format,dtype,scale",
    [
        ("pcm_s16le", "s16", np.int16, 32768),
        ("pcm_s32le", "s32", np.int32, 2147483648),
        ("pcm_f32le", "flt", np.float32, 1),
    ],
)
def test_packed_stereo_keeps_duration_and_scale(
    tmp_path: Path, codec: str, format: str, dtype, scale: int
) -> None:
    sample_rate = 16000
    values = np.tile([0.25, -0.125], (sample_rate, 1))
    packed = (values * scale).astype(dtype).reshape(1, -1)
    path = tmp_path / "audio.mkv"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream(codec, rate=sample_rate)
        stream.layout = "stereo"
        frame = av.AudioFrame.from_ndarray(packed, format=format, layout="stereo")
        frame.sample_rate = sample_rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    audio = extract_audio_from_path(path, sample_rate)

    assert audio is not None
    assert audio.shape == (sample_rate,)
    assert audio.dtype == np.float32
    np.testing.assert_allclose(audio, 0.0625, atol=1e-7)


def test_video_without_audio_returns_none(tmp_path: Path, caplog) -> None:
    path = tmp_path / "silent.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=25)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame.from_ndarray(
            np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    with caplog.at_level(logging.WARNING):
        assert extract_audio_from_path(path, 16000) is None
    assert not caplog.records


def test_broken_media_logs_decoding_failure(tmp_path: Path, caplog) -> None:
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not a media container")

    with caplog.at_level(logging.WARNING):
        assert extract_audio_from_path(path, 16000) is None
    assert "Failed to extract audio" in caplog.text
