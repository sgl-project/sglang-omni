import io
import wave

import numpy as np
import pytest

from sglang_omni.client.audio import (
    FORMAT_MIME_TYPES,
    encode_audio,
    encode_pcm,
    resample_linear,
    select_audio_delta,
    to_numpy,
)


def stereo_test_signal() -> tuple[np.ndarray, int]:
    sample_rate = 32000
    time = np.arange(sample_rate // 2, dtype=np.float32) / sample_rate
    audio = np.stack(
        [
            0.35 * np.sin(2 * np.pi * 440 * time),
            0.35 * np.sin(2 * np.pi * 880 * time),
        ]
    )
    return audio, sample_rate


def decode_audio(encoded: bytes) -> tuple[int, np.ndarray]:
    import av

    with av.open(io.BytesIO(encoded)) as container:
        stream = container.streams.audio[0]
        sample_rate = stream.sample_rate
        channels = len(stream.layout.channels)
        chunks = []
        for frame in container.decode(stream):
            chunk = frame.to_ndarray()
            if frame.format.is_planar:
                chunk = chunk.reshape(channels, -1)
            else:
                chunk = chunk.reshape(-1, channels).T
            chunks.append(chunk)
    return sample_rate, np.concatenate(chunks, axis=-1)


def assert_flac(encoded: bytes) -> None:
    import soundfile as sf

    with sf.SoundFile(io.BytesIO(encoded)) as sound_file:
        assert sound_file.format == "FLAC"


def dominant_frequency(audio: np.ndarray, sample_rate: int) -> float:
    window = np.hanning(audio.size)
    spectrum = np.abs(np.fft.rfft(audio * window))
    frequencies = np.fft.rfftfreq(audio.size, 1 / sample_rate)
    return float(frequencies[1 + np.argmax(spectrum[1:])])


def assert_stereo_content(audio: np.ndarray, sample_rate: int) -> None:
    assert audio.shape[0] == 2
    assert dominant_frequency(audio[0], sample_rate) == pytest.approx(440, abs=20)
    assert dominant_frequency(audio[1], sample_rate) == pytest.approx(880, abs=20)


def test_to_numpy():
    # Numpy array
    arr = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    res = to_numpy(arr)
    assert np.allclose(arr, res)
    assert res.dtype == np.float32

    # List/tuple
    res_list = to_numpy([0.1, -0.2, 0.3])
    assert np.allclose(arr, res_list)
    assert res_list.dtype == np.float32

    # PCM16 bytes
    pcm_bytes = b"\x00\x00\x00\x40\x00\x80"  # [0, 16384, -32768] in int16
    expected = np.array([0.0, 0.5, -1.0], dtype=np.float32)
    res_bytes = to_numpy(pcm_bytes)
    assert np.allclose(expected, res_bytes, atol=1e-4)


def test_encode_pcm():
    audio = np.array([0.0, 0.5, -1.0, 1.2], dtype=np.float32)
    pcm_bytes = encode_pcm(audio, 16000)
    assert isinstance(pcm_bytes, bytes)
    # Check expected int16 values (clamped to [-1, 1]): [0, 16383, -32767, 32767]
    expected = np.array([0, 16383, -32767, 32767], dtype=np.int16)
    actual = np.frombuffer(pcm_bytes, dtype=np.int16)
    assert np.array_equal(expected, actual)


@pytest.mark.parametrize("is_terminal", [False, True])
def test_select_single_sample_audio_delta(is_terminal: bool) -> None:
    """A one-sample mono chunk must not disappear or be replayed at completion."""
    audio = np.array([[0.5]], dtype=np.float32)
    delta, emitted_samples = select_audio_delta(
        audio, emitted_samples=0, is_terminal=is_terminal
    )

    assert delta is not None
    assert delta.shape == (1,)
    np.testing.assert_array_equal(delta, np.array([0.5], dtype=np.float32))
    assert emitted_samples == 1
    terminal_delta, terminal_samples = select_audio_delta(
        audio, emitted_samples=emitted_samples, is_terminal=True
    )
    assert terminal_delta is None
    assert terminal_samples == 1


def test_encode_single_sample_audio_with_speed() -> None:
    """A speed change must not fail on one-sample mono audio."""
    audio = np.array([[0.5]], dtype=np.float32)
    encoded, mime = encode_audio(
        audio, response_format="wav", sample_rate=24000, speed=0.5
    )

    assert mime == FORMAT_MIME_TYPES["wav"]
    with wave.open(io.BytesIO(encoded)) as waveform:
        assert waveform.getframerate() == 24000
        assert waveform.getnchannels() == 1
        assert waveform.getnframes() == 2
        assert waveform.readframes(2) == b"\xff\x3f\xff\x3f"


@pytest.mark.parametrize("response_format", ["wav", "pcm"])
def test_encode_empty_audio_with_speed(response_format: str) -> None:
    """Changing speed must not try to interpolate an empty mono waveform."""
    encoded, mime = encode_audio(
        np.array([], dtype=np.float32),
        response_format=response_format,
        sample_rate=24000,
        speed=1.5,
    )

    assert mime == FORMAT_MIME_TYPES[response_format]
    if response_format == "wav":
        with wave.open(io.BytesIO(encoded)) as waveform:
            assert waveform.getnframes() == 0
            assert waveform.getnchannels() == 1
            assert waveform.getframerate() == 24000
    else:
        assert encoded == b""


def test_resample_linear_preserves_stereo_channels():
    orig_sr = 32000
    target_sr = 24000
    audio = np.stack(
        [
            np.linspace(-1.0, 1.0, 32, dtype=np.float32),
            np.linspace(1.0, -1.0, 32, dtype=np.float32),
        ]
    )

    resampled = resample_linear(audio, orig_sr, target_sr)

    expected_samples = 24
    assert resampled.shape == (2, expected_samples)
    assert resampled.dtype == np.float32
    new_idx = np.linspace(0.0, 31.0, expected_samples)
    expected = np.stack(
        [
            np.interp(new_idx, np.arange(32), audio[0]),
            np.interp(new_idx, np.arange(32), audio[1]),
        ]
    ).astype(np.float32)
    assert np.allclose(resampled, expected)


@pytest.mark.parametrize(
    ("response_format", "expected_sample_rate"),
    [
        ("wav", 32000),
        ("mp3", 32000),
        ("opus", 48000),
        ("aac", 32000),
    ],
)
def test_encode_audio_preserves_stereo_content(
    response_format: str, expected_sample_rate: int
):
    audio, sample_rate = stereo_test_signal()
    encoded, mime = encode_audio(
        audio,
        response_format=response_format,
        sample_rate=sample_rate,
    )

    decoded_sample_rate, decoded = decode_audio(encoded)

    assert mime == FORMAT_MIME_TYPES[response_format]
    assert decoded_sample_rate == expected_sample_rate
    assert_stereo_content(decoded, decoded_sample_rate)


@pytest.mark.parametrize(
    "channel_last", [False, True], ids=["channel_first", "channel_last"]
)
def test_encode_audio_flac_preserves_stereo_orientations(channel_last: bool):
    audio, sample_rate = stereo_test_signal()
    if channel_last:
        audio = audio.T

    encoded, mime = encode_audio(
        audio,
        response_format="flac",
        sample_rate=sample_rate,
    )

    assert mime == FORMAT_MIME_TYPES["flac"]
    assert_flac(encoded)
    decoded_sample_rate, decoded = decode_audio(encoded)
    assert decoded_sample_rate == sample_rate
    assert_stereo_content(decoded, decoded_sample_rate)


def test_encode_audio_flac_preserves_mono():
    sample_rate = 32000
    time = np.arange(sample_rate // 2, dtype=np.float32) / sample_rate
    audio = 0.35 * np.sin(2 * np.pi * 440 * time)

    encoded, mime = encode_audio(
        audio,
        response_format="flac",
        sample_rate=sample_rate,
    )

    assert mime == FORMAT_MIME_TYPES["flac"]
    assert_flac(encoded)
    decoded_sample_rate, decoded = decode_audio(encoded)
    assert decoded_sample_rate == sample_rate
    assert decoded.shape[0] == 1


def test_encode_audio_opus():
    sr = 24000
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    encoded_bytes, mime = encode_audio(audio, response_format="opus", sample_rate=sr)
    assert mime == FORMAT_MIME_TYPES["opus"]
    try:
        assert encoded_bytes.startswith(b"OggS")
    except ImportError:
        assert encoded_bytes.startswith(b"RIFF")


def test_encode_audio_opus_resampling():
    sr = 44100  # invalid sample rate for Opus
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    encoded_bytes, mime = encode_audio(audio, response_format="opus", sample_rate=sr)
    assert mime == FORMAT_MIME_TYPES["opus"]
    try:
        assert encoded_bytes.startswith(b"OggS")
    except ImportError:
        assert encoded_bytes.startswith(b"RIFF")


def test_encode_audio_aac():
    sr = 24000
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    encoded_bytes, mime = encode_audio(audio, response_format="aac", sample_rate=sr)
    assert mime == FORMAT_MIME_TYPES["aac"]
    try:
        assert encoded_bytes[0] == 0xFF
        assert (encoded_bytes[1] & 0xF0) == 0xF0
    except ImportError:
        assert encoded_bytes.startswith(b"RIFF")


def test_encode_audio_aac_resampling():
    sr = 12345  # invalid sample rate for AAC
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    encoded_bytes, mime = encode_audio(audio, response_format="aac", sample_rate=sr)
    assert mime == FORMAT_MIME_TYPES["aac"]
    try:
        assert encoded_bytes[0] == 0xFF
        assert (encoded_bytes[1] & 0xF0) == 0xF0
    except ImportError:
        assert encoded_bytes.startswith(b"RIFF")


def test_encode_audio_mp3():
    sr = 24000
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    encoded_bytes, mime = encode_audio(audio, response_format="mp3", sample_rate=sr)
    assert mime == FORMAT_MIME_TYPES["mp3"]
    try:
        assert encoded_bytes.startswith(b"ID3")
    except ImportError:
        assert encoded_bytes.startswith(b"RIFF")


def test_encode_audio_mp3_resampling():
    sr = 12345  # invalid sample rate for MP3
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    encoded_bytes, mime = encode_audio(audio, response_format="mp3", sample_rate=sr)
    assert mime == FORMAT_MIME_TYPES["mp3"]
    try:
        assert encoded_bytes.startswith(b"ID3")
    except ImportError:
        assert encoded_bytes.startswith(b"RIFF")
