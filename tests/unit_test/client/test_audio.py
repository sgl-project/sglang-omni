import io

import numpy as np
import pytest

from sglang_omni.client.audio import (
    FORMAT_MIME_TYPES,
    PYAV_ENCODE_CONFIGS,
    _encode_with_pyav,
    _resample_linear,
    audio_to_base64,
    encode_audio,
    encode_pcm,
    encode_wav,
    to_numpy,
)


def _stereo_test_signal() -> tuple[np.ndarray, int]:
    sample_rate = 32000
    time = np.arange(sample_rate // 2, dtype=np.float32) / sample_rate
    audio = np.stack(
        [
            0.35 * np.sin(2 * np.pi * 440 * time),
            0.35 * np.sin(2 * np.pi * 880 * time),
        ]
    )
    return audio, sample_rate


def _decode_audio(encoded: bytes) -> tuple[int, np.ndarray]:
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


def _assert_flac(encoded: bytes) -> None:
    import soundfile as sf

    with sf.SoundFile(io.BytesIO(encoded)) as sound_file:
        assert sound_file.format == "FLAC"


def _dominant_frequency(audio: np.ndarray, sample_rate: int) -> float:
    window = np.hanning(audio.size)
    spectrum = np.abs(np.fft.rfft(audio * window))
    frequencies = np.fft.rfftfreq(audio.size, 1 / sample_rate)
    return float(frequencies[1 + np.argmax(spectrum[1:])])


def _assert_stereo_content(audio: np.ndarray, sample_rate: int) -> None:
    assert audio.shape[0] == 2
    assert _dominant_frequency(audio[0], sample_rate) == pytest.approx(440, abs=20)
    assert _dominant_frequency(audio[1], sample_rate) == pytest.approx(880, abs=20)


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


def test_encode_wav():
    # Generate 1 sec sine wave
    sr = 24000
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    wav_bytes = encode_wav(audio, sr)
    assert isinstance(wav_bytes, bytes)
    assert wav_bytes.startswith(b"RIFF")
    assert b"WAVE" in wav_bytes[:16]


def test_encode_pcm():
    audio = np.array([0.0, 0.5, -1.0, 1.2], dtype=np.float32)
    pcm_bytes = encode_pcm(audio, 16000)
    assert isinstance(pcm_bytes, bytes)
    # Check expected int16 values (clamped to [-1, 1]): [0, 16383, -32767, 32767]
    expected = np.array([0, 16383, -32767, 32767], dtype=np.int16)
    actual = np.frombuffer(pcm_bytes, dtype=np.int16)
    assert np.array_equal(expected, actual)


def test_resample_linear_preserves_stereo_channels():
    orig_sr = 32000
    target_sr = 24000
    audio = np.stack(
        [
            np.linspace(-1.0, 1.0, 32, dtype=np.float32),
            np.linspace(1.0, -1.0, 32, dtype=np.float32),
        ]
    )

    resampled = _resample_linear(audio, orig_sr, target_sr)

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


@pytest.mark.parametrize("response_format", ["mp3", "aac", "opus"])
def test_pyav_encode_preserves_stereo_content(response_format: str):
    audio, sample_rate = _stereo_test_signal()
    config = PYAV_ENCODE_CONFIGS[response_format]

    encoded = _encode_with_pyav(
        audio,
        sample_rate,
        container_format=config["container"],
        codecs=config["codecs"],
        valid_rates=config["valid_rates"],
    )

    decoded_sample_rate, decoded = _decode_audio(encoded)
    _assert_stereo_content(decoded, decoded_sample_rate)


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
    audio, sample_rate = _stereo_test_signal()
    encoded, mime = encode_audio(
        audio,
        response_format=response_format,
        sample_rate=sample_rate,
    )

    decoded_sample_rate, decoded = _decode_audio(encoded)

    assert mime == FORMAT_MIME_TYPES[response_format]
    assert decoded_sample_rate == expected_sample_rate
    _assert_stereo_content(decoded, decoded_sample_rate)


@pytest.mark.parametrize(
    "channel_last", [False, True], ids=["channel_first", "channel_last"]
)
def test_encode_audio_flac_preserves_stereo_orientations(channel_last: bool):
    audio, sample_rate = _stereo_test_signal()
    if channel_last:
        audio = audio.T

    encoded, mime = encode_audio(
        audio,
        response_format="flac",
        sample_rate=sample_rate,
    )

    assert mime == FORMAT_MIME_TYPES["flac"]
    _assert_flac(encoded)
    decoded_sample_rate, decoded = _decode_audio(encoded)
    assert decoded_sample_rate == sample_rate
    _assert_stereo_content(decoded, decoded_sample_rate)


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
    _assert_flac(encoded)
    decoded_sample_rate, decoded = _decode_audio(encoded)
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


def test_audio_to_base64():
    audio = np.array([0.0, 0.1, -0.1], dtype=np.float32)
    b64_str = audio_to_base64(audio, sample_rate=16000, output_format="pcm")
    assert isinstance(b64_str, str)
    # Decode to check
    import base64

    pcm_bytes = base64.b64decode(b64_str)
    actual = np.frombuffer(pcm_bytes, dtype=np.int16)
    expected = (audio * 32767.0).astype(np.int16)
    assert np.array_equal(expected, actual)
