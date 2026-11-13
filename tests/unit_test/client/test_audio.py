import asyncio
import io

import numpy as np
import pytest

from sglang_omni.client.audio import (
    FORMAT_MIME_TYPES,
    encode_audio,
    encode_pcm,
    resample_linear,
    to_numpy,
)
from sglang_omni.client.types import GenerateRequest
from tests.unit_test.fixtures.speech_fakes import FixedSpeechClient


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


def test_output_sample_rate_filters_aliases() -> None:
    """Downsampling must remove frequencies above the output Nyquist frequency."""
    source_sample_rate = 24000
    output_sample_rate = 8000
    seconds = np.arange(source_sample_rate, dtype=np.float32) / source_sample_rate
    audio = 0.2 * np.sin(2 * np.pi * 1000 * seconds)
    audio += 0.2 * np.sin(2 * np.pi * 10000 * seconds)
    encoded, _ = encode_audio(
        audio,
        response_format="pcm",
        sample_rate=source_sample_rate,
        output_sample_rate=output_sample_rate,
    )
    samples = np.frombuffer(encoded, dtype="<i2").astype(np.float32) / 32768
    assert samples.size == output_sample_rate
    assert dominant_frequency(samples, output_sample_rate) == pytest.approx(1000, abs=3)
    interior = samples[output_sample_rate // 10 : -output_sample_rate // 10]
    assert np.sqrt(np.mean(interior**2)) == pytest.approx(0.2 / np.sqrt(2), abs=0.003)


def test_output_sample_rate_preserves_stereo() -> None:
    """Resampling must preserve channel orientation and each channel's pitch."""
    audio, source_sample_rate = stereo_test_signal()
    output_sample_rate = 16000
    encoded, _ = encode_audio(
        audio,
        response_format="wav",
        sample_rate=source_sample_rate,
        output_sample_rate=output_sample_rate,
    )
    actual_sample_rate, decoded = decode_audio(encoded)
    assert actual_sample_rate == output_sample_rate
    assert decoded.shape == (2, output_sample_rate // 2)
    assert_stereo_content(decoded, actual_sample_rate)


@pytest.mark.parametrize(
    ("response_format", "output_sample_rate"),
    [("wav", 12345), ("flac", 12345), ("mp3", 44100), ("aac", 22050), ("opus", 48000)],
)
def test_encode_audio_honors_custom_output_sample_rate(
    response_format: str, output_sample_rate: int
) -> None:
    """Encoded audio must actually decode at the requested rate, including arbitrary WAV/FLAC rates."""
    source_sample_rate = 24000
    seconds = np.arange(source_sample_rate, dtype=np.float32) / source_sample_rate
    audio = 0.25 * np.sin(2 * np.pi * 1000 * seconds)
    encoded, mime_type = encode_audio(
        audio,
        response_format=response_format,
        sample_rate=source_sample_rate,
        output_sample_rate=output_sample_rate,
        allow_format_fallback=False,
    )
    decoded_sample_rate, decoded = decode_audio(encoded)
    assert mime_type == FORMAT_MIME_TYPES[response_format]
    assert decoded_sample_rate == output_sample_rate
    assert dominant_frequency(
        decoded.reshape(-1), decoded_sample_rate
    ) == pytest.approx(1000, abs=3)


@pytest.mark.parametrize(
    ("response_format", "output_sample_rate"),
    [("wav", 0), ("wav", True), ("mp3", 12345), ("opus", 24000)],
)
def test_encode_audio_rejects_invalid_output_sample_rate(
    response_format: str, output_sample_rate: int | bool
) -> None:
    """Direct Python encoding must reject invalid rates without relying on HTTP validation."""
    with pytest.raises(ValueError, match="sample_rate"):
        encode_audio(
            np.zeros(2400, dtype=np.float32),
            response_format=response_format,
            output_sample_rate=output_sample_rate,
        )


@pytest.mark.parametrize(
    (
        "response_format",
        "source_sample_rate",
        "output_sample_rate",
        "expected_sample_rate",
    ),
    [
        ("mp3", 96000, None, 48000),
        ("aac", 20000, None, 22050),
        ("opus", 24000, None, 48000),
    ],
)
def test_speech_sample_rate_metadata_matches_encoded_audio(
    response_format: str,
    source_sample_rate: int,
    output_sample_rate: int | None,
    expected_sample_rate: int,
) -> None:
    """Speech metadata must describe the codec output instead of an incompatible model rate."""
    speech_client = FixedSpeechClient(
        np.zeros(source_sample_rate // 2, dtype=np.float32), source_sample_rate
    )
    result = asyncio.run(
        speech_client.speech(
            GenerateRequest(prompt="Hello."),
            request_id="encoded-sample-rate",
            response_format=response_format,
            output_sample_rate=output_sample_rate,
            allow_format_fallback=False,
        )
    )
    decoded_sample_rate, _ = decode_audio(result.audio_bytes)
    assert result.sample_rate == decoded_sample_rate == expected_sample_rate
