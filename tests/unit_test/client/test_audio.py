import numpy as np

from sglang_omni.client.audio import (
    FORMAT_MIME_TYPES,
    audio_to_base64,
    encode_audio,
    encode_pcm,
    encode_wav,
    to_numpy,
)


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
