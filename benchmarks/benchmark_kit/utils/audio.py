import struct

WAV_HEADER_SIZE = 44


def get_wav_duration(wav_bytes: bytes) -> float:
    """Return duration in seconds from a WAV byte buffer."""
    if len(wav_bytes) <= WAV_HEADER_SIZE:
        return 0.0
    sample_rate = struct.unpack_from("<I", wav_bytes, 24)[0]
    num_channels = struct.unpack_from("<H", wav_bytes, 22)[0]
    bits_per_sample = struct.unpack_from("<H", wav_bytes, 34)[0]
    if sample_rate == 0 or num_channels == 0 or bits_per_sample == 0:
        return 0.0
    bytes_per_sample = num_channels * bits_per_sample // 8
    pcm_size = len(wav_bytes) - WAV_HEADER_SIZE
    return pcm_size / (sample_rate * bytes_per_sample)
