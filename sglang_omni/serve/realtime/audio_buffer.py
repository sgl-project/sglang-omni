"""Rolling PCM16 audio buffer for streaming WebSocket sessions.
"""

from __future__ import annotations

import base64
import io
import wave

PCM_SAMPLE_RATE = 16000
PCM16_BYTES_PER_SAMPLE = 2

# 60 seconds hard cap for audio buffer.
DEFAULT_MAX_BUFFER_BYTES = 60 * PCM_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE


class BufferOverflow(ValueError):
    """Raised when a realtime input audio buffer exceeds its byte limit."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        super().__init__(f"Realtime audio buffer exceeded max_bytes={max_bytes}")


class RealtimeAudioBuffer:
    """Append-only buffer of raw little-endian PCM16 bytes.

    Audio is addressed in session samples, counted from the first byte ever
    appended. Dropping a prefix advances start_sample, so positions a caller
    recorded earlier stay valid after the audio before them is gone.
    """

    def __init__(
        self,
        *,
        source_sr: int = PCM_SAMPLE_RATE,
        target_sr: int = PCM_SAMPLE_RATE,
        channels: int = 1,
        max_bytes: int = DEFAULT_MAX_BUFFER_BYTES,
    ) -> None:
        self.source_sr = source_sr
        self.target_sr = target_sr
        self.channels = channels
        self.max_bytes = max_bytes
        self.buf = bytearray()
        self.start_sample = 0
        # Across all channels, so it is also the byte size of one position.
        self.bytes_per_sample = PCM16_BYTES_PER_SAMPLE * channels

    def append_b64(self, audio_b64: str) -> int:
        chunk = base64.b64decode(audio_b64, validate=False)
        return self.append_bytes(chunk)

    def append_bytes(self, chunk: bytes) -> int:
        if len(self.buf) + len(chunk) > self.max_bytes:
            raise BufferOverflow(self.max_bytes)
        else:
            pass
        self.buf.extend(chunk)
        return len(chunk)

    def clear(self) -> None:
        self.drop_prefix(len(self.buf))

    def drop_prefix(self, num_bytes: int) -> None:
        num_bytes = min(max(num_bytes, 0), len(self.buf))
        del self.buf[:num_bytes]
        self.start_sample += num_bytes // self.bytes_per_sample

    @property
    def end_sample(self) -> int:
        return self.start_sample + self.num_samples

    def byte_offset(self, sample: int) -> int:
        """Byte offset of a session sample, clamped to the held audio."""
        sample = min(max(sample, self.start_sample), self.end_sample)
        return (sample - self.start_sample) * self.bytes_per_sample

    def slice(self, start_sample: int, end_sample: int | None = None) -> bytes:
        """PCM between two session samples; None runs to the end."""
        end_byte = len(self.buf) if end_sample is None else self.byte_offset(end_sample)
        return bytes(self.buf[self.byte_offset(start_sample) : end_byte])

    def drop_before(self, sample: int) -> None:
        self.drop_prefix(self.byte_offset(sample))

    @property
    def num_bytes(self) -> int:
        return len(self.buf)

    @property
    def num_samples(self) -> int:
        return len(self.buf) // self.bytes_per_sample

    def is_empty(self) -> bool:
        return self.num_samples == 0

    def to_full_wav_data_uri(self) -> str:
        return self.to_sliced_wav_data_uri(start_byte=0, end_byte=len(self.buf))

    def to_sliced_wav_data_uri(self, *, start_byte: int, end_byte: int) -> str:
        wav_bytes = self.to_sliced_wav_bytes(start_byte=start_byte, end_byte=end_byte)
        b64 = base64.b64encode(wav_bytes).decode("ascii")
        return f"data:audio/wav;base64,{b64}"

    def to_sliced_wav_bytes(self, *, start_byte: int, end_byte: int) -> bytes:
        return self.pcm_to_wav_bytes(bytes(self.buf[start_byte:end_byte]))

    def pcm_to_wav_bytes(self, pcm: bytes) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(PCM16_BYTES_PER_SAMPLE)
            wf.setframerate(self.source_sr)
            wf.writeframes(pcm)
        return buf.getvalue()

    def tail(self, num_bytes: int) -> bytes:
        assert 0 <= num_bytes <= len(self.buf), "Invalid tail length"
        if num_bytes == 0:
            return b""
        else:
            pass
        return bytes(self.buf[-num_bytes:])
