"""Rolling PCM16 audio buffer for streaming WebSocket sessions.
"""

from __future__ import annotations

import base64
import io
import wave

# 60 seconds hard cap for audio buffer.
DEFAULT_MAX_BUFFER_BYTES = 60 * 16000 * 2


class BufferOverflow(ValueError):
    """Raised when a realtime input audio buffer exceeds its byte limit."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        super().__init__(f"Realtime audio buffer exceeded max_bytes={max_bytes}")


class RealtimeAudioBuffer:
    """Append-only buffer of raw little-endian PCM16 bytes."""

    def __init__(
        self,
        *,
        source_sr: int = 16000,
        target_sr: int = 16000,
        channels: int = 1,
        max_bytes: int = DEFAULT_MAX_BUFFER_BYTES,
    ) -> None:
        self.source_sr = source_sr
        self.target_sr = target_sr
        self.channels = channels
        self.max_bytes = max_bytes
        self.buf = bytearray()

    def append_b64(self, audio_b64: str) -> int:
        chunk = base64.b64decode(audio_b64, validate=False)
        if len(self.buf) + len(chunk) > self.max_bytes:
            raise BufferOverflow(self.max_bytes)
        self.buf.extend(chunk)
        return len(chunk)

    def clear(self) -> None:
        self.buf.clear()

    @property
    def num_bytes(self) -> int:
        return len(self.buf)

    @property
    def num_samples(self) -> int:
        return len(self.buf) // (2 * self.channels)

    def is_empty(self) -> bool:
        return self.num_samples == 0

    def to_full_wav_data_uri(self) -> str:
        return self.to_sliced_wav_data_uri(start_byte=0, end_byte=len(self.buf))

    def to_sliced_wav_data_uri(self, *, start_byte: int, end_byte: int) -> str:
        chunk = bytes(self.buf[start_byte:end_byte])
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.source_sr)
            wf.writeframes(chunk)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:audio/wav;base64,{b64}"

    def tail(self, num_bytes: int) -> bytes:
        assert len(self.buf) >= num_bytes, "Not enough bytes in buffer"
        return bytes(self.buf[-num_bytes:])
