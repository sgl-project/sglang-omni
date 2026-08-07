# SPDX-License-Identifier: Apache-2.0
"""Long-audio chunking for /v1/audio/transcriptions.

Chunking is opt-in per model, declared via sglang_omni.config.AudioChunkingConfig.
Audio longer than max_audio_clip_s is split into non-overlapping chunks at the quietest point near each nominal boundary.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from sglang_omni.config import AudioChunkingConfig
from sglang_omni.utils.audio import load_audio

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000

# Half-open [start, end) sample range of one chunk.
Span = tuple[int, int]


def needs_chunking(duration_s: float, config: AudioChunkingConfig) -> bool:
    """Whether a probed clip duration calls for splitting.

    duration_s <= 0 means the duration probe failed (PyAV could not read
    the container). Such uploads keep today's behaviour rather than
    paying a full decode just to find out how long they are.
    """
    if not config.allow_audio_chunking:
        return False
    if duration_s <= 0:
        logger.debug("[transcription] audio duration unknown; skipping chunking")
        return False
    return duration_s > config.max_audio_clip_s


def check_total_duration(duration_s: float, config: AudioChunkingConfig) -> None:
    """Reject uploads past max_total_audio_s.

    Raises ValueError; the handler maps it to HTTP 400. The wording matches
    the engine-side limit message in preprocessing.transcription so both
    surface identically to callers.
    """
    limit = config.max_total_audio_s
    if limit is None or duration_s <= limit:
        return
    raise ValueError(
        f"transcription accepts audio up to {limit:g} seconds, "
        f"got {duration_s:.3f} seconds"
    )


class RMSSplitter:
    """Split a waveform by cutting at the quietest window before each boundary."""

    def __init__(
        self,
        *,
        search_window_s: float = 2.0,
        energy_window_samples: int = 1600,
    ) -> None:
        if search_window_s < 0:
            raise ValueError("search_window_s must not be negative")
        if energy_window_samples < 1:
            raise ValueError("energy_window_samples must be at least 1")
        self.search_window_s = float(search_window_s)
        self.energy_window_samples = int(energy_window_samples)

    def split(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        max_chunk_samples: int,
        *,
        min_tail_s: float = 0.0,
    ) -> list[Span]:
        """Split the waveform into the spans that become chunk requests.

        min_tail_s is the model's floor for a worth-transcribing final
        chunk (AudioChunkingConfig.min_tail_s); 0 disables the guard.
        """
        total_samples = int(waveform.shape[-1])
        if total_samples <= 0:
            return []
        chunk_samples = max(int(max_chunk_samples), 1)
        if total_samples <= chunk_samples:
            return [(0, total_samples)]

        search_samples = max(int(self.search_window_s * sample_rate), 0)
        spans: list[Span] = []
        start = 0
        while start < total_samples:
            if total_samples - start <= chunk_samples:
                spans.append((start, total_samples))
                break
            boundary = start + chunk_samples
            # Never search past the boundary (that would overrun the context limit)
            # and never back to `start`.
            search_start = max(boundary - search_samples, start + 1)
            cut = self._find_split_point(waveform, search_start, boundary)
            tail_samples = total_samples - cut
            min_tail = int(min_tail_s * sample_rate)
            if 0 < tail_samples < min_tail <= chunk_samples:
                # This cut would leave a sub-minimum final chunk. Pull the
                # cut earlier so the tail is worth transcribing; shrinking
                # the current chunk keeps every span within max_chunk_samples
                # (merging the tail forward instead would not).
                cut = max(total_samples - min_tail, start + 1)
            spans.append((start, cut))
            start = cut
        return spans

    def _find_split_point(
        self, waveform: np.ndarray, search_start: int, search_end: int
    ) -> int:
        """Return the sample index to cut at, within [search_start, search_end]."""
        region = waveform[search_start:search_end]
        window = self.energy_window_samples
        window_count = int(region.shape[-1]) // window
        if window_count < 1:
            # The region cannot hold one window; fall back to the boundary.
            return search_end
        frames = region[: window_count * window].reshape(window_count, window)
        # Mean square orders identically to RMS, so the square root is skipped.
        energies = np.mean(np.square(frames.astype(np.float64)), axis=1)
        # Prefer the latest equally quiet window so silence stays close to the
        # hard limit rather than introducing avoidable extra chunks.
        quietest = int(np.flatnonzero(energies == energies.min())[-1])
        return search_start + quietest * window + window // 2


@dataclass(frozen=True)
class ChunkSpan:
    """One chunk: where it sits in the upload, in samples and in seconds."""

    index: int
    start_sample: int
    end_sample: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class ChunkPlan:
    """A decoded upload plus the chunks it was split into.

    Holds the whole waveform for as long as the request runs, so chunk audio
    is encoded on demand rather than up front: materialising every chunk at
    once would double the resident memory of a long upload for no gain, since
    only the chunks currently in flight are ever needed.
    """

    sample_rate: int
    duration_s: float
    spans: list[ChunkSpan]
    waveform: np.ndarray = field(repr=False)

    def encode(self, span: ChunkSpan) -> bytes:
        """Encode one chunk as WAV bytes for a transcription request."""
        return encode_wav(
            self.waveform[span.start_sample : span.end_sample], self.sample_rate
        )


def encode_wav(waveform: np.ndarray, sample_rate: int) -> bytes:
    """Encode a mono float32 waveform as a 32-bit float WAV.

    Float rather than PCM_16 keeps the split lossless, and the engine-side
    load_audio has a dependency-free parser for float WAV, so chunks decode
    without going back through torchaudio.
    """
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(
        buffer,
        np.ascontiguousarray(waveform, dtype=np.float32),
        int(sample_rate),
        format="WAV",
        subtype="FLOAT",
    )
    return buffer.getvalue()


def plan_audio_chunks(
    audio_bytes: bytes,
    config: AudioChunkingConfig,
    *,
    splitter: RMSSplitter | None = None,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> ChunkPlan | None:
    """Decode an upload and split it, or return None to leave it untouched.

    Returning None means "send the original upload as one request".

    Blocking: decodes the whole file, so callers on the event loop must run it
    in a thread.
    """
    _, plan = decode_audio_chunks(
        audio_bytes,
        config,
        splitter=splitter,
        sample_rate=sample_rate,
    )
    return plan


def decode_audio_chunks(
    audio_bytes: bytes,
    config: AudioChunkingConfig,
    *,
    splitter: RMSSplitter | None = None,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> tuple[float, ChunkPlan | None]:
    """Decode an upload and return its duration plus any required chunk plan."""
    waveform = load_audio(
        audio_bytes, source_name="transcription", target_sample_rate=sample_rate
    )
    total_samples = int(waveform.shape[-1])
    duration_s = total_samples / sample_rate
    max_chunk_samples = config.chunk_samples(sample_rate)
    if total_samples <= max_chunk_samples:
        logger.debug(
            "[transcription] decoded audio is %d samples, within one chunk",
            total_samples,
        )
        return duration_s, None

    spans = (splitter or RMSSplitter()).split(
        waveform, sample_rate, max_chunk_samples, min_tail_s=config.min_tail_s
    )
    return duration_s, ChunkPlan(
        sample_rate=sample_rate,
        duration_s=duration_s,
        spans=[
            ChunkSpan(
                index=index,
                start_sample=start,
                end_sample=end,
                start_s=start / sample_rate,
                end_s=end / sample_rate,
            )
            for index, (start, end) in enumerate(spans)
        ],
        waveform=waveform,
    )


def join_transcript_parts(parts: Iterable[str]) -> str:
    """Join non-overlapping chunk transcripts exactly as the model returned them."""
    return "".join(parts)
