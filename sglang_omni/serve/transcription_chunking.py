# SPDX-License-Identifier: Apache-2.0
"""Model-policy-driven chunking for long transcription uploads."""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np

from sglang_omni.config import AudioChunkingConfig
from sglang_omni.utils.audio import load_audio

TARGET_SAMPLE_RATE = 16000
_BOUNDARY_SEARCH_SECONDS = 5.0
_ENERGY_WINDOW_SECONDS = 0.1
_MIN_CHUNK_SECONDS = 0.5


@dataclass(frozen=True)
class ChunkSpan:
    """Half-open sample range for one transcription chunk."""

    index: int
    start_sample: int
    end_sample: int
    sample_rate: int

    @property
    def start_s(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end_s(self) -> float:
        return self.end_sample / self.sample_rate


@dataclass
class ChunkPlan:
    """Decoded waveform and the ordered spans submitted to the model."""

    sample_rate: int
    duration_s: float
    spans: list[ChunkSpan]
    waveform: np.ndarray = field(repr=False)

    def encode(self, span: ChunkSpan) -> bytes:
        """Encode one lossless float WAV chunk for an engine request."""
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(
            buffer,
            np.ascontiguousarray(
                self.waveform[span.start_sample : span.end_sample],
                dtype=np.float32,
            ),
            self.sample_rate,
            format="WAV",
            subtype="FLOAT",
        )
        return buffer.getvalue()


def _quiet_boundary(
    waveform: np.ndarray,
    *,
    start_sample: int,
    hard_end_sample: int,
    sample_rate: int,
) -> int:
    search_samples = int(_BOUNDARY_SEARCH_SECONDS * sample_rate)
    search_start = max(start_sample + 1, hard_end_sample - search_samples)
    region = np.abs(waveform[search_start:hard_end_sample])
    if region.size == 0:
        return hard_end_sample

    window_samples = min(
        max(int(_ENERGY_WINDOW_SECONDS * sample_rate), 1),
        int(region.size),
    )
    cumulative_energy = np.empty(region.size + 1, dtype=np.float64)
    cumulative_energy[0] = 0.0
    np.cumsum(region, dtype=np.float64, out=cumulative_energy[1:])
    window_energy = (
        cumulative_energy[window_samples:] - cumulative_energy[:-window_samples]
    )
    # Prefer the latest equally quiet window so all-silence input stays close
    # to the hard boundary instead of degenerating into tiny chunks.
    quietest_window = int(np.flatnonzero(window_energy == window_energy.min())[-1])
    boundary = search_start + quietest_window + window_samples // 2
    return min(max(boundary, start_sample + 1), hard_end_sample)


def plan_audio_chunks(
    audio_bytes: bytes,
    config: AudioChunkingConfig,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> ChunkPlan | None:
    """Decode and split audio that exceeds the configured model clip limit."""

    max_audio_clip_s = config.max_audio_clip_s
    if not config.allow_audio_chunking or max_audio_clip_s is None:
        return None

    waveform = load_audio(
        audio_bytes,
        source_name="transcription",
        target_sample_rate=sample_rate,
    )
    total_samples = int(waveform.size)
    max_chunk_samples = max(int(max_audio_clip_s * sample_rate), 1)
    min_chunk_samples = max(int(_MIN_CHUNK_SECONDS * sample_rate), 1)
    if total_samples <= max_chunk_samples:
        return None

    spans: list[ChunkSpan] = []
    start_sample = 0
    while total_samples - start_sample > max_chunk_samples:
        hard_end_sample = start_sample + max_chunk_samples
        end_sample = _quiet_boundary(
            waveform,
            start_sample=start_sample,
            hard_end_sample=hard_end_sample,
            sample_rate=sample_rate,
        )
        tail_samples = total_samples - end_sample
        if 0 < tail_samples < min_chunk_samples:
            end_sample = max(total_samples - min_chunk_samples, start_sample + 1)
        spans.append(
            ChunkSpan(
                index=len(spans),
                start_sample=start_sample,
                end_sample=end_sample,
                sample_rate=sample_rate,
            )
        )
        start_sample = end_sample
    spans.append(
        ChunkSpan(
            index=len(spans),
            start_sample=start_sample,
            end_sample=total_samples,
            sample_rate=sample_rate,
        )
    )
    return ChunkPlan(
        sample_rate=sample_rate,
        duration_s=total_samples / sample_rate,
        spans=spans,
        waveform=waveform,
    )


__all__ = ["ChunkPlan", "ChunkSpan", "plan_audio_chunks"]
