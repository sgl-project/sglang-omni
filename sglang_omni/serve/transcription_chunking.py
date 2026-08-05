# SPDX-License-Identifier: Apache-2.0
"""Long-audio chunking for ``/v1/audio/transcriptions``.

Chunking is opt-in per model, declared via
:class:`~sglang_omni.config.AudioChunkingConfig`. Audio longer than
``max_audio_clip_s`` is split into non-overlapping chunks at the quietest point near each nominal boundary.
"""

from __future__ import annotations

import logging

import numpy as np

from sglang_omni.config import AudioChunkingConfig

logger = logging.getLogger(__name__)

#: Half-open ``[start, end)`` sample range of one chunk.
Span = tuple[int, int]


def needs_chunking(duration_s: float, config: AudioChunkingConfig) -> bool:
    """Whether a probed clip duration calls for splitting.

    ``duration_s <= 0`` means the duration probe failed (``soundfile`` could
    not read the container). Such uploads keep today's behaviour rather than
    paying a full decode just to find out how long they are.
    """
    if not config.allow_audio_chunking:
        return False
    if duration_s <= 0:
        logger.debug("[transcription] audio duration unknown; skipping chunking")
        return False
    return duration_s > config.max_audio_clip_s


def check_total_duration(duration_s: float, config: AudioChunkingConfig) -> None:
    """Reject uploads past ``max_total_audio_s``.

    Raises ``ValueError``; the handler maps it to HTTP 400. The wording matches
    the engine-side limit message in ``preprocessing.transcription`` so both
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
    """Split a waveform by cutting at the quietest window before each boundary.
    """

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
        self, waveform: np.ndarray, sample_rate: int, max_chunk_samples: int
    ) -> list[Span]:
        """Split ``waveform`` into the spans that become chunk requests.
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
            clamped = min(max(int(cut), search_start), boundary)
            if clamped != cut:
                # An out-of-range split point would either loop forever or
                # emit oversized chunks; repair it, but loudly.
                logger.warning(
                    "[transcription] %s returned split point %s outside "
                    "[%s, %s]; clamped to %s",
                    type(self).__name__,
                    cut,
                    search_start,
                    boundary,
                    clamped,
                )
            spans.append((start, clamped))
            start = clamped
        return spans

    def _find_split_point(
        self, waveform: np.ndarray, search_start: int, search_end: int
    ) -> int:
        """Return the sample index to cut at, within ``[search_start, search_end]``."""
        region = waveform[search_start:search_end]
        window = self.energy_window_samples
        window_count = int(region.shape[-1]) // window
        if window_count < 1:
            # The region cannot hold one window; fall back to the boundary.
            return search_end
        frames = region[: window_count * window].reshape(window_count, window)
        # Mean square orders identically to RMS, so the square root is skipped.
        energies = np.mean(np.square(frames.astype(np.float64)), axis=1)
        quietest = int(np.argmin(energies))
        return search_start + quietest * window + window // 2


__all__ = [
    "RMSSplitter",
    "Span",
    "check_total_duration",
    "needs_chunking",
]
