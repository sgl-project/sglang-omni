# SPDX-License-Identifier: Apache-2.0
"""Long-audio chunking for ``/v1/audio/transcriptions``.

Chunking is opt-in per model, declared via
:class:`~sglang_omni.config.AudioChunkingConfig`. Audio longer than
``max_audio_clip_s`` is split into non-overlapping chunks at the quietest
point near each nominal boundary; ...
"""

from __future__ import annotations

import logging

from sglang_omni.config import AudioChunkingConfig

logger = logging.getLogger(__name__)


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


__all__ = [
    "check_total_duration",
    "needs_chunking",
]
