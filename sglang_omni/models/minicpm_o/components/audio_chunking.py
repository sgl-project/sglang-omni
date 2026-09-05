# SPDX-License-Identifier: Apache-2.0
"""Whisper-chunk bookkeeping for MiniCPM-o audio input.

The checkpoint processor splits audio into 30 s mel windows and counts
placeholder tokens from the *whole* clip (``get_audio_placeholder``):

    frames  = ceil(samples / hop)
    cnn     = (frames - 1) // 2 + 1
    tokens  = (cnn - pool_step) // pool_step + 1

while the encoder emits ``tokens(frames_of_chunk)`` rows per window. A clip a
few samples past a 30 s boundary therefore gets an extra window that holds a
handful of frames, pads to a full 30 s in the mel tensor, and contributes zero
tokens — but still costs a full whisper forward (and a full STFT in
preprocessing). Daily-Omni's 30 s / 60 s clips all do this.

Note (ruoyu): both helpers below only remove work that yields no tokens; the
placeholder count and the encoder's row count are unchanged (checked
exhaustively in tests/unit_test/minicpm_o/test_audio_chunking.py).
"""

from __future__ import annotations

from typing import Any

import numpy as np

SAMPLE_RATE = 16000
HOP_LENGTH = 160
CHUNK_SECONDS = 30
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS


def pooled_token_count(num_frames: int, pool_step: int) -> int:
    """Encoder rows produced by ``num_frames`` mel frames (remote-code formula)."""
    if num_frames <= 0:
        return 0
    after_cnn = (num_frames - 1) // 2 + 1
    return max((after_cnn - pool_step) // pool_step + 1, 0)


def whole_clip_token_count(num_samples: int, pool_step: int) -> int:
    """Placeholder count the processor derives from the whole clip."""
    frames = -(-num_samples // HOP_LENGTH)  # ceil
    return pooled_token_count(frames, pool_step)


def trim_zero_token_tail(audio: np.ndarray, *, pool_step: int) -> np.ndarray:
    """Drop a trailing partial 30 s window that would produce no tokens.

    Only tails of at most ``(2 * pool_step - 2) * 2 - 1`` frames can be
    dropped without moving the whole-clip token count; for ``pool_step=5``
    that is 8 frames = 1280 samples.
    """
    n = int(audio.shape[-1])
    tail = n % CHUNK_SAMPLES
    if tail == 0 or n <= CHUNK_SAMPLES:
        return audio
    tail_frames = -(-tail // HOP_LENGTH)
    if pooled_token_count(tail_frames, pool_step) != 0:
        return audio
    trimmed = n - tail
    if whole_clip_token_count(trimmed, pool_step) != whole_clip_token_count(
        n, pool_step
    ):
        return audio
    return audio[..., :trimmed]


def drop_zero_token_chunks(
    audio_features: Any, audio_feature_lens: Any, *, pool_step: int
) -> tuple[Any, Any]:
    """Remove mel windows whose frame count yields zero encoder rows.

    ``audio_features`` is ``(num_chunks, n_mels, max_frames)`` and
    ``audio_feature_lens`` is ``(num_chunks,)``; both are returned filtered in
    order. Returns the inputs untouched when nothing can be dropped.
    """
    import torch

    lens = audio_feature_lens.reshape(-1)
    keep = torch.tensor(
        [pooled_token_count(int(length), pool_step) > 0 for length in lens.tolist()],
        dtype=torch.bool,
    )
    if bool(keep.all()):
        return audio_features, audio_feature_lens
    return audio_features[keep], lens[keep]
