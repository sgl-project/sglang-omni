# SPDX-License-Identifier: Apache-2.0
"""Zero-token whisper windows are dropped without changing the token layout."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from sglang_omni.models.minicpm_o.components.audio_chunking import (
    CHUNK_SAMPLES,
    drop_zero_token_chunks,
    pooled_token_count,
    trim_zero_token_tail,
    whole_clip_token_count,
)

POOL_STEP = 5


def _processor_placeholder_count(audio_lens: int, pool_step: int = POOL_STEP) -> int:
    # verbatim from the checkpoint's MiniCPMOProcessor.get_audio_placeholder
    feature_lens = math.ceil(audio_lens / 160)
    feature_lens = (feature_lens - 1) // 2 + 1
    return (feature_lens - pool_step) // pool_step + 1


def _per_chunk_rows(num_samples: int, pool_step: int = POOL_STEP) -> int:
    # what the encoder emits: rows per 30 s window, summed
    rows = 0
    for start in range(0, num_samples, CHUNK_SAMPLES):
        chunk = min(CHUNK_SAMPLES, num_samples - start)
        rows += pooled_token_count(math.ceil(chunk / 160), pool_step)
    return rows


@pytest.mark.parametrize(
    "frames,expected", [(0, 0), (1, 0), (8, 0), (9, 1), (3000, 300)]
)
def test_pooled_token_count_matches_remote_formula(frames, expected):
    assert pooled_token_count(frames, POOL_STEP) == expected


def test_whole_clip_count_matches_processor():
    for n in range(1, 3 * CHUNK_SAMPLES, 997):
        assert whole_clip_token_count(n, POOL_STEP) == _processor_placeholder_count(n)


def test_trim_never_moves_the_placeholder_count():
    # exhaustive around both boundaries Daily-Omni sits on
    for base in (CHUNK_SAMPLES, 2 * CHUNK_SAMPLES):
        for tail in range(0, 3200):
            n = base + tail
            audio = np.zeros(n, dtype=np.float32)
            out = trim_zero_token_tail(audio, pool_step=POOL_STEP)
            assert whole_clip_token_count(
                out.shape[-1], POOL_STEP
            ) == _processor_placeholder_count(n)
            assert _per_chunk_rows(out.shape[-1]) == _per_chunk_rows(n)
            if 0 < tail <= 1280:
                assert out.shape[-1] == base, tail
            else:
                assert out.shape[-1] == n, tail


def test_trim_leaves_short_and_exact_clips_alone():
    for n in (1, 160, CHUNK_SAMPLES - 1, CHUNK_SAMPLES, 2 * CHUNK_SAMPLES):
        audio = np.zeros(n, dtype=np.float32)
        assert trim_zero_token_tail(audio, pool_step=POOL_STEP).shape[-1] == n


def test_drop_zero_token_chunks_keeps_order_and_rows():
    features = (
        torch.arange(3, dtype=torch.float32).view(3, 1, 1).expand(3, 4, 3000).clone()
    )
    lens = torch.tensor([3000, 1, 3000])
    kept_features, kept_lens = drop_zero_token_chunks(
        features, lens, pool_step=POOL_STEP
    )
    assert kept_lens.tolist() == [3000, 3000]
    assert kept_features[:, 0, 0].tolist() == [0.0, 2.0]
    # nine frames is the smallest window that still yields a token
    same_f, same_l = drop_zero_token_chunks(
        features, torch.tensor([3000, 9, 3000]), pool_step=POOL_STEP
    )
    assert same_l.tolist() == [3000, 9, 3000]
    assert same_f is features
