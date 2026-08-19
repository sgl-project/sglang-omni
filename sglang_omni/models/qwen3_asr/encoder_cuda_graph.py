# SPDX-License-Identifier: Apache-2.0
"""Bucketed CUDA graphs for the Qwen3-ASR audio encoder layer stack.

The chunk/conv front end reads each clip's length, so its shapes and control
flow change from request to request — we leave it on the eager path. What
we capture is the 24-layer transformer stack and the output projection that
follow. By then the batch is packed as [total_tokens, hidden], so capture
buckets only need to track total token count.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

def build_buckets(max_batch: int, max_tokens_per_clip: int) -> tuple[int, ...]:
    """Return token-count bucket sizes for encoder CUDA-graph capture.

    Each captured graph pads the packed [total_tokens, hidden] encoder
    input up to one of these sizes. The caller supplies two deployment
    limits:
    1. max_batch: pre_lm_max_batch_size on the per-LM encoder service
      (maximum clips in one encode batch).
    2. max_tokens_per_clip: encoder output tokens for the longest clip the
      pipeline admits; derived from AudioChunkingConfig.max_audio_clip_s
      via qwen3_asr_num_audio_tokens.

    Buckets are power-of-two sizes from 128 up to max_batch * max_tokens_per_clip.
    """
    ceiling = max(int(max_batch) * int(max_tokens_per_clip), 1)
    buckets: list[int] = []
    step = 128
    while step < ceiling:
        buckets.append(step)
        step *= 2
    buckets.append(ceiling)
    return tuple(buckets)


def pick_bucket(total_tokens: int, buckets: tuple[int, ...]) -> int | None:
    """Smallest bucket that fits, or None (caller falls back to eager)."""
    for bucket in buckets:
        if total_tokens <= bucket:
            return bucket
    return None
