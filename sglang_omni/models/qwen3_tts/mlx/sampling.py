# SPDX-License-Identifier: Apache-2.0
"""Codec-token sampling for the MLX Qwen3-TTS talker.

Qwen3-TTS samples twice per frame with independent settings: the talker picks
codec group 0 (the "semantic" token, with a repetition penalty over recent
history), and the code predictor picks groups 1..N-1 (the "subtalker" tokens,
with no history). Both live here so the model code stays sampling-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mlx.core as mx


@dataclass
class SamplingParams:
    """Sampling settings for one of the two stages."""

    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    repetition_context_size: int = 64

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


def apply_top_k(logits: mx.array, top_k: int) -> mx.array:
    """Mask everything outside the ``top_k`` highest logits."""
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    kth = mx.sort(logits, axis=-1)[..., -top_k, None]
    return mx.where(logits < kth, mx.array(-mx.inf, logits.dtype), logits)


def apply_top_p(logits: mx.array, top_p: float) -> mx.array:
    """Keep the smallest prefix of probability mass reaching ``top_p``."""
    if top_p <= 0.0 or top_p >= 1.0:
        return logits
    probs = mx.softmax(logits.astype(mx.float32), axis=-1)
    order = mx.argsort(probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, order, axis=-1)
    cumulative = mx.cumsum(sorted_probs, axis=-1)
    # Drop tokens whose entire remaining tail is outside the nucleus. The
    # comparison is on the cumulative sum *including* the token, so the token
    # that crosses the threshold is kept.
    keep_sorted = cumulative > (1.0 - top_p)
    keep = mx.put_along_axis(mx.zeros_like(keep_sorted), order, keep_sorted, axis=-1)
    return mx.where(keep, logits, mx.array(-mx.inf, logits.dtype))


def suppress(logits: mx.array, token_ids: Sequence[int]) -> mx.array:
    """Force the given vocabulary entries to zero probability."""
    if not token_ids:
        return logits
    index = mx.array(list(token_ids), dtype=mx.int32)
    index = mx.broadcast_to(index[None, :], (logits.shape[0], index.shape[0]))
    return mx.put_along_axis(logits, index, mx.array(-mx.inf, logits.dtype), axis=-1)


def apply_repetition_penalty(
    logits: mx.array,
    recent_tokens: Sequence[int],
    penalty: float,
    context_size: int,
) -> mx.array:
    """Divide positive / multiply negative logits of recently emitted tokens."""
    if penalty == 1.0 or not recent_tokens:
        return logits
    window = {
        int(token)
        for token in recent_tokens[-context_size:]
        if 0 <= int(token) < logits.shape[-1]
    }
    if not window:
        return logits
    index = mx.array(sorted(window), dtype=mx.int32)
    index = mx.broadcast_to(index[None, :], (logits.shape[0], index.shape[0]))
    selected = mx.take_along_axis(logits, index, axis=-1)
    penalized = mx.where(selected < 0, selected * penalty, selected / penalty)
    return mx.put_along_axis(logits, index, penalized, axis=-1)


def sample_codec_token(
    logits: mx.array,
    params: SamplingParams,
    *,
    recent_tokens: Sequence[int] | None = None,
    suppress_tokens: Sequence[int] | None = None,
) -> mx.array:
    """Sample one token per row from ``[batch, vocab]`` logits.

    Returns ``[batch, 1]``. Stays lazy: no ``mx.eval`` and no host sync, so a
    caller can chain a whole frame before evaluating.
    """
    if suppress_tokens:
        logits = suppress(logits, suppress_tokens)
    if recent_tokens:
        logits = apply_repetition_penalty(
            logits,
            recent_tokens,
            params.repetition_penalty,
            params.repetition_context_size,
        )
    if params.greedy:
        return mx.argmax(logits, axis=-1, keepdims=True)

    if params.temperature != 1.0:
        logits = logits / params.temperature
    logits = apply_top_k(logits, params.top_k)
    logits = apply_top_p(logits, params.top_p)
    return mx.random.categorical(logits.astype(mx.float32), axis=-1)[:, None]


def special_codec_token_ids(vocab_size: int, keep: int) -> list[int]:
    """The trailing special-token block of the codec vocabulary, minus ``keep``.

    Qwen3-TTS reserves the last 1024 ids for control tokens; all but the EOS
    must be suppressed so the talker cannot emit one as audio.
    """
    return [
        token_id
        for token_id in range(max(0, vocab_size - 1024), vocab_size)
        if token_id != keep
    ]
