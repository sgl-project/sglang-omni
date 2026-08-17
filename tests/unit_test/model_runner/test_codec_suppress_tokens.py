# SPDX-License-Identifier: Apache-2.0
"""Shared sparse suppression: correctness of the per-row grouping and the fill."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.model_runner.base import ModelRunner

_VOCAB = 32


def _request(tokens):
    return SimpleNamespace(
        data=SimpleNamespace(suppress_tokens=tokens, req=SimpleNamespace())
    )


def _apply(runner, logits, requests):
    runner._apply_codec_suppress_tokens(
        SimpleNamespace(next_token_logits=logits), requests
    )


def test_uniform_suppress_set_fills_every_row() -> None:
    runner = object.__new__(ModelRunner)
    logits = torch.zeros(4, _VOCAB)
    tokens = (1, 5, 9)

    _apply(runner, logits, [_request(tokens) for _ in range(4)])

    assert torch.isneginf(logits[:, list(tokens)]).all()
    kept = [c for c in range(_VOCAB) if c not in tokens]
    assert torch.equal(logits[:, kept], torch.zeros(4, len(kept)))


def test_mixed_suppress_sets_only_touch_their_own_rows() -> None:
    """Two sets plus an unsuppressed row must not bleed into each other."""
    runner = object.__new__(ModelRunner)
    logits = torch.zeros(4, _VOCAB)
    set_a, set_b = (1, 2), (7,)
    requests = [
        _request(set_a),
        _request(set_b),
        _request(set_a),
        _request(None),
    ]

    _apply(runner, logits, requests)

    for row in (0, 2):
        assert torch.isneginf(logits[row, list(set_a)]).all()
        assert not torch.isneginf(logits[row, list(set_b)]).any()
    assert torch.isneginf(logits[1, list(set_b)]).all()
    assert not torch.isneginf(logits[1, list(set_a)]).any()
    # the request that opted out keeps every logit
    assert torch.equal(logits[3], torch.zeros(_VOCAB))


def test_ids_outside_the_vocabulary_are_dropped() -> None:
    runner = object.__new__(ModelRunner)
    logits = torch.zeros(2, _VOCAB)

    _apply(runner, logits, [_request((3, _VOCAB, _VOCAB + 100, -1)) for _ in range(2)])

    assert torch.isneginf(logits[:, 3]).all()
    assert torch.isfinite(logits[:, [c for c in range(_VOCAB) if c != 3]]).all()


def test_one_device_tensor_is_cached_per_distinct_suppress_set() -> None:
    runner = object.__new__(ModelRunner)
    logits = torch.zeros(6, _VOCAB)
    requests = [_request((1, 2)) for _ in range(4)] + [_request((7,)) for _ in range(2)]

    _apply(runner, logits, requests)

    # Six requests, two distinct sets: the cache is keyed by content, not request.
    assert len(runner._suppress_tensor_cache) == 2


def test_repeated_calls_reuse_the_cache_and_stay_correct() -> None:
    runner = object.__new__(ModelRunner)
    requests = [_request((4, 8)) for _ in range(3)]

    for _ in range(3):
        logits = torch.zeros(3, _VOCAB)
        _apply(runner, logits, requests)
        assert torch.isneginf(logits[:, [4, 8]]).all()

    assert len(runner._suppress_tensor_cache) == 1
