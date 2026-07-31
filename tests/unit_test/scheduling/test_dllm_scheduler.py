# SPDX-License-Identifier: Apache-2.0
"""Tests for DLLM scheduler concurrent-prefill token-id normalization.

These tests exercise the pure ``split_token_ids_for_batch`` helper that
guards ``_apply_results`` against the flat-list mis-pairing bug that the
original ``prefill_max_requests=1`` hardcoding was protecting against. No
``sglang`` / CUDA is required (the helper is module-level and side-effect
free).
"""

from __future__ import annotations

import warnings

import pytest

from sglang_omni.scheduling.dllm_token_utils import split_token_ids_for_batch


def _fake_reqs(n: int):
    return [type("Req", (), {"rid": f"r{i}"})() for i in range(n)]


def test_single_request_flat_list_wrapped() -> None:
    reqs = _fake_reqs(1)
    out = split_token_ids_for_batch(reqs, [1, 2, 3])
    assert out == [[1, 2, 3]]


def test_single_request_nested_list_passthrough() -> None:
    reqs = _fake_reqs(1)
    out = split_token_ids_for_batch(reqs, [[7, 8]])
    assert out == [[7, 8]]


def test_multi_request_nested_shape_preserved() -> None:
    reqs = _fake_reqs(2)
    out = split_token_ids_for_batch(reqs, [[1, 2], [3, 4]])
    assert out == [[1, 2], [3, 4]]


def test_multi_request_flat_shape_degrades_round_robin_with_warning(caplog) -> None:
    import logging

    reqs = _fake_reqs(2)
    with caplog.at_level(logging.WARNING, logger="sglang_omni.scheduling.dllm_token_utils"):
        out = split_token_ids_for_batch(reqs, [1, 2, 3, 4])
    # No silent single-int pairing: every request gets its round-robin share.
    assert out == [[1, 3], [2, 4]]
    assert any("flat token-id list" in rec.message for rec in caplog.records)


def test_multi_request_flat_shape_empty_is_safe() -> None:
    reqs = _fake_reqs(3)
    out = split_token_ids_for_batch(reqs, [])
    assert out == [[], [], []]


def test_default_concurrency_is_one(monkeypatch) -> None:
    # Importing the class triggers sglang import, so we only assert the
    # documented default via the bootstrap signature default instead of
    # constructing the scheduler here.
    from sglang_omni.models.llada2_uni import bootstrap

    import inspect

    sig = inspect.signature(bootstrap.create_dllm_thinker_scheduler)
    assert sig.parameters["max_concurrent_prefill"].default == 1
