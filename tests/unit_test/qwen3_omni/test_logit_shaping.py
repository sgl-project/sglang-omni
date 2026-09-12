"""Regression tests for ModelRunner logit-shaping helpers."""

from __future__ import annotations

import types

import pytest
import torch

from sglang_omni.model_runner.base import ModelRunner


def _make_suppress_requests(suppress_per_row):
    reqs = []
    for suppress in suppress_per_row:
        req = types.SimpleNamespace()
        data = types.SimpleNamespace(req=req, suppress_tokens=suppress)
        reqs.append(types.SimpleNamespace(data=data))
    return reqs


def _suppress_reference(logits, requests):
    out = logits.clone()
    vocab = out.shape[1]
    for row_idx, sched_req in enumerate(requests):
        suppress = sched_req.data.suppress_tokens
        if not suppress:
            continue
        for tok in suppress:
            tok = int(tok)
            if 0 <= tok < vocab:
                out[row_idx, tok] = float("-inf")
    return out


@pytest.mark.parametrize("share_rows", [True, False])
def test_codec_suppress_tokens_matches_reference(share_rows):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = 96
    batch = 4
    torch.manual_seed(1)
    shared = [5, 90, 95, 200, -3]
    if share_rows:
        suppress_per_row = [shared] * batch
    else:
        suppress_per_row = [shared, [1, 2], None, shared]
    requests = _make_suppress_requests(suppress_per_row)
    runner = types.SimpleNamespace()

    for _ in range(3):  # repeated calls exercise the tensor cache
        logits_orig = torch.randn(batch, vocab, dtype=torch.float32, device=device)
        logits_output = types.SimpleNamespace(next_token_logits=logits_orig.clone())
        ModelRunner._apply_codec_suppress_tokens(runner, logits_output, requests)
        expected = _suppress_reference(logits_orig, requests)
        assert torch.equal(logits_output.next_token_logits, expected)


def test_suppress_cache_holds_one_entry_across_requests():
    """Fresh list objects with identical content must share one device tensor."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = 64
    runner = types.SimpleNamespace()
    shared = [5, 20, 90]

    for step in range(5):
        # the request builder hands out a new list object per request
        requests = _make_suppress_requests([list(shared), list(shared)])
        logits = torch.randn(2, vocab, device=device)
        logits_output = types.SimpleNamespace(next_token_logits=logits.clone())
        ModelRunner._apply_codec_suppress_tokens(runner, logits_output, requests)
        assert torch.equal(
            logits_output.next_token_logits, _suppress_reference(logits, requests)
        ), step

    assert len(runner._suppress_tensor_cache) == 1
