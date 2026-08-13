"""Regression tests for ModelRunner logit-shaping helpers."""

from __future__ import annotations

import types

import pytest
import torch

from sglang_omni.model_runner.base import ModelRunner


def _make_requests(output_ids_per_row, penalty: float):
    reqs = []
    for ids in output_ids_per_row:
        sp = types.SimpleNamespace(repetition_penalty=penalty)
        req = types.SimpleNamespace(sampling_params=sp, output_ids=ids)
        data = types.SimpleNamespace(req=req)
        reqs.append(types.SimpleNamespace(data=data))
    return reqs


def _scalar_reference(logits, requests, penalty):
    out = logits.clone()
    for row_idx, sched_req in enumerate(requests):
        ids = sched_req.data.req.output_ids
        if not ids:
            continue
        unique = list({int(t) for t in ids if 0 <= int(t) < out.shape[1]})
        if not unique:
            continue
        idx = torch.tensor(unique, dtype=torch.long, device=out.device)
        scores = out[row_idx, idx]
        scores = torch.where(scores > 0, scores / penalty, scores * penalty)
        out[row_idx, idx] = scores
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_apply_repetition_penalty_matches_scalar_reference(dtype):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = 256
    batch = 8
    penalty = 1.2
    torch.manual_seed(42)
    logits_orig = (
        torch.randn(batch, vocab, dtype=dtype, device=device) * 2.0
    ).contiguous()

    rng = torch.Generator(device="cpu").manual_seed(7)
    output_ids = [
        torch.randperm(vocab, generator=rng)[:32].tolist() for _ in range(batch)
    ]

    requests = _make_requests(output_ids, penalty)
    logits_output = types.SimpleNamespace(next_token_logits=logits_orig.clone())

    ModelRunner._apply_repetition_penalty(
        types.SimpleNamespace(), logits_output, requests
    )
    actual = logits_output.next_token_logits
    expected = _scalar_reference(logits_orig, requests, penalty)

    tol = {torch.float16: 2.5e-3, torch.bfloat16: 1e-2, torch.float32: 1e-6}[dtype]
    diff = (actual - expected).abs().max().item()
    assert diff <= tol, f"max abs diff {diff:.6f} > tol {tol:.6f} for {dtype}"


def test_repetition_penalty_incremental_matches_full_rebuild():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = 128
    penalty = 1.3
    torch.manual_seed(0)
    output_ids = [3, 7, 7, 100]
    requests = _make_requests([output_ids], penalty)

    for step_ids in ([3, 7, 7, 100, 55], [3, 7, 7, 100, 55, 3], [9], [9, 200, -1, 12]):
        requests[0].data.req.output_ids = list(step_ids)
        logits_orig = torch.randn(1, vocab, dtype=torch.float32, device=device)
        logits_output = types.SimpleNamespace(next_token_logits=logits_orig.clone())
        ModelRunner._apply_repetition_penalty(
            types.SimpleNamespace(), logits_output, requests
        )
        expected = _scalar_reference(logits_orig, requests, penalty)
        assert torch.equal(logits_output.next_token_logits, expected), step_ids


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


def test_repetition_penalty_resets_after_empty_retract():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = 64
    penalty = 1.5
    requests = _make_requests([[3, 9]], penalty)
    runner = types.SimpleNamespace()

    logits = torch.zeros(1, vocab, device=device) + 1.0
    ModelRunner._apply_repetition_penalty(
        runner, types.SimpleNamespace(next_token_logits=logits), requests
    )

    # Retract replaces output_ids with an empty list, then the restart
    # generates a different same-length prefix.
    requests[0].data.req.output_ids = []
    ModelRunner._apply_repetition_penalty(
        runner, types.SimpleNamespace(next_token_logits=logits.clone()), requests
    )
    requests[0].data.req.output_ids = [11, 12]
    logits_orig = torch.ones(1, vocab, device=device)
    logits_output = types.SimpleNamespace(next_token_logits=logits_orig.clone())
    ModelRunner._apply_repetition_penalty(runner, logits_output, requests)

    expected = _scalar_reference(logits_orig, requests, penalty)
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
