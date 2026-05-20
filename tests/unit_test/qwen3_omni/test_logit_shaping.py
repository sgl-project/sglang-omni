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
