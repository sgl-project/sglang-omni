# SPDX-License-Identifier: Apache-2.0
"""Numerical tests for :func:`selected_token_logprobs` (RL rollout logprobs).

Contract: a sampled row's logprob comes from the exact temperature/top-k/top-p
distribution passed to multinomial. Deterministic rows use logprob ``0.0``.
"""

from __future__ import annotations

import torch

from sglang_omni.models.higgs_tts.sampler import (
    _GREEDY_TEMP_THRESHOLD,
    _sample_independent_batched_with_logprobs,
    selected_token_logprobs,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N = 8
V = 1026


def _manual(
    logits_NV: torch.Tensor, codes_N: torch.Tensor, temp: float
) -> torch.Tensor:
    lp = torch.log_softmax(logits_NV.float() / temp, dim=-1)
    return lp.gather(-1, codes_N.long().unsqueeze(-1)).squeeze(-1)


def _manual_filtered_probs(
    logits_BNV: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
) -> torch.Tensor:
    rows = []
    for batch_idx, logits_NV in enumerate(logits_BNV.float()):
        probs = (logits_NV / temperature[batch_idx]).softmax(dim=-1)
        k = min(int(top_k[batch_idx].item()), probs.shape[-1])
        keep_indices = probs.topk(k, dim=-1).indices
        keep_mask = torch.zeros_like(probs, dtype=torch.bool)
        keep_mask.scatter_(-1, keep_indices, True)
        probs = torch.where(keep_mask, probs, torch.zeros_like(probs))
        probs = probs / probs.sum(dim=-1, keepdim=True)

        sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
        remove = sorted_probs.cumsum(dim=-1) > top_p[batch_idx]
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        keep_sorted = ~remove
        keep_mask = torch.zeros_like(keep_sorted)
        keep_mask.scatter_(-1, sorted_indices, keep_sorted)
        probs = torch.where(keep_mask, probs, torch.zeros_like(probs))
        rows.append(probs / probs.sum(dim=-1, keepdim=True))
    return torch.stack(rows)


def test_matches_manual_log_softmax_per_row_temperature():
    """Sampled rows == full-vocab log_softmax(logits/T) at the code, per-row T."""
    torch.manual_seed(0)
    temps = [0.5, 0.7, 1.0, 2.0]
    B = len(temps)
    logits = torch.randn(B, N, V, device=DEVICE)
    codes = torch.randint(0, V, (B, N), device=DEVICE)
    temperature = torch.tensor(temps, device=DEVICE)

    got = selected_token_logprobs(logits, codes, temperature=temperature)

    expected = torch.stack([_manual(logits[b], codes[b], temps[b]) for b in range(B)])
    assert got.shape == (B, N)
    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-4)
    # Distinct temps give distinct logprobs on the same logits/codes.
    assert not torch.allclose(got[0], got[3])


def test_topk_logprob_uses_renormalized_distribution():
    torch.manual_seed(3)
    B = 2
    logits = torch.randn(B, N, V, device=DEVICE)
    codes = logits.topk(5, dim=-1).indices[..., 2]
    temperature = torch.full((B,), 1.0, device=DEVICE)

    with_topk = selected_token_logprobs(
        logits,
        codes,
        temperature=temperature,
        top_k_buf=torch.full((B,), 5, dtype=torch.long, device=DEVICE),
    )
    top_values, top_indices = logits.topk(5, dim=-1)
    top_probs = top_values.softmax(dim=-1)
    positions = (top_indices == codes.unsqueeze(-1)).to(torch.long).argmax(dim=-1)
    expected = top_probs.gather(-1, positions.unsqueeze(-1)).squeeze(-1).log()
    assert torch.allclose(with_topk, expected, atol=1e-5, rtol=1e-4)


def test_greedy_convention_is_log_one():
    torch.manual_seed(4)
    B = 2
    logits = torch.randn(B, N, V, device=DEVICE)
    codes = torch.randint(0, V, (B, N), device=DEVICE)
    # row 0: T=0 trigger; row 1: top_k==1 trigger at T=1.
    temperature = torch.tensor([0.0, 1.0], device=DEVICE)
    top_k_buf = torch.tensor([0, 1], dtype=torch.long, device=DEVICE)

    got = selected_token_logprobs(
        logits, codes, temperature=temperature, top_k_buf=top_k_buf
    )

    assert torch.equal(got, torch.zeros_like(got))


def test_mixed_greedy_and_sampled_rows():
    """A batch mixing greedy and sampled rows resolves each by its own rule."""
    torch.manual_seed(6)
    temps = [0.0, 1.5, 0.0, 0.8]
    B = len(temps)
    logits = torch.randn(B, N, V, device=DEVICE)
    codes = torch.randint(0, V, (B, N), device=DEVICE)
    temperature = torch.tensor(temps, device=DEVICE)

    got = selected_token_logprobs(logits, codes, temperature=temperature)

    for b, temp_val in enumerate(temps):
        if temp_val <= _GREEDY_TEMP_THRESHOLD:
            exp_b = torch.zeros(N, device=DEVICE)
        else:
            exp_b = _manual(logits[b], codes[b], temp_val)
        assert torch.allclose(got[b], exp_b, atol=1e-5), f"row {b}"


def test_sampler_returns_logprob_from_same_filtered_distribution():
    torch.manual_seed(9)
    B = 3
    logits = torch.randn(B, N, V, device=DEVICE)
    temperature = torch.tensor([0.7, 1.0, 1.3], device=DEVICE)
    top_p = torch.tensor([0.8, 0.9, 0.95], device=DEVICE)
    top_k = torch.tensor([20, 30, 40], dtype=torch.long, device=DEVICE)

    codes, sampled_logprobs = _sample_independent_batched_with_logprobs(
        logits,
        temperature=temperature,
        top_p=top_p,
        top_k_buf=top_k,
    )
    reference_probs = _manual_filtered_probs(logits, temperature, top_k, top_p)
    evaluated_logprobs = (
        reference_probs.gather(-1, codes.unsqueeze(-1)).squeeze(-1).log()
    )

    assert torch.allclose(sampled_logprobs, evaluated_logprobs, atol=1e-5)
