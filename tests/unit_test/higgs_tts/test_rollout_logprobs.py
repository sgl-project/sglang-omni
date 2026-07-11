# SPDX-License-Identifier: Apache-2.0
"""Numerical tests for :func:`selected_token_logprobs` (RL rollout logprobs).

Contract: a sampled row's logprob is the FULL-vocab ``log_softmax(logits/T)`` at
the sampled code (top-k/top-p are sampling filters, they must NOT truncate it);
a greedy row (``T <= 1e-5`` or ``top_k == 1``) uses ``log_softmax`` of the RAW
logits. CPU-only (log_softmax + gather, no fused kernels).
"""

from __future__ import annotations

import torch

from sglang_omni.models.higgs_tts.sampler import (
    _GREEDY_TEMP_THRESHOLD,
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


def test_topk_topp_do_not_truncate_logprob():
    """top-k>1 changes WHICH token is drawn, never the returned full-vocab logprob."""
    torch.manual_seed(3)
    B = 2
    logits = torch.randn(B, N, V, device=DEVICE)
    codes = torch.randint(0, V, (B, N), device=DEVICE)
    temperature = torch.full((B,), 1.0, device=DEVICE)

    base = selected_token_logprobs(logits, codes, temperature=temperature)
    with_topk = selected_token_logprobs(
        logits,
        codes,
        temperature=temperature,
        top_k_buf=torch.full((B,), 5, dtype=torch.long, device=DEVICE),
    )
    assert torch.allclose(base, with_topk, atol=0.0)


def test_greedy_convention_uses_raw_logits():
    """Greedy rows (T~0 OR top_k==1) use log_softmax of the RAW logits."""
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

    for b in range(B):
        raw = torch.log_softmax(logits[b].float(), dim=-1)
        exp_b = raw.gather(-1, codes[b].long().unsqueeze(-1)).squeeze(-1)
        assert torch.allclose(got[b], exp_b, atol=1e-5), f"row {b}"


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
            raw = torch.log_softmax(logits[b].float(), dim=-1)
            exp_b = raw.gather(-1, codes[b].long().unsqueeze(-1)).squeeze(-1)
        else:
            exp_b = _manual(logits[b], codes[b], temp_val)
        assert torch.allclose(got[b], exp_b, atol=1e-5), f"row {b}"
