# SPDX-License-Identifier: Apache-2.0
"""Parity test for the vectorized talker repetition penalty.

Compares ``MiniCPMOTalkerModelRunner._apply_repetition_penalty`` against a
per-request reference of the checkpoint's
``CustomRepetitionPenaltyLogitsProcessorRepeat`` semantics: count each token
in the last ``REP_PENALTY_WINDOW`` generated codes, then scale the logit by
``penalty**count`` (multiply when negative, divide when positive). Pure CPU.
"""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import torch

from sglang_omni.models.minicpm_o.talker_model_runner import (
    REP_PENALTY_WINDOW,
    MiniCPMOTalkerModelRunner,
)


def _reference_penalty(
    logits: torch.Tensor, requests: list[SimpleNamespace]
) -> torch.Tensor:
    vocab = logits.shape[1]
    out = logits.clone().to(torch.float32)
    for row, sched_req in enumerate(requests):
        data = sched_req.data
        penalty = float(data.talker_model_inputs.get("rep_penalty", 1.0))
        if penalty == 1.0:
            continue
        window = [
            tok
            for tok in map(int, data.req.output_ids[-REP_PENALTY_WINDOW:])
            if 0 <= tok < vocab
        ]
        for tok, count in Counter(window).items():
            alpha = penalty**count
            score = out[row, tok]
            out[row, tok] = score * alpha if score < 0 else score / alpha
    return out.to(logits.dtype)


def _make_request(output_ids: list[int], rep_penalty: float) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(
            talker_model_inputs={"rep_penalty": rep_penalty},
            req=SimpleNamespace(output_ids=output_ids),
        )
    )


def test_vectorized_penalty_matches_reference() -> None:
    torch.manual_seed(0)
    vocab = 64
    requests = [
        # Long history with repeats: only the last window counts.
        _make_request([5] * 40 + [7, 7, 7, 9, 9, 5], 1.1),
        # penalty == 1.0: row must be untouched.
        _make_request([1, 2, 3], 1.0),
        # Empty output_ids: nothing to penalize.
        _make_request([], 1.4),
        # Out-of-range tokens are dropped from the window.
        _make_request([-1, vocab, vocab + 5, 3, 3], 2.0),
        # Window entirely out-of-range: row must be untouched.
        _make_request([vocab, vocab + 1], 1.5),
        # Penalty < 1 (boost) with a full window of one token.
        _make_request([11] * (REP_PENALTY_WINDOW + 4), 0.5),
    ]
    logits = torch.randn(len(requests), vocab) * 8
    expected = _reference_penalty(logits, requests)

    logits_output = SimpleNamespace(next_token_logits=logits.clone())
    runner = MiniCPMOTalkerModelRunner.__new__(MiniCPMOTalkerModelRunner)
    runner._apply_repetition_penalty(logits_output, requests)

    torch.testing.assert_close(logits_output.next_token_logits, expected)


def test_penalty_noop_rows_bitwise_unchanged() -> None:
    torch.manual_seed(1)
    vocab = 32
    requests = [
        _make_request([4, 4, 4], 1.2),
        _make_request([8, 8], 1.0),
    ]
    logits = torch.randn(len(requests), vocab)
    original = logits.clone()

    logits_output = SimpleNamespace(next_token_logits=logits)
    runner = MiniCPMOTalkerModelRunner.__new__(MiniCPMOTalkerModelRunner)
    runner._apply_repetition_penalty(logits_output, requests)

    # The untouched row must be bitwise identical (never round-tripped).
    assert torch.equal(logits_output.next_token_logits[1], original[1])
    assert not torch.equal(logits_output.next_token_logits[0], original[0])


def test_sampling_logits_hook_applies_penalty() -> None:
    request = _make_request([3, 3], 2.0)
    logits = torch.ones(1, 8)
    logits_output = SimpleNamespace(next_token_logits=logits)
    runner = MiniCPMOTalkerModelRunner.__new__(MiniCPMOTalkerModelRunner)

    runner._process_sampling_logits(logits_output, [request])

    assert logits_output.next_token_logits[0, 3].item() == 0.25
