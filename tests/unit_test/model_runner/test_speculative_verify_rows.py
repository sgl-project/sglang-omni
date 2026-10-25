# SPDX-License-Identifier: Apache-2.0
"""The logits hooks a speculative verify gives its rows."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.model_runner.base import ModelRunner

SUPPRESSED_TOKEN_ID = 2


def bare_runner() -> ModelRunner:
    return object.__new__(ModelRunner)


def request(suppress_tokens: list[int] | None) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(suppress_tokens=suppress_tokens, req=SimpleNamespace())
    )


def logits_output(rows: int) -> SimpleNamespace:
    return SimpleNamespace(
        next_token_logits=torch.tensor([[0.5, 2.0, -1.0, -4.0]]).repeat(rows, 1)
    )


def test_each_request_s_rows_take_its_suppress_set_under_a_uniform_split() -> None:
    output = logits_output(6)

    bare_runner().apply_speculative_verify_logits(
        output, [request(None), request([SUPPRESSED_TOKEN_ID])], 3
    )

    logits = output.next_token_logits
    assert torch.isfinite(logits[:3]).all()
    assert torch.isinf(logits[3:, SUPPRESSED_TOKEN_ID]).all()
    assert torch.isfinite(logits[3:, :SUPPRESSED_TOKEN_ID]).all()


def test_a_compact_verify_s_packed_rows_take_their_request_s_set_and_the_padding_none() -> (
    None
):
    # Requests verified 1 and 2 rows, packed back to back and padded to a tier of 4.
    output = logits_output(4)

    bare_runner().apply_speculative_verify_logits(
        output,
        [request(None), request([SUPPRESSED_TOKEN_ID])],
        torch.tensor([1, 2], dtype=torch.int32),
    )

    logits = output.next_token_logits
    assert torch.isfinite(logits[0]).all()
    assert torch.isinf(logits[1:3, SUPPRESSED_TOKEN_ID]).all()
    assert torch.isfinite(
        logits[3]
    ).all(), "the tier's padding row belongs to no request"


def test_requests_without_hooks_never_read_the_compact_row_counts() -> None:
    output = logits_output(4)
    # note (0xtoward): reading a meta tensor raises, as the host copy of device counts would cost a sync.
    unreadable_counts = torch.empty(2, dtype=torch.int32, device="meta")

    bare_runner().apply_speculative_verify_logits(
        output, [request(None), request(None)], unreadable_counts
    )

    assert torch.isfinite(output.next_token_logits).all()
