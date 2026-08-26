# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.vendor.sglang.sampling import _build_logit_bias


def _req(logit_bias: dict | None) -> SimpleNamespace:
    return SimpleNamespace(sampling_params=SimpleNamespace(logit_bias=logit_bias))


def test_build_logit_bias_returns_none_without_any_bias() -> None:
    reqs = [_req(None), _req(None)]
    assert _build_logit_bias(reqs, vocab_size=8, device="cpu") is None


def test_build_logit_bias_writes_each_key_for_its_own_row() -> None:
    reqs = [_req({"1": -5.0, "3": 2.0}), _req(None), _req({"2": 4.0})]

    logit_bias = _build_logit_bias(reqs, vocab_size=8, device="cpu")

    expected = torch.zeros(3, 8)
    expected[0, 1] = -5.0
    expected[0, 3] = 2.0
    expected[2, 2] = 4.0
    assert torch.equal(logit_bias, expected)


def test_build_logit_bias_resolves_duplicate_canonical_keys_last_write_wins() -> None:
    # "05" and "5" both canonicalize to token id 5 via int(key); a dict
    # keeps only the last value written for a given (row, token_id), which
    # is also what the original sequential per-element loop produced.
    reqs = [_req({"05": 1.0, "5": 9.0})]

    logit_bias = _build_logit_bias(reqs, vocab_size=8, device="cpu")

    expected = torch.zeros(1, 8)
    expected[0, 5] = 9.0
    assert torch.equal(logit_bias, expected)
