# SPDX-License-Identifier: Apache-2.0
"""Tests for :func:`build_omni_rollout_trace` (meta_info.omni_rollout builder).

Mask geometry itself is covered by ``test_action_mask``; here we pin the schema /
action count and the fail-loud input guards.
"""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.higgs_tts.rollout_trace import (
    OMNI_ROLLOUT_VERSION,
    build_omni_rollout_trace,
)
from sglang_omni.models.higgs_tts.utils import apply_delay_pattern

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N = 8
V = 1026


def _delayed(t_raw: int, n: int = N) -> torch.Tensor:
    return apply_delay_pattern(torch.randint(0, 1024, (t_raw, n), device=DEVICE))


def test_schema_actions_logprobs_and_action_count():
    torch.manual_seed(0)
    t_raw = 10
    delayed = _delayed(t_raw)
    lp = torch.randn(*delayed.shape, device=DEVICE)

    trace = build_omni_rollout_trace(
        delayed, num_codebooks=N, codebook_vocab_size=V, delayed_logprobs=lp
    )

    assert trace["version"] == OMNI_ROLLOUT_VERSION
    assert trace["model_family"] == "higgs_tts"
    s = trace["action_streams"][0]
    assert (s["name"], s["layout"], s["action_type"]) == (
        "higgs_codes",
        "codebook_2d",
        "discrete",
    )
    assert s["shape"] == list(delayed.shape)
    assert s["channel_ids"] == list(range(N))
    # actions / logprobs serialize verbatim and aligned.
    assert s["actions"] == delayed.to(torch.long).tolist()
    assert torch.allclose(torch.tensor(s["logprobs"], device=DEVICE), lp, atol=1e-5)
    # total_action_count == sum(mask) == T*N.
    assert trace["total_action_count"] == sum(sum(row) for row in s["action_mask"])
    assert trace["total_action_count"] == t_raw * N


def test_input_guards():
    delayed = _delayed(6)

    with pytest.raises(ValueError, match="codebooks"):
        build_omni_rollout_trace(delayed, num_codebooks=N + 1, codebook_vocab_size=V)

    with pytest.raises(ValueError, match="shape"):
        build_omni_rollout_trace(
            delayed,
            num_codebooks=N,
            codebook_vocab_size=V,
            delayed_logprobs=torch.randn(delayed.shape[0] + 1, N, device=DEVICE),
        )

    # NaN on a real-action cell (3, 0) fails loud...
    lp = torch.randn(*delayed.shape, device=DEVICE)
    lp[3, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        build_omni_rollout_trace(
            delayed, num_codebooks=N, codebook_vocab_size=V, delayed_logprobs=lp
        )

    # ...but a non-finite value on a masked (BOC scaffolding) cell is harmless.
    lp = torch.randn(*delayed.shape, device=DEVICE)
    lp[0, N - 1] = float("-inf")
    trace = build_omni_rollout_trace(
        delayed, num_codebooks=N, codebook_vocab_size=V, delayed_logprobs=lp
    )
    assert trace["total_action_count"] > 0
