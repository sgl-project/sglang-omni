# SPDX-License-Identifier: Apache-2.0
"""apply_higgs_result -> HiggsTtsState.omni_rollout -> to_dict serialization."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.models.higgs_tts.request_builders import apply_higgs_result
from sglang_omni.models.higgs_tts.utils import apply_delay_pattern

N = 8
V = 1026


def _fake_data(*, return_omni_rollout, return_logprob, t_raw=6):
    delayed = apply_delay_pattern(torch.randint(0, 1024, (t_raw, N)))
    return SimpleNamespace(
        output_codes=list(delayed.unbind(0)),
        output_logprobs=list(torch.randn(*delayed.shape).unbind(0)),
        num_codebooks=N,
        codebook_size=V,
        return_omni_rollout=return_omni_rollout,
        return_logprob=return_logprob,
        input_ids=list(range(5)),
    )


def test_omni_rollout_built_and_roundtrips():
    torch.manual_seed(0)
    state = HiggsTtsState(num_codebooks=N, codebook_size=V)
    apply_higgs_result(state, _fake_data(return_omni_rollout=True, return_logprob=True))

    stream = state.omni_rollout["action_streams"][0]
    assert stream["name"] == "higgs_codes"
    assert stream["logprobs"] is not None
    assert state.omni_rollout["total_action_count"] == 6 * N
    # Survives the StagePayload dict round-trip the client reads from.
    assert HiggsTtsState.from_dict(state.to_dict()).omni_rollout == state.omni_rollout


def test_flag_gating():
    torch.manual_seed(1)
    # no rollout flag -> nothing emitted.
    off = HiggsTtsState(num_codebooks=N, codebook_size=V)
    apply_higgs_result(off, _fake_data(return_omni_rollout=False, return_logprob=True))
    assert off.omni_rollout is None
    assert "omni_rollout" not in off.to_dict()

    # rollout but no logprob flag -> trace without logprobs.
    no_lp = HiggsTtsState(num_codebooks=N, codebook_size=V)
    apply_higgs_result(
        no_lp, _fake_data(return_omni_rollout=True, return_logprob=False)
    )
    assert no_lp.omni_rollout["action_streams"][0]["logprobs"] is None
