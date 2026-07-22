# SPDX-License-Identifier: Apache-2.0
"""Higgs omni-rollout serialization through pipeline state."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.models.higgs_tts.request_builders import apply_higgs_result
from sglang_omni.models.higgs_tts.utils import (
    apply_delay_pattern,
    delay_pattern_codec_content_mask,
)

N = 8
V = 1026


def _fake_data(*, return_logprob, return_omni_rollout=True, t_raw=6):
    delayed = apply_delay_pattern(torch.randint(0, 1024, (t_raw, N)))
    action_mask = delay_pattern_codec_content_mask(delayed)
    action_mask[t_raw, 0] = True
    return SimpleNamespace(
        output_code_buffer=None,
        output_code_count=0,
        output_codes=list(delayed.unbind(0)),
        output_action_masks=list(action_mask.unbind(0)),
        output_logprobs=list(torch.randn(*delayed.shape).unbind(0)),
        num_codebooks=N,
        codebook_size=V,
        return_logprob=return_logprob,
        return_omni_rollout=return_omni_rollout,
        input_ids=list(range(5)),
        weight_version="7",
    )


def test_omni_rollout_built_and_roundtrips():
    torch.manual_seed(0)
    state = HiggsTtsState(num_codebooks=N, codebook_size=V)
    data = _fake_data(return_logprob=True)
    apply_higgs_result(state, data)

    stream = state.omni_rollout["action_streams"][0]
    assert stream["stage"] == "tts_engine"
    assert stream["logprobs"] is not None
    assert state.omni_rollout["version"] == 1
    assert state.weight_version == "7"
    # Survives the StagePayload dict round-trip the client reads from.
    assert HiggsTtsState.from_dict(state.to_dict()).omni_rollout == state.omni_rollout
    assert HiggsTtsState.from_dict(state.to_dict()).weight_version == "7"


def test_flag_gating():
    torch.manual_seed(1)
    off = HiggsTtsState(num_codebooks=N, codebook_size=V)
    apply_higgs_result(off, _fake_data(return_logprob=False, return_omni_rollout=False))
    assert off.omni_rollout is None
    assert "omni_rollout" not in off.to_dict()

    no_lp = HiggsTtsState(num_codebooks=N, codebook_size=V)
    data = _fake_data(return_logprob=True)
    data.output_logprobs = []
    with pytest.raises(ValueError, match="missing aligned"):
        apply_higgs_result(no_lp, data)


def test_rollout_logprob_shape_mismatch_fails_loudly():
    torch.manual_seed(5)
    data = _fake_data(return_logprob=True, t_raw=3)
    data.output_logprobs = data.output_logprobs[:-1]
    state = HiggsTtsState(num_codebooks=N, codebook_size=V)

    with pytest.raises(ValueError, match="rollout logprob shape"):
        apply_higgs_result(state, data)
