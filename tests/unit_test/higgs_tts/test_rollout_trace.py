# SPDX-License-Identifier: Apache-2.0
"""Higgs omni rollout validation and serialization."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.higgs_tts.rollout_trace import (
    OMNI_ROLLOUT_VERSION,
    build_omni_rollout_trace,
)
from sglang_omni.models.higgs_tts.utils import (
    apply_delay_pattern,
    delay_pattern_codec_content_mask,
)

N = 8
V = 1026


def _inputs(t_raw: int = 4):
    codes = apply_delay_pattern(torch.randint(0, 1024, (t_raw, N)))
    mask = delay_pattern_codec_content_mask(codes)
    mask[t_raw, 0] = True  # sampled terminating EOC is an action, not content
    logprobs = torch.randn(codes.shape)
    return codes, mask, logprobs


def test_schema_uses_explicit_sample_time_mask():
    codes, mask, logprobs = _inputs()
    trace = build_omni_rollout_trace(
        codes,
        num_codebooks=N,
        codebook_vocab_size=V,
        delayed_logprobs=logprobs,
        action_mask=mask,
    )

    assert trace["version"] == OMNI_ROLLOUT_VERSION == 1
    assert trace["model_family"] == "higgs_tts"
    assert trace["stages"] == ["tts_engine"]
    assert trace["total_action_count"] == int(mask.sum())
    stream = trace["action_streams"][0]
    assert stream["name"] == "higgs_codes"
    assert stream["layout"] == "codebook_2d"
    assert stream["shape"] == list(codes.shape)
    assert stream["actions"] == codes.tolist()
    assert stream["action_mask"] == mask.to(torch.int64).tolist()
    serialized_lp = torch.tensor(stream["logprobs"])
    assert torch.all(serialized_lp[~mask] == 0)
    assert torch.allclose(serialized_lp[mask], logprobs[mask])


def test_input_guards_and_nonfinite_masking():
    codes, mask, logprobs = _inputs()
    kwargs = dict(
        num_codebooks=N,
        codebook_vocab_size=V,
        delayed_logprobs=logprobs,
        action_mask=mask,
    )
    with pytest.raises(ValueError, match="codebooks"):
        build_omni_rollout_trace(codes, **{**kwargs, "num_codebooks": N + 1})
    with pytest.raises(ValueError, match="action_mask shape"):
        build_omni_rollout_trace(codes, **{**kwargs, "action_mask": mask[:-1]})
    with pytest.raises(ValueError, match="delayed_logprobs shape"):
        build_omni_rollout_trace(codes, **{**kwargs, "delayed_logprobs": logprobs[:-1]})

    row, channel = mask.nonzero()[0]
    logprobs[row, channel] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        build_omni_rollout_trace(codes, **kwargs)
    codes[0, 0] = V
    with pytest.raises(ValueError, match="vocabulary"):
        build_omni_rollout_trace(codes, **kwargs)
