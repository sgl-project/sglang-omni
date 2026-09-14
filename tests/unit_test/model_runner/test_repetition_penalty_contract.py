# SPDX-License-Identifier: Apache-2.0
"""Penalty ownership at the shared sampler boundary."""

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.model_runner.base import ModelRunner


def test_qwen3_omni_thinker_explicit_penalty_is_preserved():
    from sglang_omni.models.qwen3_omni.request_builders import (
        build_sglang_thinker_request,
    )
    from tests.unit_test.fixtures.qwen_fakes import FakeQwenTokenizer, make_qwen_state

    data = build_sglang_thinker_request(
        make_qwen_state(),
        params={"repetition_penalty": 1.7},
        tokenizer=FakeQwenTokenizer(),
        vocab_size=32000,
    )
    assert isinstance(data.req.sampling_params, SamplingParams)
    assert data.req.sampling_params.repetition_penalty == 1.7


@pytest.mark.parametrize("with_sglang_state", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_shared_sampler_receives_only_codec_shaping(with_sglang_state, dtype):
    runner = object.__new__(ModelRunner)
    original = torch.tensor([[4.0, -6.0, 8.0, 3.0], [-4.0, 6.0, 2.0, 1.0]], dtype=dtype)
    expected = original.clone()
    expected[0, 3] = -float("inf")
    seen = []

    def sample(output, forward):
        seen.append(output.next_token_logits.clone())
        return torch.tensor([2, 1])

    runner.tp_worker = SimpleNamespace(model_runner=SimpleNamespace(sample=sample))
    requests = [
        SimpleNamespace(
            data=SimpleNamespace(
                req=SimpleNamespace(
                    sampling_params=SimpleNamespace(
                        repetition_penalty=p, sampling_seed=None
                    ),
                    output_ids=[0, 1, 0],
                ),
                suppress_tokens=suppress,
                return_logprob=False,
            )
        )
        for p, suppress in [(2.0, [3]), (0.5, None)]
    ]
    forward = SimpleNamespace(
        sampling_info=SimpleNamespace(
            sampling_seed=None,
            penalizer_orchestrator=None,
            acc_scaling_penalties=torch.ones(2, 4) if with_sglang_state else None,
        )
    )
    result = runner._sample_next_token_ids(
        SimpleNamespace(next_token_logits=original.clone()),
        forward,
        SimpleNamespace(),
        requests,
    )
    assert result.tolist() == [2, 1]
    assert len(seen) == 1
    assert torch.equal(seen[0], expected)
