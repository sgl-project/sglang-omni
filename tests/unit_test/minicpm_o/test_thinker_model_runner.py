# SPDX-License-Identifier: Apache-2.0
"""Thinker length penalty scales only the end-of-sequence logits."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.minicpm_o.thinker_model_runner import MiniCPMOThinkerModelRunner
from sglang_omni.proto import OmniRequest, StagePayload

EOS_TOKEN_IDS = [1, 3]


def bare_runner() -> MiniCPMOThinkerModelRunner:
    runner = object.__new__(MiniCPMOThinkerModelRunner)
    runner.eos_token_ids = EOS_TOKEN_IDS
    return runner


def sampling_params() -> SimpleNamespace:
    return SimpleNamespace(
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        min_new_tokens=0,
        sampling_seed=None,
        logit_bias=None,
        custom_params=None,
    )


def stage_payload(params: dict[str, float]) -> StagePayload:
    return StagePayload(
        request_id="request-0",
        request=OmniRequest(
            inputs=None,
            params=params,
            metadata={"output_modalities": ["text"]},
        ),
        data=None,
    )


def schedule_request(params: dict[str, float]) -> SimpleNamespace:
    return SimpleNamespace(
        sampling_params=sampling_params(),
        omni_data=SimpleNamespace(
            return_logprob=False,
            stage_payload=stage_payload(params),
        ),
    )


def request_with_params(params: dict[str, float]) -> SimpleNamespace:
    return SimpleNamespace(data=SimpleNamespace(stage_payload=stage_payload(params)))


def test_length_penalty_scales_eos_logits_per_request() -> None:
    original = torch.tensor(
        [
            [0.5, 2.0, -1.0, -4.0],
            [0.5, 2.0, -1.0, -4.0],
            [0.5, 2.0, -1.0, -4.0],
        ]
    )
    logits_output = SimpleNamespace(next_token_logits=original.clone())

    bare_runner().process_sampling_logits(
        logits_output,
        [
            request_with_params({"length_penalty": 1.0}),
            request_with_params({"length_penalty": 2.0}),
            request_with_params({}),
        ],
    )

    torch.testing.assert_close(logits_output.next_token_logits[0], original[0])
    torch.testing.assert_close(logits_output.next_token_logits[2], original[2])
    expected = original[1].clone()
    expected[EOS_TOKEN_IDS] = torch.tensor([2.0 / 2.0, -4.0 * 2.0])
    torch.testing.assert_close(logits_output.next_token_logits[1], expected)


def test_length_penalty_disables_lookahead() -> None:
    runner = bare_runner()
    plain = SimpleNamespace(
        reqs=[schedule_request({}), schedule_request({"length_penalty": 1.0})]
    )
    penalized = SimpleNamespace(
        reqs=[
            schedule_request({"length_penalty": 1.0}),
            schedule_request({"length_penalty": 1.5}),
        ]
    )

    assert runner.lookahead_eligible(plain) is True
    assert runner.lookahead_eligible(penalized) is False
