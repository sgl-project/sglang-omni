# SPDX-License-Identifier: Apache-2.0
"""Rollout logprob recording uses sampler-computed logprobs."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner.base import ModelRunner


@pytest.mark.parametrize(
    "sampler_logprobs",
    [torch.tensor([-0.5, -1.5]), torch.tensor([[-0.5], [-1.5]])],
    ids=["flat", "column"],
)
def test_rollout_logprobs_align_per_request_at_batch_2(
    sampler_logprobs: torch.Tensor,
) -> None:
    runner = object.__new__(ModelRunner)
    final_chunk = SimpleNamespace(inflight_middle_chunks=0)
    data0 = SimpleNamespace(
        return_logprob=True, output_token_logprobs=[], req=final_chunk
    )
    data1 = SimpleNamespace(
        return_logprob=True, output_token_logprobs=[], req=final_chunk
    )
    requests = [SimpleNamespace(data=data0), SimpleNamespace(data=data1)]

    runner.record_rollout_logprobs(sampler_logprobs, torch.tensor([11, 22]), requests)

    # each request must get ITS OWN row, not the other request's token/logprob
    assert data0.output_token_logprobs[0][1] == 11
    assert data1.output_token_logprobs[0][1] == 22
    assert math.isclose(data0.output_token_logprobs[0][0], -0.5, abs_tol=1e-4)
    assert math.isclose(data1.output_token_logprobs[0][0], -1.5, abs_tol=1e-4)


def test_rollout_logprobs_skip_a_middle_prefill_chunk() -> None:
    """A prompt prefilled in two chunks commits only the last chunk's sample, so
    its logprob list used to start with one for a token that was never output."""
    runner = object.__new__(ModelRunner)
    req = SimpleNamespace(inflight_middle_chunks=1)
    data = SimpleNamespace(return_logprob=True, output_token_logprobs=[], req=req)
    request = SimpleNamespace(data=data)

    runner.record_rollout_logprobs(torch.tensor([-2.0]), torch.tensor([99]), [request])
    req.inflight_middle_chunks = 0
    runner.record_rollout_logprobs(torch.tensor([-0.5]), torch.tensor([5]), [request])
    runner.record_rollout_logprobs(torch.tensor([-0.25]), torch.tensor([6]), [request])

    assert data.output_token_logprobs == [[-0.5, 5], [-0.25, 6]]


def test_rollout_logprobs_raises_on_batch_size_mismatch() -> None:
    runner = object.__new__(ModelRunner)
    data = SimpleNamespace(return_logprob=True, output_token_logprobs=[])

    # more logprobs than requests => batching assumption broke => fail loud
    with pytest.raises(RuntimeError, match="batch-size mismatch"):
        runner.record_rollout_logprobs(
            torch.tensor([-0.5, -1.5]),
            torch.tensor([11, 22]),
            [SimpleNamespace(data=data)],
        )

    assert data.output_token_logprobs == []


def test_rollout_logprobs_raises_on_malformed_sampler_shape() -> None:
    runner = object.__new__(ModelRunner)
    data = SimpleNamespace(return_logprob=True, output_token_logprobs=[])

    with pytest.raises(RuntimeError, match="Failed to convert"):
        runner.record_rollout_logprobs(
            [[-0.5, -0.7]],
            torch.tensor([11]),
            [SimpleNamespace(data=data)],
        )

    assert data.output_token_logprobs == []


def test_rollout_logprobs_raises_when_sampler_omits_token_ids() -> None:
    runner = object.__new__(ModelRunner)
    data = SimpleNamespace(return_logprob=True, output_token_logprobs=[])

    with pytest.raises(RuntimeError, match="next_token_ids"):
        runner.record_rollout_logprobs(
            torch.tensor([-0.5]),
            None,
            [SimpleNamespace(data=data)],
        )

    assert data.output_token_logprobs == []


def test_enable_sampler_logprobs_initializes_missing_forward_batch_fields() -> None:
    forward_batch = SimpleNamespace(top_logprobs_nums=None, token_ids_logprobs=None)

    ModelRunner.enable_sampler_logprobs(forward_batch, batch_size=2)

    assert forward_batch.return_logprob is True
    assert forward_batch.top_logprobs_nums == [0, 0]
    assert forward_batch.token_ids_logprobs == [None, None]


def test_record_rollout_logprobs_skips_without_return_flag() -> None:
    runner = object.__new__(ModelRunner)
    data = SimpleNamespace(return_logprob=False, output_token_logprobs=[])

    runner.record_rollout_logprobs(
        torch.tensor([-0.25]), torch.tensor([33]), [SimpleNamespace(data=data)]
    )

    assert data.output_token_logprobs == []


def test_sample_next_token_ids_requires_sampler_logprobs_when_requested() -> None:
    runner = object.__new__(ModelRunner)
    runner.apply_codec_suppress_tokens = lambda *args: None
    runner.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            sample=lambda _logits_output, _forward_batch: torch.tensor([44])
        )
    )
    req = SimpleNamespace(sampling_params=SimpleNamespace(sampling_seed=None))
    data = SimpleNamespace(return_logprob=True, output_token_logprobs=[], req=req)
    request = SimpleNamespace(data=data)
    forward_batch = SimpleNamespace(
        sampling_info=SimpleNamespace(device="cpu", sampling_seed=None),
        top_logprobs_nums=None,
        token_ids_logprobs=None,
    )
    logits_output = SimpleNamespace()

    with pytest.raises(RuntimeError, match="next_token_logprobs"):
        runner.sample_next_token_ids(
            logits_output,
            forward_batch,
            SimpleNamespace(),
            [request],
        )

    assert forward_batch.return_logprob is True
    assert data.output_token_logprobs == []
