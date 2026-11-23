# SPDX-License-Identifier: Apache-2.0
"""Tests for the target worker SGLang's draft workers call."""

from __future__ import annotations

from array import array
from types import SimpleNamespace
from typing import Literal

import pytest
import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.penalties import PenaltyStatusPool, StepPenalties
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.target_verify import adjust_verify_logits

from sglang_omni.speculative.target_worker import SpeculativeTargetWorker


class PrefillRunner:
    def __init__(self, model_worker: SimpleNamespace) -> None:
        self.tp_worker = model_worker

    def prepare_and_forward(
        self,
        forward_batch: SimpleNamespace,
        batch: SimpleNamespace,
        requests: list[SimpleNamespace],
        is_prefill: bool,
    ) -> GenerationBatchResult:
        return GenerationBatchResult(next_token_ids=torch.tensor([7, 8]))


def test_prefill_hands_the_draft_its_iteration_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Draft workers need the target's iteration IDs to resolve their KV slots."""
    iteration_ids = SimpleNamespace()
    forward_batch = SimpleNamespace(iteration_ids=iteration_ids)
    monkeypatch.setattr(
        ForwardBatch,
        "init_new",
        staticmethod(lambda batch, model_runner, **options: forward_batch),
    )
    model_worker = SimpleNamespace(
        model_runner=None, model_config=None, device="cpu", random_seed=0
    )
    target_worker = SpeculativeTargetWorker(model_worker, SpeculativeAlgorithm.DFLASH)
    target_worker.attach_runner(PrefillRunner(model_worker))
    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(rid="request-a", omni_data=None),
            SimpleNamespace(rid="request-b", omni_data=None),
        ],
        is_prefill_only=False,
    )

    planned = target_worker.forward_batch_generation(batch, return_iteration_ids=True)
    unplanned = target_worker.forward_batch_generation(batch)

    assert planned.iteration_ids is iteration_ids
    assert unplanned.iteration_ids is None
    assert planned.next_token_ids.tolist() == [7, 8]


class VerifyRunner:
    def __init__(
        self,
        model_worker: SimpleNamespace,
        takes_logits_hooks: bool,
        repetition_penalty_owner: Literal["draft_worker", "verify_logits"] | None,
    ) -> None:
        self.tp_worker = model_worker
        self.speculative_verify_takes_logits_hooks = takes_logits_hooks
        self.speculative_repetition_penalty_owner = repetition_penalty_owner
        self.applied_hooks: list[str] = []

    def apply_speculative_verify_logits(
        self,
        logits_output: SimpleNamespace,
        requests: list[SimpleNamespace],
        rows_per_request: int,
    ) -> None:
        self.applied_hooks.append(f"logits hooks, {rows_per_request} rows per request")

    def apply_speculative_repetition_penalty(
        self,
        verify_logits: torch.Tensor,
        verify_batch: SimpleNamespace,
        requests: list[SimpleNamespace],
    ) -> None:
        self.applied_hooks.append(f"repetition penalty on {tuple(verify_logits.shape)}")


@pytest.mark.parametrize(
    ("takes_logits_hooks", "repetition_penalty_owner", "applied_hooks"),
    [
        (True, None, ["logits hooks, 3 rows per request"]),
        (False, "verify_logits", []),
        (False, "draft_worker", []),
    ],
)
def test_verify_rows_take_only_the_hooks_their_runner_opted_into(
    takes_logits_hooks: bool,
    repetition_penalty_owner: Literal["draft_worker", "verify_logits"] | None,
    applied_hooks: list[str],
) -> None:
    """The main verify sampler owns standard penalties; target hooks must not double them."""
    verify_output = SimpleNamespace(
        logits_output=SimpleNamespace(next_token_logits=torch.zeros(6, 5)),
        can_run_graph=False,
        expert_distribution_metrics=None,
        routed_experts_output=None,
        indexer_topk_output=None,
    )
    model_worker = SimpleNamespace(
        model_runner=SimpleNamespace(forward=lambda forward_batch: verify_output),
        model_config=None,
        device="cpu",
        random_seed=0,
    )
    target_worker = SpeculativeTargetWorker(model_worker, SpeculativeAlgorithm.DFLASH)
    runner = VerifyRunner(model_worker, takes_logits_hooks, repetition_penalty_owner)
    target_worker.attach_runner(runner)
    target_worker.verify_requests = [SimpleNamespace(), SimpleNamespace()]
    verify_batch = SimpleNamespace(
        apply_deprecated_skip_attn_backend_init=lambda skip_attn_backend_init: None,
        iteration_ids=None,
    )

    target_worker.forward_batch_generation(None, verify_batch, is_verify=True)

    assert runner.applied_hooks == applied_hooks


def test_verify_sampler_applies_each_rows_history_once() -> None:
    """Target-side penalties would penalize history twice before the verify sampler."""
    with get_context().override_server_args():
        request = Req(
            rid="verify-history",
            origin_input_text="",
            origin_input_ids=array("q", [0]),
            sampling_params=SamplingParams(temperature=0, repetition_penalty=2.0),
        )
        request.output_ids = array("q", [1])
        pool = PenaltyStatusPool(vocab_size=5, device="cpu")
        pool.alloc([request])
        pool.reset_history([request])
        sampling_info = SamplingBatchInfo(
            vocab_size=5,
            device="cpu",
            step_penalties=StepPenalties.build([request], pool),
        )
        logits = torch.tensor([[4.0, 4.0, -4.0, 4.0, 4.0]]).repeat(3, 1)
        verify_output = SimpleNamespace(
            logits_output=SimpleNamespace(next_token_logits=logits),
            can_run_graph=False,
            expert_distribution_metrics=None,
            routed_experts_output=None,
            indexer_topk_output=None,
        )
        model_worker = SimpleNamespace(
            model_runner=SimpleNamespace(forward=lambda forward_batch: verify_output),
            model_config=None,
            device="cpu",
            random_seed=0,
        )
        target_worker = SpeculativeTargetWorker(
            model_worker, SpeculativeAlgorithm.DFLASH
        )
        runner = VerifyRunner(model_worker, False, "verify_logits")
        target_worker.attach_runner(runner)
        target_worker.verify_requests = [SimpleNamespace()]
        verify_batch = SimpleNamespace(
            apply_deprecated_skip_attn_backend_init=lambda value: None,
            iteration_ids=None,
        )
        result = target_worker.forward_batch_generation(
            None, verify_batch, is_verify=True
        )
        assert runner.applied_hooks == []
        with torch._dynamo.config.patch(
            disable=True
        ):  # noqa: leading-underscore  # disable compilation for the CPU numerical check
            adjust_verify_logits(
                result.logits_output.next_token_logits,
                sampling_info=sampling_info,
                grammar_mask=None,
                draft_tokens=torch.tensor([[1, 2, 3]]),
                has_custom_logit_processor=False,
            )
        torch.testing.assert_close(
            logits,
            torch.tensor(
                [
                    [4.0, 2.0, -4.0, 4.0, 4.0],
                    [4.0, 2.0, -8.0, 4.0, 4.0],
                    [4.0, 2.0, -8.0, 2.0, 4.0],
                ]
            ),
        )
