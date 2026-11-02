# SPDX-License-Identifier: Apache-2.0
"""Tests for the target worker SGLang's draft workers call."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from sglang_omni.speculative.target_worker import SpeculativeTargetWorker


class TargetRunner:
    """Narrows the hidden states each forward hands the draft, as a runner sharing the capture does."""

    def __init__(self, model_worker: SimpleNamespace) -> None:
        self.tp_worker = model_worker
        self.hook_calls: list[
            tuple[SimpleNamespace, SimpleNamespace | None, list[str]]
        ] = []

    def prepare_and_forward(
        self,
        forward_batch: SimpleNamespace,
        batch: SimpleNamespace,
        requests: list[SimpleNamespace],
        is_prefill: bool,
    ) -> GenerationBatchResult:
        return GenerationBatchResult(
            logits_output=SimpleNamespace(hidden_states=torch.zeros(2, 4)),
            next_token_ids=torch.tensor([7, 8]),
        )

    def post_speculative_target_forward(
        self,
        result: GenerationBatchResult,
        forward_batch: SimpleNamespace,
        schedule_batch: SimpleNamespace | None,
        requests: list[SimpleNamespace],
    ) -> None:
        self.hook_calls.append(
            (
                forward_batch,
                schedule_batch,
                [request.request_id for request in requests],
            )
        )
        result.logits_output.hidden_states = result.logits_output.hidden_states[..., :2]


def test_every_target_forward_runs_the_runner_hook_before_the_draft_reads_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the hook on prefill the talker's prompt rows are never stored, and
    without it on verify the draft reads the talker's layer as one of its own."""
    prefill_forward_batch = SimpleNamespace(kv_loc_plan=None)
    monkeypatch.setattr(
        ForwardBatch,
        "init_new",
        staticmethod(lambda batch, model_runner, **options: prefill_forward_batch),
    )
    model_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            forward=lambda forward_batch: SimpleNamespace(
                logits_output=SimpleNamespace(hidden_states=torch.zeros(3, 4)),
                can_run_graph=False,
                expert_distribution_metrics=None,
                routed_experts_output=None,
                indexer_topk_output=None,
            )
        ),
        model_config=None,
        device="cpu",
        random_seed=0,
    )
    target_worker = SpeculativeTargetWorker(model_worker, SpeculativeAlgorithm.DFLASH)
    runner = TargetRunner(model_worker)
    target_worker.attach_runner(runner)
    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(rid="request-a", omni_data=None),
            SimpleNamespace(rid="request-b", omni_data=None),
        ],
        is_prefill_only=False,
    )
    verify_forward_batch = SimpleNamespace(
        kv_loc_plan=None,
        apply_deprecated_skip_attn_backend_init=lambda skip_attn_backend_init: None,
    )

    prefill = target_worker.forward_batch_generation(batch)
    verify = target_worker.forward_batch_generation(
        None, forward_batch=verify_forward_batch, is_verify=True
    )

    assert runner.hook_calls == [
        (prefill_forward_batch, batch, ["request-a", "request-b"]),
        (verify_forward_batch, None, []),
    ]
    assert prefill.logits_output.hidden_states.shape == (2, 2)
    assert verify.logits_output.hidden_states.shape == (3, 2)


def test_prefill_hands_the_draft_its_kv_location_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Draft workers ask the target prefill for its KV location plan; the adapter
    refused the keyword, so every draft failed on its first request."""
    kv_location_plan = SimpleNamespace()
    forward_batch = SimpleNamespace(kv_loc_plan=kv_location_plan)
    monkeypatch.setattr(
        ForwardBatch,
        "init_new",
        staticmethod(lambda batch, model_runner, **options: forward_batch),
    )
    model_worker = SimpleNamespace(
        model_runner=None, model_config=None, device="cpu", random_seed=0
    )
    target_worker = SpeculativeTargetWorker(model_worker, SpeculativeAlgorithm.DFLASH)
    target_worker.attach_runner(TargetRunner(model_worker))
    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(rid="request-a", omni_data=None),
            SimpleNamespace(rid="request-b", omni_data=None),
        ],
        is_prefill_only=False,
    )

    planned = target_worker.forward_batch_generation(batch, return_kv_loc_plan=True)
    unplanned = target_worker.forward_batch_generation(batch)

    assert planned.kv_loc_plan is kv_location_plan
    assert unplanned.kv_loc_plan is None
    assert planned.next_token_ids.tolist() == [7, 8]
