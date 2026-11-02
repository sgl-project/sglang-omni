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
    target_worker.attach_runner(PrefillRunner(model_worker))
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


class VerifyRunner:
    def __init__(self, model_worker: SimpleNamespace) -> None:
        self.tp_worker = model_worker
        self.calls: list[tuple[list[SimpleNamespace], int | torch.Tensor]] = []

    def apply_speculative_verify_logits(
        self,
        logits_output: SimpleNamespace,
        requests: list[SimpleNamespace],
        rows_per_request: int | torch.Tensor,
    ) -> None:
        self.calls.append((requests, rows_per_request))


def verify_target_worker(
    logits_rows: int,
) -> tuple[SpeculativeTargetWorker, VerifyRunner]:
    forward_output = SimpleNamespace(
        logits_output=SimpleNamespace(next_token_logits=torch.zeros(logits_rows, 4)),
        can_run_graph=False,
        expert_distribution_metrics=None,
        routed_experts_output=None,
        indexer_topk_output=None,
    )
    model_worker = SimpleNamespace(
        model_runner=SimpleNamespace(forward=lambda forward_batch: forward_output),
        model_config=None,
        device="cpu",
        random_seed=0,
    )
    target_worker = SpeculativeTargetWorker(model_worker, SpeculativeAlgorithm.DSPARK)
    runner = VerifyRunner(model_worker)
    target_worker.attach_runner(runner)
    target_worker.verify_requests = [
        SimpleNamespace(request_id="request-a"),
        SimpleNamespace(request_id="request-b"),
    ]
    return target_worker, runner


def verify_forward_batch(spec_info: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        spec_info=spec_info,
        kv_loc_plan=None,
        apply_deprecated_skip_attn_backend_init=lambda skip: None,
    )


def test_a_uniform_verify_hands_the_hooks_an_even_row_split() -> None:
    target_worker, runner = verify_target_worker(logits_rows=16)

    target_worker.forward_batch_generation(
        None, forward_batch=verify_forward_batch(SimpleNamespace()), is_verify=True
    )

    ((requests, rows_per_request),) = runner.calls
    assert [request.request_id for request in requests] == ["request-a", "request-b"]
    assert rows_per_request == 8


def test_a_compact_verify_hands_the_hooks_its_layout_s_row_counts() -> None:
    verify_lens = torch.tensor([3, 5], dtype=torch.int32)
    # Eight packed rows padded to a 13-row tier: no even split exists.
    target_worker, runner = verify_target_worker(logits_rows=13)
    spec_info = SimpleNamespace(
        ragged_verify_layout=SimpleNamespace(verify_lens=verify_lens)
    )

    target_worker.forward_batch_generation(
        None, forward_batch=verify_forward_batch(spec_info), is_verify=True
    )

    ((_, rows_per_request),) = runner.calls
    assert rows_per_request is verify_lens
