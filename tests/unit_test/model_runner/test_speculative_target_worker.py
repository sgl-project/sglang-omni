# SPDX-License-Identifier: Apache-2.0
"""Tests for the target worker SGLang's draft workers call."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.model_runner.speculative_target_worker import SpeculativeTargetWorker


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
    target_worker = SpeculativeTargetWorker(model_worker)
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
