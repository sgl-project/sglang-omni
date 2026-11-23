# SPDX-License-Identifier: Apache-2.0
"""Retained output penalties survive admission, launch and row reordering."""

from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.penalties import PenaltyStatusPool
from sglang.srt.sampling.sampler_params import SamplerParamsPool
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.sampling.sampling_pipeline import _adjust_before_watermark

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.sglang_execution import SGLangExecutionBridge
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


@pytest.fixture(autouse=True)
def runtime_config():
    with get_context().override_server_args():
        yield


def make_batch(rows):
    scheduler = object.__new__(OmniScheduler)
    scheduler.forward_ct = 0
    scheduler._sched_idled = (
        True  # noqa: leading-underscore  # exercise the existing runtime API
    )
    scheduler.processed_tokens_counter = 0
    scheduler.inflight_step_requests = {}
    scheduler.invariant_checker = Mock()
    scheduler.penalty_status_pool = PenaltyStatusPool(vocab_size=8, device="cpu")
    scheduler.sampler_params_pool = SamplerParamsPool(num_rows=len(rows), device="cpu")
    requests = []
    for index, (history, repetition, frequency, presence) in enumerate(rows):
        request = Req(
            rid=str(index),
            origin_input_text="",
            origin_input_ids=array("q", [1]),
            sampling_params=SamplingParams(
                temperature=0,
                repetition_penalty=repetition,
                frequency_penalty=frequency,
                presence_penalty=presence,
            ),
        )
        request.kv.req_pool_idx = index
        request.output_ids = array("q", history)
        requests.append(request)
    batch = ScheduleBatch(
        reqs=requests,
        device="cpu",
        forward_mode=ForwardMode.EXTEND,
        extend_num_tokens=len(requests),
        sampling_info=SamplingBatchInfo(vocab_size=8, device="cpu"),
    )
    scheduler.sampler_params_pool.write(requests)
    scheduler.penalty_status_pool.alloc(requests)
    scheduler.penalty_status_pool.reset_history(requests)
    return scheduler, batch


def forward(scheduler, batch):
    batch.req_pool_indices = torch.tensor(
        [request.kv.req_pool_idx for request in batch.reqs], dtype=torch.int64
    )
    batch.req_pool_indices_cpu = batch.req_pool_indices
    scheduler.stamp_batch_launch(batch)
    bridge = object.__new__(SGLangExecutionBridge)
    bridge.future_map = SimpleNamespace()
    runner = object.__new__(ModelRunner)
    runner.execution_bridge = bridge
    with patch("sglang.srt.managers.overlap_utils.resolve_forward_inputs"):
        with runner.execution_context(batch, isolate_sampling=True):
            snapshot = batch.sampling_info
    batch.drop_inflight_step()
    scheduler.inflight_step_requests.pop(batch.forward_iter)
    return snapshot


def make_logits(snapshot):
    logits = torch.tensor([[2.6, -2, 2.6, -2, 2.6, -2, 2.6, -2]]).repeat(
        snapshot.temperatures.shape[0], 1
    )
    with torch._dynamo.config.patch(
        disable=True
    ):  # noqa: leading-underscore  # exercise the existing runtime API
        _adjust_before_watermark(logits, snapshot, observer=None)
    return logits


def test_mixed_restore_and_forward_snapshot_isolation():
    scheduler, batch = make_batch(
        [
            ([2, 2, 5], 1.3, 0.25, -0.5),
            ([2, 5, 5], 2.0, 0.5, 0.25),
            ([2, 5], 1.0, 0.0, 0.0),
            ([], 1.1, -0.25, 0.5),
        ]
    )
    snapshot = forward(scheduler, batch)
    expected = torch.tensor([[2.6, -2, 2.6, -2, 2.6, -2, 2.6, -2]]).repeat(4, 1)
    expected[0, 2], expected[0, 5] = 2.0, -2.275
    expected[1, 2], expected[1, 5] = 0.925, -6.5
    torch.testing.assert_close(make_logits(snapshot), expected)
    torch.testing.assert_close(make_logits(forward(scheduler, batch)), expected)

    snapshot.step_penalties.commit(torch.tensor([3, 3, 3, 3]))
    changed = expected.clone()
    changed[:, 3] = torch.tensor([-2.275, -5.5, -2.0, -2.475])
    torch.testing.assert_close(make_logits(forward(scheduler, batch)), changed)
    torch.testing.assert_close(make_logits(snapshot), expected)


def test_chunked_reprefill_rebuild_then_one_committed_decode():
    scheduler, batch = make_batch([([2, 2, 5], 1.3, 0.25, -0.5)])
    for _ in range(3):
        scheduler.penalty_status_pool.reset_history(batch.reqs)
        snapshot = forward(scheduler, batch)
        torch.testing.assert_close(
            make_logits(snapshot)[0, [2, 5]], torch.tensor([2.0, -2.275])
        )
    batch.reqs[0].output_ids.append(2)
    snapshot.step_penalties.commit(torch.tensor([2]))
    batch.forward_mode = ForwardMode.DECODE
    torch.testing.assert_close(
        make_logits(forward(scheduler, batch))[0, [2, 5]],
        torch.tensor([2.35 / 1.3, -2.275]),
    )


def test_sampling_batch_filter_merge_preserves_row_behavior():
    scheduler, batch = make_batch(
        [
            ([2, 2], 1.3, 0.25, -0.5),
            ([5], 2.0, 0.5, 0.25),
            ([3, 3, 3], 1.0, 0.0, 0.0),
            ([5, 5], 1.3, 0.25, -0.5),
        ]
    )
    make_logits(forward(scheduler, batch))
    batch.reqs = batch.reqs[1:]
    expected = torch.tensor([[2.6, -2, 2.6, -2, 2.6, -2, 2.6, -2]]).repeat(3, 1)
    expected[0, 5], expected[2, 5] = -5.5, -2.6
    torch.testing.assert_close(make_logits(forward(scheduler, batch)), expected)
