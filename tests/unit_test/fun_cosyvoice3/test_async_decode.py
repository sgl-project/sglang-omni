# SPDX-License-Identifier: Apache-2.0
"""CPU checks for Cosy's asynchronous sampling and delayed token collection."""

from __future__ import annotations

from array import array
from queue import Queue
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.managers import overlap_utils
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    Req,
    ScheduleBatch,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.penaltylib import (
    BatchedMinNewTokensPenalizer,
    BatchedPenalizerOrchestrator,
    BatchedRepetitionPenalizer,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from sglang_omni.model_runner.sglang_execution import SGLangExecutionBridge
from sglang_omni.models.fun_cosyvoice3.model_runner import FunCosyVoice3ModelRunner
from sglang_omni.models.fun_cosyvoice3.request_builders import (
    CosyVoice3SGLangRequestData,
)
from sglang_omni.scheduling.types import SchedulerRequest

_VOCAB_SIZE = 16
_STOP_TOKEN = _VOCAB_SIZE - 1


def _make_batch(initial_tokens: list[int]) -> ScheduleBatch:
    reqs = []
    for index, token in enumerate(initial_tokens):
        req = Req(
            rid=f"req-{index}",
            origin_input_text="",
            origin_input_ids=array("q", [0]),
            sampling_params=SamplingParams(repetition_penalty=1.21, min_new_tokens=3),
            eos_token_ids={_STOP_TOKEN},
        )
        req.tokenizer = SimpleNamespace(
            eos_token_id=_STOP_TOKEN, additional_stop_token_ids=set()
        )
        req.output_ids.append(token)
        reqs.append(req)
    count = len(reqs)
    batch = ScheduleBatch(
        reqs=reqs,
        device="cpu",
        forward_mode=ForwardMode.DECODE,
        req_pool_indices=torch.arange(1, count + 1),
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.sampling_info = SamplingBatchInfo(
        temperatures=torch.ones(count, 1),
        top_ps=torch.ones(count),
        top_ks=torch.ones(count, dtype=torch.int32),
        min_ps=torch.zeros(count),
        is_all_greedy=True,
        is_any_greedy=True,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=_VOCAB_SIZE,
        device="cpu",
        penalizer_orchestrator=BatchedPenalizerOrchestrator(
            vocab_size=_VOCAB_SIZE,
            batch=batch,
            penalizers={BatchedRepetitionPenalizer, BatchedMinNewTokensPenalizer},
        ),
    )
    return batch


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> FunCosyVoice3ModelRunner:
    # note (ql): Keep the real relay on CPU even when the host supports CUDA.
    monkeypatch.setattr(overlap_utils, "_is_cuda", False)
    future_map = FutureMap(
        device=torch.device("cpu"),
        spec_algo=SpeculativeAlgorithm.NONE,
        req_to_token_pool=SimpleNamespace(req_to_token=torch.empty(8, 1)),
        needs_cpu_seq_lens=False,
    )
    bridge = SGLangExecutionBridge(
        device=torch.device("cpu"),
        worker=SimpleNamespace(model_runner=SimpleNamespace()),
        spec_algorithm=SpeculativeAlgorithm.NONE,
        future_map=future_map,
    )
    model_runner = object.__new__(FunCosyVoice3ModelRunner)
    model_runner.device = torch.device("cpu")
    model_runner._async_enabled = True
    model_runner.bind_execution_bridge(bridge)
    return model_runner


def _snapshot(
    runner: FunCosyVoice3ModelRunner,
    batch: ScheduleBatch,
    actual_tokens: list[int],
) -> SamplingBatchInfo:
    original = batch.sampling_info
    runner._execution_bridge.publish_next_tokens(batch, torch.tensor(actual_tokens))
    batch.cumulate_penalty_output_tokens()
    with runner._execution_context(batch, isolate_sampling=True):
        assert batch.input_ids.tolist() == actual_tokens
        snapshot = batch.sampling_info
        assert snapshot.penalizer_orchestrator is None
    assert batch.sampling_info is original
    return snapshot


def test_decode_history_uses_resolved_tokens_and_advances_min_length_once(
    runner: FunCosyVoice3ModelRunner,
) -> None:
    batch = _make_batch([1, 5])
    history: list[list[int]] = [[], []]
    saved: list[tuple[SamplingBatchInfo, torch.Tensor, torch.Tensor]] = []
    for step, actual_tokens in enumerate(([1, 5], [2, 6], [3, 7]), start=1):
        host_tokens = [req.output_ids[-1] for req in batch.reqs]
        if step > 1:
            assert host_tokens != actual_tokens
        snapshot = _snapshot(runner, batch, actual_tokens)
        expected_scaling = torch.ones(2, _VOCAB_SIZE)
        for row, token in enumerate(actual_tokens):
            history[row].append(token)
            expected_scaling[row, history[row]] = 1.21
        expected_additive = torch.zeros(2, _VOCAB_SIZE)
        if step < 3:
            expected_additive[:, _STOP_TOKEN] = -torch.inf
        torch.testing.assert_close(snapshot.acc_scaling_penalties, expected_scaling)
        torch.testing.assert_close(snapshot.acc_additive_penalties, expected_additive)
        minimum = batch.sampling_info.penalizer_orchestrator.penalizers[
            BatchedMinNewTokensPenalizer
        ]
        assert minimum.len_output_tokens.tolist() == [[step], [step]]
        saved.append((snapshot, expected_scaling, expected_additive))
        for previous, scaling, additive in saved:
            torch.testing.assert_close(previous.acc_scaling_penalties, scaling)
            torch.testing.assert_close(previous.acc_additive_penalties, additive)
        if step > 1:
            # note (ql): Resolve the previous launch only after this launch's snapshot.
            for req, token in zip(batch.reqs, actual_tokens):
                req.output_ids.append(token)


def test_decode_history_follows_filtered_and_reordered_pool_rows(
    runner: FunCosyVoice3ModelRunner,
) -> None:
    batch = _make_batch([1, 5, 9])
    _snapshot(runner, batch, [1, 5, 9])
    _snapshot(runner, batch, [2, 6, 10])

    keep = [2, 0]
    indices = torch.tensor(keep)
    batch.reqs = [batch.reqs[index] for index in keep]
    batch.req_pool_indices = batch.req_pool_indices[indices]
    batch.sampling_info.filter_batch(keep, indices)
    snapshot = _snapshot(runner, batch, [11, 3])

    expected = torch.ones(2, _VOCAB_SIZE)
    expected[0, [9, 10, 11]] = 1.21
    expected[1, [1, 2, 3]] = 1.21
    torch.testing.assert_close(snapshot.acc_scaling_penalties, expected)


def test_resolve_skips_inactive_requests_without_compacting_token_rows(
    runner: FunCosyVoice3ModelRunner,
) -> None:
    batch = _make_batch([1, 2, 3, 4, 5])
    batch.reqs[0].finished_reason = FINISH_LENGTH(1)
    batch.reqs[2].is_retracted = True
    batch.reqs[3].to_finish = FINISH_ABORT()
    requests = [
        SchedulerRequest(
            request_id=req.rid,
            data=CosyVoice3SGLangRequestData(
                req=req,
                stream_metadata={},
                stream_code_next_flush=1,
                stream_prompt_sent=True,
            ),
        )
        for req in batch.reqs
    ]
    runner._outbox = Queue()
    runner._vocoder_target = "vocoder"
    runner._ar_followup_flush_tokens = 1
    launch_buf = torch.tensor([11, 22, 33, 44, 55])
    result = SimpleNamespace(next_token_ids=torch.full((5,), -1))

    runner.post_decode_resolve(launch_buf, result, None, batch, requests)

    assert [[token.item() for token in req.data.output_codes] for req in requests] == [
        [],
        [22],
        [],
        [],
        [55],
    ]
    assert [req.data.stream_code_seen for req in requests] == [0, 1, 0, 0, 1]
    messages = [runner._outbox.get_nowait(), runner._outbox.get_nowait()]
    assert runner._outbox.empty()
    assert [message.request_id for message in messages] == ["req-1", "req-4"]
    assert [message.data.tolist() for message in messages] == [[22], [55]]


@pytest.mark.parametrize(
    "field,value",
    [
        ("frequency_penalty", 0.25),
        ("presence_penalty", 0.25),
        ("custom_logit_processor", "custom-processor"),
        ("grammar", object()),
    ],
)
def test_lookahead_rejects_unsupported_history(
    runner: FunCosyVoice3ModelRunner,
    field: str,
    value: object,
) -> None:
    batch = _make_batch([1, 5])
    assert not runner.lookahead_eligible(batch)
    runner.device = torch.device("cuda")
    assert runner.lookahead_eligible(batch)
    req = batch.reqs[1]
    target = req.sampling_params if field.endswith("penalty") else req
    setattr(target, field, value)
    assert not runner.lookahead_eligible(batch)
