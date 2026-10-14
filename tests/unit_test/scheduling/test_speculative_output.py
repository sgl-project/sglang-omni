# SPDX-License-Identifier: Apache-2.0
"""Committed speculative output stays attached to its request across row changes."""

from __future__ import annotations

import queue
import threading
from array import array
from collections import deque
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req, ScheduleBatch
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.qwen3_omni.request_builders import (
    make_thinker_stream_output_builder,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


def make_request(request_id: str, pool_index: int, maximum_tokens: int) -> Req:
    sampling = SamplingParams(temperature=0, max_new_tokens=maximum_tokens)
    sampling.normalize(None)
    request = Req(
        request_id,
        "prompt",
        array("q", [1, 2]),
        sampling,
        stream=True,
        eos_token_ids={99},
    )
    request.kv.req_pool_idx = pool_index
    request.kv.kv_committed_len = 2
    request._omni_terminal_claimed = (
        False  # noqa: leading-underscore  # Shared request ownership field.
    )
    request.omni_data = SGLangARRequestData(
        req=request,
        stage_payload=StagePayload(
            request_id=request_id,
            request=OmniRequest(
                inputs="prompt",
                params={"stream": True},
                metadata={"output_modalities": ["text"]},
            ),
            data={},
        ),
    )
    return request


def make_batch(requests: list[Req]) -> ScheduleBatch:
    batch = ScheduleBatch.__new__(ScheduleBatch)
    batch.reqs = requests
    batch.forward_mode = ForwardMode.DECODE
    batch.has_grammar = False
    return batch


def make_scheduler(batch: ScheduleBatch) -> OmniScheduler:
    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler.request_admission_lock = threading.RLock()
    scheduler.aborted_request_ids = set()
    scheduler.aborted_request_id_order = deque()
    scheduler.running_batch = batch
    scheduler.cur_batch = batch
    scheduler.last_batch = None
    scheduler.waiting_queue = []
    scheduler.async_pending_batch = Mock(return_value=None)
    scheduler.stream_output_builder = make_thinker_stream_output_builder(
        speech_enabled=True
    )
    scheduler.outbox = queue.Queue()
    scheduler.first_emit_done = set()
    scheduler.scheduler_thread_id = threading.get_ident()
    scheduler.tp_size = 1
    scheduler.pending_request_builds = {}
    scheduler.pending_request_admissions = {}
    scheduler.backlogged_request_build_payloads = deque()
    scheduler.pending_stream_ingress = {}
    scheduler.deferred_request_payloads = {}
    scheduler.dirty_deferred_request_ids = set()
    scheduler.prefill_start_done = set()
    scheduler.prefill_end_done = set()
    scheduler.emit_model_path_end_once = Mock()
    scheduler.drain_inbox_for_request = Mock()
    scheduler.run_abort_callback = Mock()
    scheduler.release_request_kv_cache = Mock()
    return scheduler


def resolve_tokens(
    batch: ScheduleBatch, token_rows: list[list[int]], accepted_lengths: list[int]
) -> list[list[int]]:
    processor = SchedulerBatchResultProcessor.__new__(SchedulerBatchResultProcessor)
    object.__setattr__(processor, "model_worker", Mock())
    result = GenerationBatchResult(
        next_token_ids=torch.tensor(token_rows).flatten(),
        accept_lens=torch.tensor(accepted_lengths),
        speculative_num_draft_tokens=8,
    )
    return processor._resolve_spec_v2_tokens(
        result, batch
    )  # noqa: leading-underscore  # Native result processor ownership boundary.


def drain_messages(scheduler: OmniScheduler) -> dict[str, list[int]]:
    outputs: dict[str, list[int]] = {}
    while not scheduler.outbox.empty():
        message = scheduler.outbox.get_nowait()
        assert message.target == "decode"
        outputs.setdefault(message.request_id, []).extend(message.data.tolist())
    return outputs


@pytest.mark.parametrize("maximum_tokens", [1, 2, 7, 8, 9])
@pytest.mark.parametrize("eos_index", [0, 3, 7])
def test_batched_resolver_stop_and_cap_publish_only_committed_prefix(
    maximum_tokens: int,
    eos_index: int,
) -> None:
    requests = [
        make_request("capped", 7, maximum_tokens),
        make_request("eos", 2, 20),
        make_request("continuing", 12, 20),
    ]
    batch = make_batch(requests)
    scheduler = make_scheduler(batch)
    token_rows = [list(range(10, 18)), list(range(20, 28)), list(range(30, 38))]
    token_rows[1][eos_index] = 99
    resolved = resolve_tokens(batch, token_rows, [8, 8, 3])
    for request, tokens in zip(requests, resolved, strict=True):
        request.output_ids.extend(tokens)
        request.update_finish_state(len(tokens))
        scheduler.emit_speculative_tokens(request)
        scheduler.emit_speculative_tokens(request)
    expected = {
        "capped": token_rows[0][:maximum_tokens],
        "eos": token_rows[1][: eos_index + 1],
        "continuing": [30, 31, 32],
    }
    assert drain_messages(scheduler) == expected
    for request in requests:
        assert list(request.output_ids_through_stop) == expected[request.rid]
        assert request.omni_data.speculative_emitted_tokens == len(
            expected[request.rid]
        )
    assert requests[2].finished() is False


def test_running_abort_defers_release_and_reused_slot_has_fresh_output_owner() -> None:
    requests = [
        make_request("cancelled", 7, 20),
        make_request("continuing", 2, 20),
        make_request("capped", 12, 1),
    ]
    batch = make_batch(requests)
    scheduler = make_scheduler(batch)
    old_request_data = requests[0].omni_data
    scheduler.abort("cancelled")
    scheduler.release_request_kv_cache.assert_not_called()
    scheduler.run_abort_callback.assert_not_called()
    assert requests[0].kv.req_pool_idx == 7
    assert requests[0].omni_data is old_request_data
    assert isinstance(requests[0].to_finish, FINISH_ABORT)
    resolved = resolve_tokens(
        batch,
        [list(range(10, 18)), list(range(20, 28)), list(range(30, 38))],
        [3, 2, 4],
    )
    for request, tokens in zip(requests, resolved, strict=True):
        request.output_ids.extend(tokens)
        request.update_finish_state(len(tokens))
        scheduler.emit_speculative_tokens(request)
    assert drain_messages(scheduler) == {"continuing": [20, 21], "capped": [30]}
    assert isinstance(requests[0].finished_reason, FINISH_ABORT)
    replacement = make_request("replacement", 7, 20)
    next_batch = make_batch([requests[1], replacement])
    scheduler.running_batch = next_batch
    scheduler.cur_batch = next_batch
    scheduler.last_batch = batch
    next_tokens = resolve_tokens(
        next_batch, [list(range(40, 48)), list(range(50, 58))], [3, 1]
    )
    for request, tokens in zip(next_batch.reqs, next_tokens, strict=True):
        request.output_ids.extend(tokens)
        request.update_finish_state(len(tokens))
        scheduler.emit_speculative_tokens(request)
    scheduler.emit_speculative_tokens(requests[0])
    assert drain_messages(scheduler) == {
        "continuing": [40, 41, 42],
        "replacement": [50],
    }
    assert list(requests[1].output_ids) == [20, 21, 40, 41, 42]
    assert replacement.omni_data.speculative_emitted_tokens == 1
    assert old_request_data.speculative_emitted_tokens == 0


@pytest.mark.parametrize(
    "suppression", ["middle_chunk", "terminal_claimed", "finished_abort"]
)
def test_nonpublishable_request_state_does_not_advance_stream_cursor(
    suppression: str,
) -> None:
    request = make_request("request", 7, 20)
    request.output_ids.extend([10, 11])
    scheduler = make_scheduler(make_batch([request]))
    if suppression == "middle_chunk":
        request.inflight_middle_chunks = 1
    elif suppression == "terminal_claimed":
        request._omni_terminal_claimed = (
            True  # noqa: leading-underscore  # Shared request ownership field.
        )
    else:
        request.finished_reason = FINISH_ABORT()
    scheduler.emit_speculative_tokens(request)
    assert drain_messages(scheduler) == {}
    assert request.omni_data.speculative_emitted_tokens == 0


def test_cap_above_draft_width_finishes_only_in_second_cycle() -> None:
    request = make_request("cap-nine", 7, 9)
    batch = make_batch([request])
    scheduler = make_scheduler(batch)
    first_tokens = resolve_tokens(batch, [list(range(10, 18))], [8])[0]
    request.output_ids.extend(first_tokens)
    request.update_finish_state(len(first_tokens))
    scheduler.emit_speculative_tokens(request)
    assert request.finished() is False
    assert drain_messages(scheduler) == {"cap-nine": list(range(10, 18))}
    second_tokens = resolve_tokens(batch, [list(range(20, 28))], [4])[0]
    request.output_ids.extend(second_tokens)
    request.update_finish_state(len(second_tokens))
    scheduler.emit_speculative_tokens(request)
    assert request.finished() is True
    assert drain_messages(scheduler) == {"cap-nine": [20]}
    assert list(request.output_ids_through_stop) == list(range(10, 18)) + [20]
    assert request.omni_data.speculative_emitted_tokens == 9


def test_explicit_stop_token_crops_one_row_without_stopping_its_neighbor() -> None:
    request = make_request("stop-token", 7, 20)
    request.sampling_params.stop_token_ids = {12}
    neighbor = make_request("continuing", 2, 20)
    batch = make_batch([request, neighbor])
    scheduler = make_scheduler(batch)
    resolved = resolve_tokens(batch, [list(range(10, 18)), list(range(20, 28))], [8, 4])
    for current_request, tokens in zip(batch.reqs, resolved, strict=True):
        current_request.output_ids.extend(tokens)
        current_request.update_finish_state(len(tokens))
        scheduler.emit_speculative_tokens(current_request)
    assert request.finished() is True
    assert neighbor.finished() is False
    assert drain_messages(scheduler) == {
        "stop-token": [10, 11, 12],
        "continuing": [20, 21, 22, 23],
    }
