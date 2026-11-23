# SPDX-License-Identifier: Apache-2.0
"""Real main request ownership and exit accounting across Omni terminal paths."""

import queue
import threading
from array import array
from collections import deque
from unittest.mock import Mock

import pytest
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    KeyedReqList,
    ListKind,
    move,
)
from sglang.srt.managers.scheduler_components.request_retirer import RequestRetirer
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.penalties import PenaltyStatusPool
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.omni_scheduler import (
    NoOpGrammarManager,
    NoOpSender,
    OmniIpcChannels,
    OmniScheduler,
)
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


@pytest.fixture
def scheduler():
    with (
        get_context().override_server_args(),
        envs.SGLANG_ENABLE_RUNNING_REQS_CHECK.override(True),
    ):
        scheduler = object.__new__(OmniScheduler)
        scheduler.waiting_queue = IndexedReqList(ListKind.WAITING)
        scheduler.chunked_reqs = IndexedReqList(ListKind.CHUNKED)
        scheduler.running_reqs = KeyedReqList(ListKind.DECODING)
        scheduler.prefill_launched_reqs = KeyedReqList(ListKind.PREFILL_LAUNCHED)
        scheduler.grammar_manager = NoOpGrammarManager()
        scheduler.disagg_prefill_transfer_queue = IndexedReqList(ListKind.PD_SENDING)
        for name in (
            "disagg_prefill_bootstrap_queue",
            "disagg_decode_prealloc_queue",
            "disagg_decode_transfer_queue",
            "hisparse_coordinator",
            "uocr_runtime",
            "deferred_responses",
            "decode_offload_manager",
            "dllm_config",
            "session_bridge",
            "cur_batch",
            "last_batch",
            "async_pending",
            "previous_pending_decode",
            "abort_callback",
            "draft_worker",
            "request_finished_callback",
            "_pending_chunked_abort_req",
        ):
            setattr(scheduler, name, None)
        scheduler.inbox = queue.Queue()
        scheduler.outbox = queue.Queue()
        scheduler.running_batch = ScheduleBatch(reqs=[])
        scheduler.decode_batch_states = {}
        scheduler.decode_path = "plain"
        scheduler.enable_plain_fallback = False
        scheduler.enable_overlap = False
        scheduler.async_prefill = False
        scheduler.early_request_builds = False
        scheduler.scheduler_thread_id = threading.get_ident()
        scheduler.is_entry_rank = True
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.send_to_detokenizer = NoOpSender()
        scheduler.ipc_channels = OmniIpcChannels(scheduler)
        scheduler.completed_request_ids = {}
        scheduler.tp_size = 1
        scheduler.request_admission_lock = threading.RLock()
        for name in (
            "aborted_request_ids",
            "dirty_deferred_request_ids",
            "first_emit_done",
            "prefill_start_done",
            "prefill_end_done",
        ):
            setattr(scheduler, name, set())
        for name in (
            "pending_stream_ingress",
            "deferred_request_payloads",
            "pending_request_builds",
            "pending_request_admissions",
            "inflight_step_requests",
        ):
            setattr(scheduler, name, {})
        for name in (
            "aborted_request_id_order",
            "backlogged_request_build_payloads",
            "result_queue",
        ):
            setattr(scheduler, name, deque())
        scheduler.poll_request_timeout_aborts = lambda: ()
        scheduler.recv_scheduler_messages = lambda: [scheduler.inbox.get_nowait()]
        scheduler.tree_cache = Mock()
        scheduler.penalty_status_pool = PenaltyStatusPool(vocab_size=16, device="cpu")
        scheduler.invariant_checker = SchedulerInvariantChecker(
            disaggregation_mode=DisaggregationMode.NULL,
            page_size=128,
            tree_cache=scheduler.tree_cache,
            token_to_kv_pool_allocator=None,
            req_to_token_pool=None,
            pool_stats_observer=None,
            scheduler_stage_metrics=None,
            get_request_lists=scheduler.request_lists,
        )
        scheduler.request_retirer = RequestRetirer(
            tree_cache=scheduler.tree_cache,
            beam_coordinator=Mock(),
            hisparse_coordinator=None,
            uocr_runtime=None,
            penalty_status_pool=scheduler.penalty_status_pool,
            send_abort_output=scheduler.send_abort_output,
            stream_output=scheduler.stream_output,
        )
        yield scheduler


def add_request(scheduler, owner):
    request = Req(
        rid="request",
        origin_input_text="",
        origin_input_ids=array("q", [1]),
        sampling_params=SamplingParams(temperature=0, repetition_penalty=1.2),
    )
    request.omni_data = SGLangARRequestData(req=request)
    request._omni_terminal_claimed = (
        False  # noqa: leading-underscore  # exercise the existing runtime API
    )
    move([request], owner)
    scheduler.penalty_status_pool.alloc([request])
    scheduler.invariant_checker.expect_output(request)
    return request


def receive_cancel(scheduler, request):
    scheduler.inbox.put(IncomingMessage(request_id=request.rid, type="abort"))
    assert scheduler.recv_requests() == []


def assert_released(scheduler, request):
    assert request.owner_list is None
    assert request.penalty_index is None
    assert request.exit_via == "release"
    scheduler.invariant_checker._check_ended_requests()  # noqa: leading-underscore  # exercise the existing runtime API
    assert not scheduler.invariant_checker.unended
    assert not scheduler.invariant_checker.cancelled_outputs


def test_queued_client_cancel_releases_without_sending_a_result(scheduler):
    request = add_request(scheduler, scheduler.waiting_queue)
    receive_cancel(scheduler, request)
    assert_released(scheduler, request)
    assert request.send_via is None
    assert scheduler.outbox.empty()
    receive_cancel(scheduler, request)
    assert scheduler.outbox.empty()


def test_running_client_cancel_defers_release_until_the_step_finishes(scheduler):
    request = add_request(scheduler, scheduler.running_reqs)
    batch = ScheduleBatch(reqs=[request], forward_mode=ForwardMode.DECODE)
    scheduler.running_batch = batch
    batch.add_inflight_step()
    receive_cancel(scheduler, request)
    assert request.owner_list is scheduler.running_reqs
    assert request.exit_via is None
    assert request.penalty_index is not None
    # The result processor ends the request and calls the real retirer/output boundary.
    request.update_finish_state()
    scheduler.request_retirer.release(request, checkpoint=False)
    scheduler.stream_output([request])
    batch.drop_inflight_step()
    assert_released(scheduler, request)
    assert request.inflight_steps == 0
    assert request.send_via is None
    assert scheduler.outbox.empty()


def test_model_failure_sends_once_releases_and_balances_the_launched_step(scheduler):
    request = add_request(scheduler, scheduler.running_reqs)
    batch = ScheduleBatch(
        reqs=[request], forward_mode=ForwardMode.DECODE, forward_iter=1
    )
    scheduler.running_batch = batch
    batch.add_inflight_step()
    scheduler.inflight_step_requests[1] = batch.launched_reqs
    scheduler.handle_batch_failure(batch, RuntimeError("device failed"))
    assert_released(scheduler, request)
    assert request.inflight_steps == 0
    assert not scheduler.inflight_step_requests
    message = scheduler.outbox.get_nowait()
    assert message.type == "error"
    assert str(message.data) == "device failed"
    assert request.send_via == "omni_error"
    assert scheduler.outbox.empty()
    scheduler.finish_failed_step(batch)
    receive_cancel(scheduler, request)
    assert request.inflight_steps == 0
    assert scheduler.outbox.empty()


def test_chunked_client_cancel_releases_at_the_next_scheduling_boundary(scheduler):
    request = add_request(scheduler, scheduler.chunked_reqs)
    batch = ScheduleBatch(
        reqs=[request], chunked_req=request, forward_mode=ForwardMode.EXTEND
    )
    batch.add_inflight_step()
    receive_cancel(scheduler, request)
    assert (
        scheduler._pending_chunked_abort_req is request
    )  # noqa: leading-underscore  # exercise the existing runtime API
    assert request.exit_via is None
    scheduler.process_pending_chunked_abort()
    assert scheduler.chunked_req is None
    assert not scheduler.still_in_batch(request)
    assert_released(scheduler, request)
    batch.drop_inflight_step()
    assert request.inflight_steps == 0
    assert request.send_via is None
    assert scheduler.outbox.empty()


def test_resolve_keeps_all_launched_rows_and_skips_a_retracted_owner(scheduler):
    request = add_request(scheduler, scheduler.running_reqs)
    batch = ScheduleBatch(
        reqs=[request], forward_mode=ForwardMode.DECODE, forward_iter=1
    )
    batch.add_inflight_step()
    move([request], scheduler.waiting_queue)
    result = object()
    scheduler.run_batch_resolve = Mock(return_value=result)
    scheduler.process_batch_result = Mock()
    scheduler.resolve_and_process(batch, "scheduled", "pending")
    scheduler.run_batch_resolve.assert_called_once_with(
        batch, "scheduled", "pending", skip_rids={request.rid}
    )
    scheduler.process_batch_result.assert_called_once_with(batch, result)
    assert batch.reqs == list(batch.launched_reqs) == [request]
