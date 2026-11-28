# SPDX-License-Identifier: Apache-2.0
"""Scheduler roles for explicit Prefill/Decode pipeline stages."""

from __future__ import annotations

import logging
import queue
import threading
import types
from collections import deque
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from uuid import uuid4

import torch
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler import Scheduler as UpstreamScheduler
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.allocator.swa import (
    PureSWATokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

from sglang_omni.admission import AdminDroppedRequestError
from sglang_omni.comm.kv_transfer import DEFAULT_KV_POOL_ID, KVPageTransfer
from sglang_omni.scheduling.message import (
    IncomingMessage,
    OutgoingMessage,
    put_messages,
)
from sglang_omni.scheduling.omni_scheduler import AdminActionResult, OmniScheduler
from sglang_omni.scheduling.pd_receiver import (
    DEFAULT_DECODE_RUNWAY_TOKENS,
    DecodeAdmission,
    DecodeKVReceiver,
)
from sglang_omni.scheduling.pd_utils import (
    SGLangKVLease,
    StateBuilder,
    StateRestorer,
    build_kv_pool,
    continuation_from_req,
    defer_first_token_finish,
    request_from_continuation,
    request_page_indices,
    serialize_kv_allocator,
)
from sglang_omni.scheduling.types import RequestOutput

logger = logging.getLogger(__name__)

DEFAULT_MAX_IN_FLIGHT_HANDOFFS = 32


class PrefillChunkReserveAdder(PrefillAdder):
    """PrefillAdder that guards chunked requests against in-flight PD leases."""

    has_in_flight_leases: bool = False
    is_chunked_request_parked: bool = False

    def budget_state(self) -> AddReqResult:
        if self.is_chunked_request_parked:
            return AddReqResult.OTHER
        else:
            return super().budget_state()

    def reserve_unscheduled_tail(self, request: Req) -> None:
        if request.extend_end is None:
            return
        else:
            pass
        total_input_tokens = len(request.full_untruncated_fill_ids)
        if total_input_tokens == 0:
            total_input_tokens = len(request.origin_input_ids)
        else:
            pass
        unscheduled_tokens = max(total_input_tokens - request.extend_end, 0)
        if unscheduled_tokens > 0:
            reserved_tokens = self.ceil_paged_tokens(unscheduled_tokens)
            self.memory_budget.total_offset += reserved_tokens
            self.memory_budget.current_offset += reserved_tokens
        else:
            pass

    def add_chunked_req(self, req: Req) -> Req | None:
        if self.has_in_flight_leases:
            remaining_current = self.memory_budget.remaining_current
            if remaining_current < self.page_size:
                # note (YzXiao): in-flight PD leases temporarily pin KV; park the chunked request until leases drain instead of failing prefill allocation.
                self.rem_chunk_tokens = 0
                self.is_chunked_request_parked = True
                return req
            elif (
                self.rem_chunk_tokens is not None
                and remaining_current < self.rem_chunk_tokens
            ):
                self.rem_chunk_tokens = (
                    remaining_current // self.page_size
                ) * self.page_size
            else:
                pass
        else:
            pass

        remaining_chunked = super().add_chunked_req(req)
        if remaining_chunked is not None:
            if req in self.can_run_list:
                self.reserve_unscheduled_tail(req)
            else:
                self.is_chunked_request_parked = True
        else:
            pass
        return remaining_chunked

    def add_one_req(
        self,
        req: Req,
        has_chunked_req: bool,
        truncation_align_size: int | None,
    ) -> AddReqResult:
        result = super().add_one_req(req, has_chunked_req, truncation_align_size)
        if self.new_chunked_req is req:
            self.reserve_unscheduled_tail(req)
            return self.budget_state()
        else:
            return result


class LeasedPrefillChunkReserveAdder(PrefillChunkReserveAdder):
    """PrefillChunkReserveAdder variant active while PD KV leases are in flight."""

    has_in_flight_leases: bool = True


class PDKVLifecycle(OmniScheduler):
    """Own PD-only KV state that upstream scheduler lifecycle checks cannot see."""

    def __init__(self, *args, **kwargs) -> None:
        self.pd_due_releases: queue.SimpleQueue[Req] = queue.SimpleQueue()
        self.pd_outstanding_releases: set[str] = set()
        self.pd_wakeup_lock = threading.Lock()
        self.pd_wakeup_pending = False
        super().__init__(*args, **kwargs)

    def notify_pd_wakeup(self) -> None:
        """Coalesce off-thread PD release or admission events into one inbox wakeup."""
        with self.pd_wakeup_lock:
            if self.pd_wakeup_pending:
                return
            else:
                pass
            self.pd_wakeup_pending = True
        self.inbox.put(IncomingMessage(request_id="", type="wakeup"))

    def reset_pd_wakeup(self) -> None:
        with self.pd_wakeup_lock:
            self.pd_wakeup_pending = False

    def pd_holds_kv(self) -> bool:
        return bool(self.pd_outstanding_releases)

    def pd_lifecycle_guard(self) -> AbstractContextManager[None]:
        return nullcontext()

    def lease_pd_kv(self, req: Req) -> SGLangKVLease:
        lease = SGLangKVLease(req, self.pd_due_releases, wakeup_notifier=self)
        self.pd_outstanding_releases.add(req.rid)
        return lease

    def defer_pd_kv_release(self, req: Req) -> None:
        self.pd_outstanding_releases.add(req.rid)
        self.pd_due_releases.put(req)
        self.notify_pd_wakeup()

    def is_fully_idle(
        self, for_health_check: bool = False, ignore_waiting: bool = False
    ) -> bool:
        # Health checks only care whether a running request can carry their
        # result. Destructive operations must also see PD-owned KV.
        if not for_health_check and self.pd_holds_kv():
            return False
        else:
            pass
        return UpstreamScheduler.is_fully_idle(
            self, for_health_check=for_health_check, ignore_waiting=ignore_waiting
        )

    def drain_due_releases(self) -> None:
        while True:
            try:
                req = self.pd_due_releases.get_nowait()
            except queue.Empty:
                return
            try:
                self.release_request_kv_cache(req)
            except Exception:
                # One bad request must not strand the rest of the queue.
                logger.exception(f"PD release failed for {req.rid!r}")
            else:
                self.pd_outstanding_releases.discard(req.rid)

    def run_weight_update_with_lifecycle(
        self,
        payload: dict[str, object],
        update_fn,
        result_data: Mapping[str, object],
        *,
        keep_pause_on_failure: bool = False,
    ) -> AdminActionResult:
        def update_after_pd_drains(
            update_payload: dict[str, object],
        ) -> tuple[bool, str]:
            self.drain_due_releases()
            if self.pd_holds_kv():
                return False, "PD-owned KV is still in flight"
            else:
                pass
            return update_fn(update_payload)

        with self.pd_lifecycle_guard():
            return super().run_weight_update_with_lifecycle(
                payload,
                update_after_pd_drains,
                result_data,
                keep_pause_on_failure=keep_pause_on_failure,
            )

    def flush_cache(self, empty_cache: bool = True) -> bool:
        with self.pd_lifecycle_guard():
            # Upstream clears both pools once it reads the scheduler as idle.
            self.drain_due_releases()
            return bool(UpstreamScheduler.flush_cache(self, empty_cache=empty_cache))

    def admin_model_info(self) -> AdminActionResult:
        self.drain_due_releases()
        result = super().admin_model_info()
        data = result.get("data")
        if result.get("success") and isinstance(data, dict):
            data["pd_outstanding_releases"] = len(self.pd_outstanding_releases)
        else:
            pass
        return result


class OmniPrefillScheduler(PDKVLifecycle):
    """Omni scheduler whose generated requests stop after Prefill."""

    scheduler_role = "prefill"
    max_in_flight_handoffs: int = DEFAULT_MAX_IN_FLIGHT_HANDOFFS
    chunked_req: Req | None = None

    def __init__(
        self,
        *args,
        stage_name: str,
        partner_stage: str | None = None,
        state_builder: StateBuilder,
        max_in_flight_handoffs: int = DEFAULT_MAX_IN_FLIGHT_HANDOFFS,
        **kwargs,
    ) -> None:
        if max_in_flight_handoffs < 1:
            raise ValueError(
                f"max_in_flight_handoffs must be >= 1, got {max_in_flight_handoffs}"
            )
        else:
            pass
        super().__init__(*args, **kwargs)
        validate_pd_runtime(self)

        self.pd_stage_name = stage_name
        self.pd_partner_stage = partner_stage
        self.pd_state_builder = state_builder
        self.max_in_flight_handoffs = max_in_flight_handoffs
        if self.req_to_token_pool.size < self.max_in_flight_handoffs:
            logger.warning(
                f"Prefill stage {stage_name!r} req_to_token_pool.size "
                f"({self.req_to_token_pool.size}) is smaller than "
                f"max_in_flight_handoffs ({self.max_in_flight_handoffs}); "
                "effective in-flight handoffs will be bounded by "
                "req_to_token_pool.size"
            )
        else:
            pass
        self.pd_pool_id = DEFAULT_KV_POOL_ID
        pool = build_kv_pool(
            self.token_to_kv_pool_allocator.get_kvcache(),
            pool_id=self.pd_pool_id,
        )
        self.kv_registrations = ((pool, None),)

    def active_request_ids(self) -> list[str]:
        request_ids = set(super().active_request_ids())
        chunked_request = self.chunked_req
        if (
            chunked_request is not None
            and chunked_request.rid is not None
            and not chunked_request.finished()
        ):
            request_ids.add(chunked_request.rid)
        else:
            pass
        return sorted(request_ids)

    def mark_running_request_aborted(self, request_id: str) -> bool:
        marked = super().mark_running_request_aborted(request_id)
        chunked_request = self.chunked_req
        if (
            chunked_request is not None
            and chunked_request.rid == request_id
            and not chunked_request.finished()
            and not chunked_request.is_retracted
        ):
            chunked_request.to_finish = FINISH_ABORT()
            return True
        else:
            return marked

    def mark_request_finished_immediately(self, request_id: str) -> list[Req]:
        matches = super().mark_request_finished_immediately(request_id)
        chunked_request = self.chunked_req
        if (
            chunked_request is not None
            and chunked_request.rid == request_id
            and all(existing is not chunked_request for existing in matches)
        ):
            if not chunked_request.finished():
                if chunked_request.to_finish is None:
                    chunked_request.to_finish = FINISH_ABORT()
                else:
                    pass
                chunked_request.update_finish_state()
            else:
                pass
            matches.append(chunked_request)
        else:
            pass
        return matches

    def abort(self, request_id: str, *, defer_running_cleanup: bool = True) -> None:
        super().abort(request_id, defer_running_cleanup=defer_running_cleanup)
        off_thread = (
            self.scheduler_thread_id is not None
            and threading.get_ident() != self.scheduler_thread_id
        )
        if off_thread or (
            self.chunked_req is not None and self.chunked_req.rid == request_id
        ):
            self.notify_pd_wakeup()
        else:
            pass

    def get_num_allocatable_reqs(
        self,
        running_bs: int,
        beam_width: int | None = None,
        running_batch: ScheduleBatch | None = None,
        num_held_rows: int = 0,
    ) -> int:
        base = super().get_num_allocatable_reqs(
            running_bs,
            beam_width=beam_width,
            running_batch=running_batch,
            num_held_rows=num_held_rows,
        )
        handoff_budget = max(
            self.max_in_flight_handoffs
            - len(self.pd_outstanding_releases)
            - running_bs,
            0,
        )
        return min(base, handoff_budget)

    def get_prefill_adder_class(self) -> type[PrefillAdder]:
        if self.pd_outstanding_releases:
            return LeasedPrefillChunkReserveAdder
        else:
            return PrefillChunkReserveAdder

    def get_next_batch_to_run(self) -> ScheduleBatch | None:
        self.reset_pd_wakeup()
        self.drain_due_releases()
        aborted_chunked_request: Req | None = None
        with self.request_admission_lock:
            if (
                self.chunked_req is not None
                and self.chunked_req.rid in self.aborted_request_ids
            ):
                aborted_chunked_request = self.chunked_req
                self.chunked_req = None
            else:
                pass
        if aborted_chunked_request is not None:
            already_finished = aborted_chunked_request.finished()
            # note (YzXiao): only run abort_callback here when mark_running_request_aborted deferred it on this parked chunked request.
            should_run_abort_callback = not already_finished and isinstance(
                aborted_chunked_request.to_finish, FINISH_ABORT
            )
            if not already_finished:
                if aborted_chunked_request.to_finish is None:
                    aborted_chunked_request.to_finish = FINISH_ABORT()
                else:
                    pass
                aborted_chunked_request.update_finish_state()
            else:
                pass
            if aborted_chunked_request.kv.req_pool_idx is not None:
                self.release_request_kv_cache(aborted_chunked_request)
            else:
                pass
            self.finalize_prefill_request(
                aborted_chunked_request,
                run_abort_callback=should_run_abort_callback,
            )
        else:
            pass
        if (
            self.running_batch.is_empty()
            and self.running_batch.batch_is_full
            and self.get_num_allocatable_reqs(0, running_batch=self.running_batch) > 0
        ):
            self.running_batch.batch_is_full = False
        else:
            pass
        return super().get_next_batch_to_run()

    def process_batch_result(
        self, batch: ScheduleBatch, result: GenerationBatchResult
    ) -> None:
        if not batch.forward_mode.is_extend():
            UpstreamScheduler.process_batch_result(self, batch, result)
            return
        else:
            pass

        output_lengths = {id(req): len(req.output_ids) for req in batch.reqs}
        # The Prefill process owns the first sample but never terminalizes it.
        # Decode's existing PREBUILT result path applies the real stop policy.
        with defer_first_token_finish(batch.reqs):
            UpstreamScheduler.process_batch_result(self, batch, result)
        sampled = {
            id(req)
            for req in batch.reqs
            if len(req.output_ids) > output_lengths[id(req)]
        }
        self.handoff_prefilled_requests(batch, sampled)

    def stream_output(self, reqs, return_logprob=False, skip_req=None):
        # A Prefill result must never become a normal StagePayload edge. Normal
        # stop conditions are deferred; anything else is a failed handoff.
        for req in reqs:
            if (
                req is skip_req
                or not req.finished()
                or isinstance(req.finished_reason, FINISH_ABORT)
            ):
                continue
            else:
                pass
            error = RuntimeError(
                f"Prefill request {req.rid!r} terminated before KV handoff"
            )
            self.emit_request_error(req.rid, error)
            req.finished_reason = FINISH_ABORT(str(error))
        return super().stream_output(reqs, return_logprob, skip_req)

    def handoff_prefilled_requests(
        self,
        batch: ScheduleBatch,
        sampled: set[int],
    ) -> None:
        retained: list[Req] = []
        ready_event: torch.cuda.Event | None = None
        use_cuda_ready_event = torch.cuda.is_available() and str(
            self.device
        ).startswith("cuda")
        target_pool_id = DEFAULT_KV_POOL_ID
        for req in batch.reqs:
            if id(req) not in sampled or req.inflight_middle_chunks > 0:
                retained.append(req)
                continue
            else:
                pass
            if req.finished():
                # Abort and invalid-token failures still terminalize locally.
                continue
            else:
                pass
            try:
                transfer_id = f"{req.rid}:pd:{uuid4().hex}"
                continuation = continuation_from_req(
                    req, transfer_id, self.pd_state_builder
                )
                source_page_indices = request_page_indices(
                    self.req_to_token_pool,
                    req,
                    page_size=self.page_size,
                )
                metadata = {"decode_continuation": continuation.encode()}
            except Exception as exc:
                terminal_error, _ = self.finalize_prefill_request(
                    req, terminal_error=exc
                )
                self.release_request_kv_cache(req)
                self.emit_request_error(req.rid, terminal_error)
                continue

            terminal_error, abort_cleanup_needed = self.finalize_prefill_request(req)
            if terminal_error is not None or abort_cleanup_needed:
                self.release_request_kv_cache(req)
                if terminal_error is not None:
                    self.emit_request_error(req.rid, terminal_error)
                else:
                    pass
                continue
            else:
                pass

            if use_cuda_ready_event and ready_event is None:
                # note (YzXiao): under PD with overlap disabled, Prefill executes on the scheduler thread's current stream rather than forward_stream.
                with torch.cuda.device(self.device):
                    ready_event = torch.cuda.Event(interprocess=True)
                    ready_event.record(torch.cuda.current_stream(self.device))
            else:
                pass

            transfer = KVPageTransfer(
                request_id=req.rid,
                transfer_id=transfer_id,
                source_pool_id=self.pd_pool_id,
                target_pool_id=target_pool_id,
                source_page_indices=source_page_indices,
                to_stage=self.pd_partner_stage,
                metadata=metadata,
                lease=self.lease_pd_kv(req),
                ready_event=ready_event,
            )
            self.outbox.put(
                OutgoingMessage(
                    request_id=req.rid,
                    type="kv_transfer",
                    data=transfer,
                )
            )
        batch.reqs = retained
        if not retained:
            batch.batch_is_full = False
        else:
            pass

    def finalize_prefill_request(
        self,
        req: Req,
        *,
        terminal_error: Exception | None = None,
        run_abort_callback: bool = True,
    ) -> tuple[Exception | None, bool]:
        callback_error = self.run_request_finished_callback(req.rid)
        if terminal_error is None:
            terminal_error = callback_error
        else:
            pass

        if req.rid in self.aborted_request_ids:
            status = "aborted"
        elif terminal_error is not None:
            status = "error"
        else:
            status = "success"
        self.emit_model_path_end_once(req.rid, status=status)

        abort_cleanup_needed = self.close_completed_request(req)
        if abort_cleanup_needed and run_abort_callback:
            self.run_abort_callback(req.rid)
        else:
            pass
        return terminal_error, abort_cleanup_needed

    def admin_model_info(self) -> AdminActionResult:
        result = super().admin_model_info()
        data = result.get("data")
        if result.get("success") and isinstance(data, dict):
            data.update(
                {
                    "pd_role": self.scheduler_role,
                    "pd_pool_id": self.pd_pool_id,
                    "pd_partner_stage": self.pd_partner_stage,
                    "pd_max_in_flight_handoffs": self.max_in_flight_handoffs,
                    "pd_req_pool_available": self.req_to_token_pool.available_size(),
                    "pd_kv_pool_available": self.token_to_kv_pool_allocator.available_size(),
                }
            )
        else:
            pass
        return result


class OmniDecodeScheduler(PDKVLifecycle):
    """Omni scheduler that admits transferred Prefill state for Decode."""

    scheduler_role = "decode"

    def __init__(
        self,
        *args,
        stage_name: str,
        state_restorer: StateRestorer,
        resume_schema: str,
        emit_transferred_token: bool = False,
        decode_runway_tokens: int = DEFAULT_DECODE_RUNWAY_TOKENS,
        **kwargs,
    ) -> None:
        self.pd_admissions: deque[DecodeAdmission] = deque()
        self.pd_lifecycle_lock = threading.RLock()
        self.pd_state_restorer = state_restorer
        self.emit_transferred_token = emit_transferred_token
        super().__init__(*args, **kwargs)
        validate_pd_runtime(self)
        serialize_kv_allocator(
            self.token_to_kv_pool_allocator,
            lock=self.pd_lifecycle_lock,
            sync_cuda_streams=True,
        )

        pool_id = DEFAULT_KV_POOL_ID
        self.pd_pool_id = pool_id
        pool = build_kv_pool(
            self.token_to_kv_pool_allocator.get_kvcache(),
            pool_id=pool_id,
        )
        receiver = DecodeKVReceiver(
            pool_id=pool_id,
            allocator=self.token_to_kv_pool_allocator,
            req_to_token_pool=self.req_to_token_pool,
            admissions=self.pd_admissions,
            resume_schema=resume_schema,
            lifecycle_lock=self.pd_lifecycle_lock,
            page_size=self.page_size,
            decode_runway_tokens=decode_runway_tokens,
            wakeup_notifier=self,
            capacity_check=self,
        )
        self.pd_receiver = receiver
        self.kv_registrations = ((pool, receiver),)
        self.disagg_decode_prealloc_queue = types.SimpleNamespace(
            queue=[],
            retracted_queue=[],
            num_tokens_pre_allocated=0,
            pre_allocated_token_counts=lambda: (0, 0),
        )
        self.disagg_decode_transfer_queue = types.SimpleNamespace(
            queue=[],
            num_pending_deferred_releases=lambda: 0,
            has_pending_deferred_releases=lambda: False,
        )

    def active_request_ids(self) -> list[str]:
        request_ids = set(super().active_request_ids())
        with self.pd_lifecycle_lock:
            for admission in self.pd_admissions:
                request_id = admission.continuation.request_id
                if request_id not in self.aborted_request_ids:
                    request_ids.add(request_id)
                else:
                    pass
        return sorted(request_ids)

    def abort_all_requests(self) -> list[str]:
        dropped_admissions: list[DecodeAdmission] = []
        with self.pd_lifecycle_lock:
            while self.pd_admissions:
                admission = self.pd_admissions.popleft()
                self.pd_receiver.free_slots_on_allocation_stream(
                    admission.allocation.slots
                )
                admission.allocation.release_row_credit()
                request_id = admission.continuation.request_id
                if request_id not in self.aborted_request_ids:
                    self.aborted_request_ids.add(request_id)
                    dropped_admissions.append(admission)
                else:
                    pass
        dropped_request_ids: list[str] = []
        for admission in dropped_admissions:
            request_id = admission.continuation.request_id
            dropped_request_ids.append(request_id)
            self.emit_request_error(request_id, AdminDroppedRequestError())
        other_request_ids = super().abort_all_requests()
        self.drain_due_releases()
        self.pd_receiver.notify_capacity_released()
        return sorted(set(dropped_request_ids) | set(other_request_ids))

    def has_active_decode_requests(self) -> bool:
        with self.pd_lifecycle_lock:
            # note (YzXiao): finished requests stay in running_batch.reqs until filter_batch on the next step, so check unfinished requests directly for the zero-active bypass.
            has_unfinished_running = self.running_batch is not None and any(
                not request.finished() for request in self.running_batch.reqs
            )
            return (
                has_unfinished_running
                or bool(self.waiting_queue)
                or bool(self.pd_admissions)
                or self.pd_receiver.has_reservations()
            )

    def decode_capacity_snapshot(self) -> tuple[int, int, bool]:
        return (
            self.req_to_token_pool.available_size()
            - self.pd_receiver.reserved_request_rows,
            self.token_to_kv_pool_allocator.available_size(),
            self.has_active_decode_requests(),
        )

    def pd_holds_kv(self) -> bool:
        with self.pd_lifecycle_lock:
            return (
                bool(self.pd_admissions)
                or self.pd_receiver.has_reservations()
                or super().pd_holds_kv()
            )

    def is_fully_idle(
        self, for_health_check: bool = False, ignore_waiting: bool = False
    ) -> bool:
        # note (ratish): an admitted request waits holding its transferred KV,
        # so the waiting queue counts even when a paused flush asks to skip it.
        with self.pd_lifecycle_lock:
            return super().is_fully_idle(for_health_check=for_health_check)

    def drain_due_releases(self) -> None:
        with self.pd_lifecycle_lock:
            super().drain_due_releases()

    def pd_lifecycle_guard(self) -> AbstractContextManager[None]:
        return self.pd_receiver.suspend_reservations()

    def initial_disaggregation_mode(self) -> DisaggregationMode:
        return DisaggregationMode.DECODE

    def get_next_batch_to_run(self) -> ScheduleBatch | None:
        with self.pd_lifecycle_lock:
            rows_before, kv_before, was_active = self.decode_capacity_snapshot()
            self.reset_pd_wakeup()
            self.drain_due_releases()
            freed_admission = self.drain_decode_admissions()
            # Do not let a new transfer consume the space between the decode
            # memory check and allocation.
            plan = UpstreamScheduler.get_next_disagg_decode_batch_to_run(
                self, self.running_batch
            )
            self.running_batch = plan.running_batch
            rows_after, kv_after, is_active_after = self.decode_capacity_snapshot()
        # note (YzXiao): compare effective free rows and active transitions so simultaneous release/admission or draining to zero active cannot miss a wakeup.
        if (
            freed_admission
            or rows_after > rows_before
            or kv_after > kv_before
            or (was_active and not is_active_after)
        ):
            self.pd_receiver.notify_capacity_released()
        else:
            pass
        return plan.batch_to_run

    def process_batch_result(
        self, batch: ScheduleBatch, result: GenerationBatchResult
    ) -> None:
        # note (YzXiao): hold pd_lifecycle_lock across process_batch_result so upstream free_group_begin/free_group_end cannot hijack concurrent receiver aborts.
        with self.pd_lifecycle_lock:
            rows_before, kv_before, was_active = self.decode_capacity_snapshot()
            super().process_batch_result(batch, result)
            rows_after, kv_after, is_active_after = self.decode_capacity_snapshot()
        if (
            rows_after > rows_before
            or kv_after > kv_before
            or (was_active and not is_active_after)
        ):
            self.pd_receiver.notify_capacity_released()
        else:
            pass

    def release_request_kv_cache(self, req: Req) -> None:
        with self.pd_lifecycle_lock:
            rows_before, kv_before, _ = self.decode_capacity_snapshot()
            super().release_request_kv_cache(req)
            rows_after, kv_after, _ = self.decode_capacity_snapshot()
        if rows_after > rows_before or kv_after > kv_before:
            self.pd_receiver.notify_capacity_released()
        else:
            pass

    def flush_cache(self, empty_cache: bool = True) -> bool:
        flushed = super().flush_cache(empty_cache=empty_cache)
        if flushed:
            self.pd_receiver.notify_capacity_released()
        else:
            pass
        return flushed

    def _add_request_to_queue(self, req, is_retracted=False):
        # Upstream retraction frees KV and expects its own rebootstrap queues.
        # This handoff has no re-prefill protocol; fail only the affected request.
        self.emit_request_error(
            req.rid, RuntimeError("PD decode cannot resume a retracted request")
        )
        self.abort(req.rid)

    def process_input_requests(self, recv_reqs):
        for payload in recv_reqs:
            self.emit_request_error(
                payload.request_id,
                TypeError("Decode stages accept committed KV transfers only"),
            )
            self.abort(payload.request_id)

    def drain_decode_admissions(self) -> bool:
        freed_any = False
        with self.pd_lifecycle_lock:
            while self.pd_admissions:
                admission = self.pd_admissions.popleft()
                request_id = admission.continuation.request_id
                if request_id in self.aborted_request_ids:
                    self.pd_receiver.free_slots_on_allocation_stream(
                        admission.allocation.slots
                    )
                    admission.allocation.release_row_credit()
                    freed_any = True
                    continue
                else:
                    pass
                try:
                    req = request_from_continuation(
                        admission.continuation,
                        admission.allocation,
                        req_to_token_pool=self.req_to_token_pool,
                        state_restorer=self.pd_state_restorer,
                    )
                except Exception as exc:
                    self.pd_receiver.free_slots_on_allocation_stream(
                        admission.allocation.slots
                    )
                    admission.allocation.release_row_credit()
                    freed_any = True
                    self.emit_request_error(request_id, exc)
                    continue
                self.waiting_queue.append(req)
        return freed_any

    def stream_output(
        self,
        reqs: list[Req],
        return_logprob: bool = False,
        skip_req: Req | None = None,
    ) -> None:
        if not self.emit_transferred_token:
            super().stream_output(reqs, return_logprob, skip_req)
            return
        else:
            pass
        step_messages: list[OutgoingMessage] = []
        for request in reqs:
            if (
                request is skip_req
                or request.rid in self.aborted_request_ids
                or isinstance(request.finished_reason, FINISH_ABORT)
                or request.rid in self.first_emit_done
                or request.omni_data is None
                or len(request.output_ids) != 1
                or self.stream_output_builder is None
            ):
                continue
            else:
                pass
            try:
                messages = list(
                    self.stream_output_builder(
                        request.rid,
                        request.omni_data,
                        RequestOutput(
                            request_id=request.rid, data=request.output_ids[0]
                        ),
                    )
                )
            except Exception as error:
                logger.exception(f"PD first-token stream failed for {request.rid}")
                request.finished_reason = FINISH_ABORT(str(error))
                self.release_request_kv_cache(request)
                self.emit_model_path_end_once(request.rid, status="error")
                self.close_completed_request(request)
                self.emit_request_error(request.rid, error)
            else:
                self.stage_stream_messages(request.rid, messages, step_messages)
        put_messages(self.outbox, step_messages)
        super().stream_output(reqs, return_logprob, skip_req)

    def discard_pending_request_admissions(self) -> None:
        super().discard_pending_request_admissions()
        self.pd_receiver.close()
        with self.pd_lifecycle_lock:
            while self.pd_admissions:
                admission = self.pd_admissions.popleft()
                self.pd_receiver.free_slots_on_allocation_stream(
                    admission.allocation.slots
                )
                admission.allocation.release_row_credit()

    def abort(self, request_id: str, *, defer_running_cleanup: bool = True) -> None:
        should_wakeup = False
        with self.pd_lifecycle_lock:
            for req in self.waiting_queue:
                if req.rid == request_id:
                    # Note(Yue Yin): abort runs on the Stage event loop, and
                    # only the scheduler thread may mutate the request table.
                    self.defer_pd_kv_release(req)
                    break
                else:
                    pass
            if self.pd_admissions:
                should_wakeup = True
            else:
                pass
            super().abort(
                request_id,
                defer_running_cleanup=defer_running_cleanup,
            )
        if should_wakeup:
            self.notify_pd_wakeup()
        else:
            pass
        self.pd_receiver.cancel_request_waiters(request_id)

    def admin_model_info(self) -> AdminActionResult:
        result = super().admin_model_info()
        data = result.get("data")
        if result.get("success") and isinstance(data, dict):
            with self.pd_lifecycle_lock:
                data.update(
                    {
                        "pd_role": self.scheduler_role,
                        "pd_pool_id": self.pd_pool_id,
                        "pd_outstanding_releases": len(self.pd_outstanding_releases),
                        "pd_admissions_pending": len(self.pd_admissions),
                        "pd_reservations_in_flight": len(self.pd_receiver.reservations),
                        "pd_prealloc_waiters": len(self.pd_receiver.waiters),
                        "pd_reserved_request_rows": (
                            self.pd_receiver.reserved_request_rows
                        ),
                        "pd_req_pool_available": (
                            self.req_to_token_pool.available_size()
                        ),
                        "pd_kv_pool_available": (
                            self.token_to_kv_pool_allocator.available_size()
                        ),
                    }
                )
        else:
            pass
        return result


def validate_pd_runtime(scheduler: OmniScheduler) -> None:
    if scheduler.session_adapter is not None:
        raise NotImplementedError(
            "PD session adapters require a retained-state ownership contract"
        )
    else:
        pass
    if isinstance(scheduler.req_to_token_pool, HybridReqToTokenPool):
        raise NotImplementedError("PD does not transfer linear-attention state")
    elif isinstance(
        scheduler.token_to_kv_pool_allocator,
        (PureSWATokenToKVPoolAllocator, SWATokenToKVPoolAllocator),
    ):
        raise NotImplementedError("PD does not transfer sliding-window KV ownership")
    else:
        pass
    if scheduler.tp_size != 1:
        raise NotImplementedError("PD currently requires tp_size == 1")
    else:
        pass
    if scheduler.page_size < 1:
        raise ValueError(f"PD requires page_size >= 1, got {scheduler.page_size}")
    else:
        pass
    if scheduler.async_prefill:
        raise NotImplementedError("PD does not support async_prefill")
    else:
        pass
    kv_cache = scheduler.token_to_kv_pool_allocator.get_kvcache()
    if kv_cache is not None and kv_cache.post_capture_active:
        raise NotImplementedError("PD does not support post-capture VMM KV cache pools")
    else:
        pass
    if not scheduler.server_args.disable_radix_cache:
        raise NotImplementedError("PD currently requires RadixCache disabled")
    else:
        pass
    if not scheduler.spec_algorithm.is_none():
        raise NotImplementedError("PD does not support speculative decoding")
    else:
        pass
