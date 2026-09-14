# SPDX-License-Identifier: Apache-2.0
"""Scheduler roles for explicit Prefill/Decode pipeline stages."""

from __future__ import annotations

import logging
import queue
import threading
import types
from contextlib import nullcontext
from typing import Any, Callable
from uuid import uuid4

from sglang.srt.managers.schedule_batch import FINISH_ABORT, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler as _Upstream

from sglang_omni.comm import KVPageTransfer
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.pd_utils import (
    DecodeKVReceiver,
    DecodeRequestPoolExhausted,
    SGLangKVLease,
    build_kv_pool,
    continuation_from_req,
    defer_first_token_finish,
    req_from_continuation,
    request_page_indices,
    serialize_kv_allocator,
)

logger = logging.getLogger(__name__)


class _PDKVLifecycle(OmniScheduler):
    """Own PD-only KV state that upstream scheduler lifecycle checks cannot see."""

    def __init__(self, *args, **kwargs) -> None:
        self._pd_due_releases: queue.SimpleQueue = queue.SimpleQueue()
        self._pd_outstanding_releases: set[str] = set()
        super().__init__(*args, **kwargs)

    def _pd_holds_kv(self) -> bool:
        return bool(self._pd_outstanding_releases)

    def _pd_lifecycle_guard(self):
        return nullcontext()

    def _lease_pd_kv(self, req) -> SGLangKVLease:
        lease = SGLangKVLease(req, self._pd_due_releases)
        self._pd_outstanding_releases.add(req.rid)
        return lease

    def _defer_pd_kv_release(self, req) -> None:
        self._pd_outstanding_releases.add(req.rid)
        self._pd_due_releases.put(req)

    def is_fully_idle(self, for_health_check: bool = False) -> bool:
        # Health checks only care whether a running request can carry their
        # result. Destructive operations must also see PD-owned KV.
        if not for_health_check and self._pd_holds_kv():
            return False
        return _Upstream.is_fully_idle(self, for_health_check=for_health_check)

    def _drain_due_releases(self) -> None:
        while True:
            try:
                req = self._pd_due_releases.get_nowait()
            except queue.Empty:
                return
            try:
                self._release_request_kv_cache(req)
            except Exception:
                # One bad request must not strand the rest of the queue.
                logger.exception("PD release failed for %r", getattr(req, "rid", req))
            else:
                self._pd_outstanding_releases.discard(req.rid)

    def _run_weight_update_with_lifecycle(
        self,
        payload: dict[str, Any],
        update_fn,
        result_data: dict[str, Any],
        *,
        keep_pause_on_failure: bool = False,
    ) -> dict[str, Any]:
        def update_after_pd_drains(update_payload):
            self._drain_due_releases()
            if self._pd_holds_kv():
                return False, "PD-owned KV is still in flight"
            return update_fn(update_payload)

        with self._pd_lifecycle_guard():
            return super()._run_weight_update_with_lifecycle(
                payload,
                update_after_pd_drains,
                result_data,
                keep_pause_on_failure=keep_pause_on_failure,
            )

    def flush_cache(self, *args, **kwargs):
        with self._pd_lifecycle_guard():
            # Upstream clears both pools once it reads the scheduler as idle.
            self._drain_due_releases()
            return _Upstream.flush_cache(self, *args, **kwargs)


class OmniPrefillScheduler(_PDKVLifecycle):
    """Omni scheduler whose generated requests stop after Prefill."""

    scheduler_role = "prefill"

    def __init__(
        self,
        *args,
        stage_name: str,
        partner_stage: str,
        state_builder: Callable,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        _validate_pd_runtime(self)

        self._pd_stage_name = stage_name
        self._pd_partner_stage = partner_stage
        self._pd_state_builder = state_builder
        self._pd_pool_id = f"{stage_name}:kv"
        pool = build_kv_pool(
            self.token_to_kv_pool_allocator.get_kvcache(),
            pool_id=self._pd_pool_id,
        )
        self.kv_registrations = ((pool, None),)

    def get_next_batch_to_run(self):
        self._drain_due_releases()
        if (
            self.running_batch.is_empty()
            and self.running_batch.batch_is_full
            and self.req_to_token_pool.available_size() > 0
        ):
            self.running_batch.batch_is_full = False
        return super().get_next_batch_to_run()

    def process_batch_result(self, batch, result):
        if not batch.forward_mode.is_extend():
            return _Upstream.process_batch_result(self, batch, result)

        output_lengths = {id(req): len(req.output_ids) for req in batch.reqs}
        # The Prefill process owns the first sample but never terminalizes it.
        # Decode's existing PREBUILT result path applies the real stop policy.
        with defer_first_token_finish(batch.reqs):
            _Upstream.process_batch_result(self, batch, result)
        sampled = {
            id(req)
            for req in batch.reqs
            if len(req.output_ids) > output_lengths[id(req)]
        }
        self._handoff_prefilled_requests(batch, sampled)
        return None

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
            error = RuntimeError(
                f"Prefill request {req.rid!r} terminated before KV handoff"
            )
            self._emit_request_error(req.rid, error)
            req.finished_reason = FINISH_ABORT(str(error))
        return super().stream_output(reqs, return_logprob, skip_req)

    def _handoff_prefilled_requests(
        self,
        batch: ScheduleBatch,
        sampled: set[int],
    ) -> None:
        retained = []
        for req in batch.reqs:
            if id(req) not in sampled or req.inflight_middle_chunks > 0:
                retained.append(req)
                continue
            if req.finished():
                # Abort and invalid-token failures still terminalize locally.
                continue
            try:
                transfer_id = f"{req.rid}:pd:{uuid4().hex}"
                continuation = continuation_from_req(
                    req, transfer_id, self._pd_state_builder
                )
                source_page_indices = request_page_indices(self.req_to_token_pool, req)
                metadata = {"decode_continuation": continuation.encode()}
            except Exception as exc:
                terminal_error, _ = self._finalize_prefill_request(
                    req, terminal_error=exc
                )
                self._release_request_kv_cache(req)
                self._emit_request_error(req.rid, terminal_error)
                continue

            terminal_error, abort_cleanup_needed = self._finalize_prefill_request(req)
            if terminal_error is not None or abort_cleanup_needed:
                self._release_request_kv_cache(req)
                if terminal_error is not None:
                    self._emit_request_error(req.rid, terminal_error)
                continue

            transfer = KVPageTransfer(
                request_id=req.rid,
                transfer_id=transfer_id,
                source_pool_id=self._pd_pool_id,
                target_pool_id=f"{self._pd_partner_stage}:kv",
                source_page_indices=source_page_indices,
                to_stage=self._pd_partner_stage,
                metadata=metadata,
                lease=self._lease_pd_kv(req),
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

    def _finalize_prefill_request(
        self,
        req,
        *,
        terminal_error: Exception | None = None,
    ) -> tuple[Exception | None, bool]:
        callback_error = self._run_request_finished_callback(req.rid)
        if terminal_error is None:
            terminal_error = callback_error

        if req.rid in self._aborted_request_ids:
            status = "aborted"
        elif terminal_error is not None:
            status = "error"
        else:
            status = "success"
        self._emit_model_path_end_once(req.rid, status=status)

        abort_cleanup_needed = self._close_completed_request(req)
        if abort_cleanup_needed:
            self._run_abort_callback(req.rid)
        return terminal_error, abort_cleanup_needed


class OmniDecodeScheduler(_PDKVLifecycle):
    """Omni scheduler that admits transferred Prefill state for Decode."""

    scheduler_role = "decode"

    def __init__(
        self,
        *args,
        stage_name: str,
        state_restorer: Callable,
        resume_schema: str,
        **kwargs,
    ) -> None:
        self._pd_admissions = queue.SimpleQueue()
        self._pd_deferred_admission = None
        self._pd_lifecycle_lock = threading.RLock()
        self._pd_state_restorer = state_restorer
        super().__init__(*args, **kwargs)
        _validate_pd_runtime(self)
        serialize_kv_allocator(
            self.token_to_kv_pool_allocator,
            lock=self._pd_lifecycle_lock,
        )

        pool_id = f"{stage_name}:kv"
        pool = build_kv_pool(
            self.token_to_kv_pool_allocator.get_kvcache(),
            pool_id=pool_id,
        )
        receiver = DecodeKVReceiver(
            pool_id=pool_id,
            allocator=self.token_to_kv_pool_allocator,
            admissions=self._pd_admissions,
            resume_schema=resume_schema,
            lifecycle_lock=self._pd_lifecycle_lock,
        )
        self._pd_receiver = receiver
        self.kv_registrations = ((pool, receiver),)
        self.disagg_decode_prealloc_queue = types.SimpleNamespace(
            queue=[], retracted_queue=[], num_tokens_pre_allocated=0
        )
        self.disagg_decode_transfer_queue = types.SimpleNamespace(queue=[])

    def _pd_holds_kv(self) -> bool:
        with self._pd_lifecycle_lock:
            return (
                self._pd_deferred_admission is not None
                or not self._pd_admissions.empty()
                or self._pd_receiver.has_reservations()
                or super()._pd_holds_kv()
            )

    def is_fully_idle(self, for_health_check: bool = False) -> bool:
        with self._pd_lifecycle_lock:
            return super().is_fully_idle(for_health_check=for_health_check)

    def _drain_due_releases(self) -> None:
        with self._pd_lifecycle_lock:
            super()._drain_due_releases()

    def _pd_lifecycle_guard(self):
        return self._pd_receiver.suspend_reservations()

    def _initial_disaggregation_mode(self):
        from sglang.srt.disaggregation.utils import DisaggregationMode

        return DisaggregationMode.DECODE

    def get_next_batch_to_run(self):
        with self._pd_lifecycle_lock:
            self._drain_due_releases()
            self._drain_decode_admissions()
            # Do not let a new transfer consume the space between the decode
            # memory check and allocation.
            plan = _Upstream.get_next_disagg_decode_batch_to_run(
                self, self.running_batch
            )
            self.running_batch = plan.running_batch
            return plan.batch_to_run

    def _add_request_to_queue(self, req, is_retracted=False):
        # Upstream retraction frees KV and expects its own rebootstrap queues.
        # This handoff has no re-prefill protocol; fail only the affected request.
        self._emit_request_error(
            req.rid, RuntimeError("PD decode cannot resume a retracted request")
        )
        self.abort(req.rid)

    def process_input_requests(self, recv_reqs):
        for payload in recv_reqs:
            self._emit_request_error(
                payload.request_id,
                TypeError("Decode stages accept committed KV transfers only"),
            )
            self.abort(payload.request_id)

    def _drain_decode_admissions(self) -> None:
        with self._pd_lifecycle_lock:
            while True:
                admission = self._pd_deferred_admission
                if admission is None:
                    try:
                        admission = self._pd_admissions.get_nowait()
                    except queue.Empty:
                        return
                request_id = admission.continuation.request_id
                admitted = OutgoingMessage(
                    request_id=request_id,
                    type="admitted",
                    metadata={"replica_bindings": admission.replica_bindings},
                )
                if request_id in self._aborted_request_ids:
                    self._pd_deferred_admission = None
                    self.token_to_kv_pool_allocator.free(admission.allocation.slots)
                    continue
                try:
                    req = req_from_continuation(
                        admission.continuation,
                        admission.allocation,
                        req_to_token_pool=self.req_to_token_pool,
                        state_restorer=self._pd_state_restorer,
                    )
                except DecodeRequestPoolExhausted:
                    self._pd_deferred_admission = admission
                    return
                except Exception as exc:
                    self._pd_deferred_admission = None
                    self.token_to_kv_pool_allocator.free(admission.allocation.slots)
                    self.outbox.put(admitted)
                    self._emit_request_error(request_id, exc)
                    continue
                self._pd_deferred_admission = None
                self.waiting_queue.append(req)
                self.outbox.put(admitted)

    def _discard_pending_request_admissions(self) -> None:
        super()._discard_pending_request_admissions()
        self._pd_receiver.close()
        with self._pd_lifecycle_lock:
            admission = self._pd_deferred_admission
            self._pd_deferred_admission = None
            if admission is not None:
                self.token_to_kv_pool_allocator.free(admission.allocation.slots)
            while True:
                try:
                    admission = self._pd_admissions.get_nowait()
                except queue.Empty:
                    return
                self.token_to_kv_pool_allocator.free(admission.allocation.slots)

    def abort(self, request_id: str, *, defer_running_cleanup: bool = True) -> None:
        with self._pd_lifecycle_lock:
            for req in self.waiting_queue:
                if req.rid == request_id:
                    # Note(Yue Yin): abort runs on the Stage event loop, and
                    # only the scheduler thread may mutate the request table.
                    self._defer_pd_kv_release(req)
                    break
            super().abort(
                request_id,
                defer_running_cleanup=defer_running_cleanup,
            )


def _validate_pd_runtime(scheduler: OmniScheduler) -> None:
    if scheduler.tp_size != 1:
        raise NotImplementedError("PD currently requires tp_size == 1")
    if scheduler.page_size != 1:
        raise NotImplementedError("PD currently requires page_size == 1")
    if not scheduler.server_args.disable_radix_cache:
        raise NotImplementedError("PD currently requires RadixCache disabled")
    if not scheduler.spec_algorithm.is_none():
        raise NotImplementedError("PD does not support speculative decoding")
