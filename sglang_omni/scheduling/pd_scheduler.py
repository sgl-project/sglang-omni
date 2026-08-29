# SPDX-License-Identifier: Apache-2.0
"""Scheduler roles for explicit Prefill/Decode pipeline stages."""

from __future__ import annotations

import logging
import queue
import threading
import types
from typing import Any, Callable, Literal

from sglang.srt.managers.schedule_batch import FINISH_ABORT, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler as _Upstream

from sglang_omni.comm import KVPageTransfer
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler, _detach_request_data
from sglang_omni.scheduling.pd_alloc_lock import LockedKVAllocator
from sglang_omni.scheduling.pd_decode_selection import select_decode_stage
from sglang_omni.scheduling.pd_utils import (
    DecodeKVReceiver,
    DecodeRequestPoolExhausted,
    SGLangKVLease,
    build_kv_pool,
    continuation_from_req,
    default_state_builder,
    default_state_restorer,
    defer_first_token_finish,
    drain_due_releases,
    req_from_continuation,
    request_page_indices,
)


def scheduler_class_for_role(role: Literal["prefill", "decode"]) -> type[OmniScheduler]:
    """The single compiler role-to-concrete-scheduler mapping."""

    return {
        "prefill": OmniPrefillScheduler,
        "decode": OmniDecodeScheduler,
    }[role]


def model_pd_scheduler_kwargs(
    scheduler_cls: type,
    *,
    state_builder: Callable,
    state_restorer: Callable,
    resume_schema: str,
) -> dict[str, Any]:
    """Select the model seam required by one concrete PD scheduler."""

    if scheduler_cls is OmniPrefillScheduler:
        return {"state_builder": state_builder}
    if scheduler_cls is OmniDecodeScheduler:
        return {
            "state_restorer": state_restorer,
            "allowed_resume_schemas": frozenset({resume_schema}),
        }
    if scheduler_cls is OmniScheduler:
        return {}
    raise TypeError(f"unsupported scheduler class {scheduler_cls!r}")


logger = logging.getLogger(__name__)


class OmniPrefillScheduler(OmniScheduler):
    """Omni scheduler whose generated requests stop after Prefill."""

    def __init__(
        self,
        *args,
        stage_name: str,
        partner_stage: str,
        decode_targets: tuple[str, ...] = (),
        state_builder: Callable = default_state_builder,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        _validate_pd_runtime(self)

        self._pd_stage_name = stage_name
        self._pd_partner_stage = partner_stage
        # Note (Audrey Zheng): every Decode half this Prefill may send to.
        # Sorted, because the KV send is rank-addressed: two Prefill ranks that
        # disagree on the order would split one request's pages across
        # different Decode halves, and nothing in the transfer path checks for
        # it. Empty falls back to `partner`, which is 1:1.
        self._pd_decode_targets = tuple(sorted(decode_targets)) or (partner_stage,)
        self._pd_state_builder = state_builder
        self._pd_handoff_seq = 0
        # Requests whose copy Decode has acknowledged. The comm thread puts
        # them here; this thread releases them. See SGLangKVLease.
        self._pd_due_releases: queue.SimpleQueue = queue.SimpleQueue()
        self._pd_pool_id = f"{stage_name}:kv"
        pool = build_kv_pool(
            self.token_to_kv_pool_allocator.get_kvcache(),
            pool_id=self._pd_pool_id,
        )
        self.kv_registrations = ((pool, None),)

    def _resolve_decode_stage(self, req: Any) -> str:
        """Return the Decode stage that receives this request's KV.

        The choice lives in `select_decode_stage`, which takes only values
        every Prefill rank sees identically. That constraint is not cosmetic --
        read that function before changing the policy.
        """
        return select_decode_stage(self._pd_decode_targets, str(req.rid))

    def get_next_batch_to_run(self):
        drain_due_releases(self._pd_due_releases, self.tree_cache)
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
                decode_stage = self._resolve_decode_stage(req)
                # Note (Audrey Zheng): every join key downstream is the
                # transfer id, and under tp_size > 1 each rank builds this
                # line for the same request. uuid4 is drawn per rank, so the
                # ranks would disagree about what to call one transfer.
                #
                # The counter is rank-identical because every TP rank runs the
                # same batches in the same order -- that is what makes TP a
                # collective. The rid alone would not do: a client may supply
                # its own request_id (`serve/openai_api.py`), and a repeat
                # would collide with the receiver's tombstone window.
                self._pd_handoff_seq += 1
                transfer_id = f"{req.rid}:pd:{self._pd_handoff_seq}"
                continuation = continuation_from_req(
                    req, transfer_id, self._pd_state_builder
                )
                transfer = KVPageTransfer(
                    request_id=req.rid,
                    transfer_id=transfer_id,
                    source_pool_id=self._pd_pool_id,
                    target_pool_id=f"{decode_stage}:kv",
                    source_page_indices=request_page_indices(
                        self.req_to_token_pool, req
                    ),
                    to_stage=decode_stage,
                    metadata={"decode_continuation": continuation.encode()},
                    lease=SGLangKVLease(req, self.tree_cache, self._pd_due_releases),
                )
            except Exception as exc:
                self._release_request_kv_cache(req)
                _detach_request_data(req)
                self._emit_request_error(req.rid, exc)
                continue
            _detach_request_data(req)
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


class OmniDecodeScheduler(OmniScheduler):
    """Omni scheduler that admits transferred Prefill state for Decode."""

    def __init__(
        self,
        *args,
        stage_name: str,
        state_restorer: Callable = default_state_restorer,
        allowed_resume_schemas: frozenset[str] = frozenset(),
        partner_stage: str | None = None,
        decode_targets: tuple[str, ...] = (),
        **kwargs,
    ) -> None:
        # The compiler hands both halves the same kwargs. A Decode half has no
        # peer to choose, so it drops the two that describe one.
        del partner_stage, decode_targets
        self._pd_admissions = queue.SimpleQueue()
        self._pd_deferred_admission = None
        self._pd_admission_lock = threading.Lock()
        self._pd_state_restorer = state_restorer
        super().__init__(*args, **kwargs)
        _validate_pd_runtime(self)

        _serialize_kv_allocation(self)
        _warn_if_decode_queue_unbounded(self, stage_name)

        pool_id = f"{stage_name}:kv"
        pool = build_kv_pool(
            self.token_to_kv_pool_allocator.get_kvcache(),
            pool_id=pool_id,
        )
        receiver = DecodeKVReceiver(
            pool_id=pool_id,
            allocator=self.token_to_kv_pool_allocator,
            admissions=self._pd_admissions,
            allowed_resume_schemas=allowed_resume_schemas,
        )
        self._pd_receiver = receiver
        self.kv_registrations = ((pool, receiver),)
        self.disagg_decode_prealloc_queue = types.SimpleNamespace(
            queue=[], retracted_queue=[], num_tokens_pre_allocated=0
        )
        self.disagg_decode_transfer_queue = types.SimpleNamespace(queue=[])

    def _initial_disaggregation_mode(self):
        from sglang.srt.disaggregation.utils import DisaggregationMode

        return DisaggregationMode.DECODE

    def get_next_batch_to_run(self):
        self._drain_decode_admissions()
        plan = _Upstream.get_next_disagg_decode_batch_to_run(self, self.running_batch)
        self.running_batch = plan.running_batch
        return plan.batch_to_run

    def process_input_requests(self, recv_reqs):
        for payload in recv_reqs:
            self._emit_request_error(
                payload.request_id,
                TypeError("Decode stages accept committed KV transfers only"),
            )
            self.abort(payload.request_id)

    def _drain_decode_admissions(self) -> None:
        with self._pd_admission_lock:
            while True:
                admission = self._pd_deferred_admission
                if admission is None:
                    try:
                        admission = self._pd_admissions.get_nowait()
                    except queue.Empty:
                        return
                request_id = admission.continuation.request_id
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
                    self.outbox.put(
                        OutgoingMessage(request_id=request_id, type="admitted")
                    )
                    self._emit_request_error(request_id, exc)
                    continue
                self._pd_deferred_admission = None
                self.waiting_queue.append(req)
                self.outbox.put(OutgoingMessage(request_id=request_id, type="admitted"))

    def _discard_pending_request_admissions(self) -> None:
        super()._discard_pending_request_admissions()
        self._pd_receiver.close()
        with self._pd_admission_lock:
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


# Note (Audrey Zheng): every object that allocates or frees from one KV pool.
# `bootstrap.py` hands the same allocator to `create_tree_cache` before this
# scheduler exists, so rebinding only our own attribute would leave the tree
# cache calling an unwrapped alias.
_KV_ALLOCATOR_HOLDERS = ("tree_cache",)


def _serialize_kv_allocation(scheduler: OmniDecodeScheduler) -> None:
    """Give every holder of the KV allocator the same locked object.

    A Decode half allocates from two threads: this scheduler for decode steps
    and the comm event loop through `DecodeKVReceiver.reserve`. Upstream's
    `alloc` reads `free_pages`, slices it and writes the remainder back with no
    lock, which is correct while one thread owns the allocator and hands the
    same slots to both callers when two do.

    One lock only helps if every caller goes through it, so the wrapper has to
    reach the tree cache as well as this scheduler.

    Wrapping here rather than in `bootstrap.py` is deliberate: upstream's
    `SWAChunkCache.__init__` asserts the allocator's concrete type, and a proxy
    handed to that constructor would fail the assert for models that use it.
    Wrapping at construction rather than after it means no request has been
    served through an unwrapped alias.
    """
    locked = LockedKVAllocator(scheduler.token_to_kv_pool_allocator)
    scheduler.token_to_kv_pool_allocator = locked
    for name in _KV_ALLOCATOR_HOLDERS:
        holder = getattr(scheduler, name, None)
        if holder is None:
            continue
        if not hasattr(holder, "token_to_kv_pool_allocator"):
            raise RuntimeError(
                f"PD decode half: {name} has no token_to_kv_pool_allocator to "
                "rebind, so its allocations would bypass the lock"
            )
        holder.token_to_kv_pool_allocator = locked


def kv_allocator_holders(scheduler: OmniDecodeScheduler) -> dict[str, Any]:
    """Every allocator reference the scheduler can reach, by holder name.

    A test asserts these are one object. That is what catches a new holder
    appearing between the allocator's creation and this scheduler's.
    """
    found: dict[str, Any] = {
        "scheduler": scheduler.__dict__.get("token_to_kv_pool_allocator")
    }
    for name in _KV_ALLOCATOR_HOLDERS:
        holder = getattr(scheduler, name, None)
        if holder is not None:
            found[name] = getattr(holder, "token_to_kv_pool_allocator", None)
    return found


def _warn_if_decode_queue_unbounded(scheduler: Any, stage_name: str) -> None:
    """Say which admission policy this Decode half is running under.

    A colocated replica throttles admission without trying to: prefill and
    decode contend for one card and one scheduler thread, so accepting more
    work slows the accepting. Splitting the halves removes that feedback and
    nothing replaces it.

    With no queue bound the excess arrives as latency rather than as
    rejection. Measured on two H200s at offered 16, the Decode half held about
    437 requests against max_running_requests 64, and a request took 40.96 s
    against 2.29 s colocated -- while admission read 100% throughout, which is
    why it does not surface as a failure.

    Unbounded is a legitimate choice for an offline workload and a poor one
    for interactive serving, so state which one is in effect rather than pick.
    """
    if getattr(scheduler.server_args, "max_queued_requests", 0):
        return
    logger.info(
        "PD decode stage %s has no max_queued_requests, so its queue is "
        "unbounded and overload will appear as latency rather than "
        "rejection. Set --max-queued-requests to bound it; "
        "models/qwen3_tts sets 16 for its generation stage.",
        stage_name,
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
