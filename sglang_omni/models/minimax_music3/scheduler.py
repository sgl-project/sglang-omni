# SPDX-License-Identifier: Apache-2.0
"""OmniScheduler specialization that keeps MiniMax Music 3 CFG rows paired."""

from __future__ import annotations

from collections.abc import Iterable
from threading import get_ident
from typing import Literal

from sglang.srt.managers.schedule_batch import NextBatchPlan, Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import SchedulePolicy
from sglang.srt.mem_cache.prefill_budget import PrefillBudget
from sglang.srt.runtime_context import get_schedule
from typing_extensions import Unpack

from sglang_omni.models.minimax_music3.serial_offload import get_coordinator
from sglang_omni.models.minimax_music3.sglang_request_builder import (
    MiniMaxMusic3SGLangRequestData,
    cfg_uncond_rid,
    is_cfg_uncond_rid,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler, OmniSchedulerArguments


class MiniMaxMusic3SchedulePolicy(SchedulePolicy):
    """Keep CFG rows adjacent after the configured queue ordering."""

    def __init__(self, scheduler: MiniMaxMusic3Scheduler) -> None:
        policy = scheduler.policy
        super().__init__(
            scheduler.schedule_policy,
            policy.tree_cache,
            policy.enable_hierarchical_cache,
            policy.enable_priority_scheduling,
            policy.schedule_low_priority_values_first,
        )
        self.scheduler: MiniMaxMusic3Scheduler = scheduler

    def calc_priority(
        self,
        waiting_queue: list[Req],
        running_batch: ScheduleBatch | None = None,
        processed_tokens: int = 0,
    ) -> None:
        super().calc_priority(waiting_queue, running_batch, processed_tokens)
        # note (ruiling): a policy such as random places each row alone, and the runner needs the two rows of a request next to each other.
        twins = {req.rid: req for req in waiting_queue if is_cfg_uncond_rid(req.rid)}
        paired = {cfg_uncond_rid(req.rid) for req in waiting_queue} & twins.keys()
        ordered: list[Req] = []
        for req in waiting_queue:
            if req.rid in paired:
                continue
            else:
                ordered.append(req)
            twin_rid = cfg_uncond_rid(req.rid)
            if twin_rid in paired:
                ordered.append(twins[twin_rid])
            else:
                pass
        waiting_queue[:] = ordered
        if running_batch is None:
            return
        else:
            pass
        self.scheduler.prepare_cfg_prefill(waiting_queue, running_batch)


class MiniMaxMusic3Scheduler(OmniScheduler["MiniMaxMusic3SGLangRequestData"]):
    """Admit, decode and retire every request as a CFG row pair."""

    def __init__(
        self,
        **kwargs: Unpack[OmniSchedulerArguments[MiniMaxMusic3SGLangRequestData]],
    ) -> None:
        super().__init__(**kwargs)
        self.policy = MiniMaxMusic3SchedulePolicy(self)

    def enqueue_built_request(
        self,
        payload: StagePayload,
        pending_stream_done: bool,
        req_data: MiniMaxMusic3SGLangRequestData,
        *,
        request_admission_lock_held: bool = False,
    ) -> None:
        super().enqueue_built_request(
            payload,
            pending_stream_done,
            req_data,
            request_admission_lock_held=request_admission_lock_held,
        )
        uncond = req_data.cfg_uncond
        if uncond is None:
            return
        else:
            pass
        if request_admission_lock_held:
            self.enqueue_cfg_uncond(req_data, uncond)
            return
        else:
            pass
        with self.request_admission_lock:
            self.enqueue_cfg_uncond(req_data, uncond)

    def enqueue_cfg_uncond(
        self,
        req_data: MiniMaxMusic3SGLangRequestData,
        uncond: MiniMaxMusic3SGLangRequestData,
    ) -> None:
        cond_req = req_data.req
        if not self.waiting_queue or self.waiting_queue[-1] is not cond_req:
            return
        else:
            pass
        req = uncond.req
        self.normalize_req_token_arrays(req)
        # note (ruiling): priority scheduling sorts the queue by these two, and SGLang sets them on each row alone.
        req.priority = cond_req.priority
        enqueue_seconds = (
            cond_req._coalesce_enqueue_t
        )  # noqa: leading-underscore  # upstream spelling
        cond_req.time_stats.set_wait_queue_entry_time(enqueue_seconds)
        req.time_stats.set_wait_queue_entry_time(enqueue_seconds)
        req._coalesce_enqueue_t = (
            cond_req._coalesce_enqueue_t
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        req._omni_terminal_claimed = False  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        req.omni_data = uncond  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        self.waiting_queue.append(req)

    def get_new_batch_prefill(self, running_batch: ScheduleBatch) -> NextBatchPlan:
        input_budget = self.max_prefill_tokens
        self.deferred_prefill_requests: list[Req] = []
        self.prefill_claim_request_id: str | None = None
        plan: NextBatchPlan | None = None
        try:
            plan = super().get_new_batch_prefill(running_batch)
            return plan
        finally:
            self.max_prefill_tokens = input_budget
            self.waiting_queue.extend(self.deferred_prefill_requests)
            self.deferred_prefill_requests.clear()
            if self.prefill_claim_request_id is not None:
                admitted_ids = (
                    {request.rid for request in plan.batch_to_run.reqs}
                    if plan is not None and plan.batch_to_run is not None
                    else set()
                )
                if (
                    not {
                        self.prefill_claim_request_id,
                        cfg_uncond_rid(self.prefill_claim_request_id),
                    }
                    <= admitted_ids
                ):
                    get_coordinator().cancel_ar(self.prefill_claim_request_id)
                else:
                    pass
                self.prefill_claim_request_id = None
            else:
                pass

    def prepare_cfg_prefill(
        self, waiting_queue: list[Req], running_batch: ScheduleBatch
    ) -> None:
        """Reserve a sorted CFG prefix before the prefill adder runs."""
        limit = self.pair_admission_limit(waiting_queue, running_batch)
        coordinator = get_coordinator()
        if coordinator.enabled and limit >= 2:
            request_id = waiting_queue[0].rid
            previous_owner = coordinator.owner_request_id
            if coordinator.try_acquire_ar(request_id):
                limit = 2
                if previous_owner != request_id:
                    self.prefill_claim_request_id = request_id
                else:
                    pass
            else:
                limit = 0
        else:
            pass
        if limit:
            first_pair_tokens = sum(
                -(-len(request.origin_input_ids) // self.page_size) * self.page_size
                for request in waiting_queue[:2]
            )
            self.max_prefill_tokens = max(
                self.max_prefill_tokens, first_pair_tokens + 1
            )
        else:
            pass
        self.deferred_prefill_requests.extend(waiting_queue[limit:])
        waiting_queue[:] = waiting_queue[:limit]

    def request_kv_capacity_error(
        self,
        req: Req,
        *,
        session_capacity_checked: bool = False,
        decode_path: Literal["plain", "spec"] | None = None,
    ) -> str | None:
        error = super().request_kv_capacity_error(
            req,
            session_capacity_checked=session_capacity_checked,
            decode_path=decode_path,
        )
        if error is not None:
            return error
        else:
            pass
        prefill_max_requests = get_schedule().prefill_max_requests
        if prefill_max_requests is not None and prefill_max_requests < 2:
            return (
                "MiniMax Music 3 requires at least 2 prefill rows for a CFG pair. "
                f"prefill_max_requests={prefill_max_requests} cannot admit a pair. "
                "Set prefill_max_requests to at least 2 or leave it unset."
            )
        else:
            pass
        memory_budget = self.token_to_kv_pool_allocator.create_prefill_budget(
            self.tree_cache
        )
        self.reserve_request(memory_budget, req)
        self.reserve_request(memory_budget, req)
        pair_tokens = memory_budget.total_offset
        capacity_tokens = self.token_to_kv_pool_allocator.size
        if pair_tokens >= capacity_tokens:
            return (
                f"MiniMax Music 3 requires {pair_tokens} KV tokens for the full "
                f"CFG pair, above the usable {capacity_tokens - 1}-token budget. "
                "Raise engine.kv_cache_bytes or lower max_new_tokens."
            )
        else:
            return None

    def reserve_request(self, memory_budget: PrefillBudget, request: Req) -> None:
        """Full prompt and decode reservation for a row that cannot be replayed."""
        prompt_tokens = len(request.origin_input_ids)
        memory_budget.reserve(
            prompt_tokens,
            request.sampling_params.max_new_tokens,
            extra_tokens=self.token_to_kv_pool_allocator.request_slot_reserve(
                has_req_pool_slot=request.kv.req_pool_idx is not None,
                prefill_tokens=prompt_tokens,
            ),
        )

    def pair_admission_limit(
        self, queue: list[Req], running_batch: ScheduleBatch
    ) -> int:
        """Limit prefill to whole CFG pairs.

        Reserve running rows because retraction cannot replay Music3 audio feedback.
        """
        allocatable = int(self.get_num_allocatable_reqs(len(running_batch.reqs)))
        limit = min(len(queue), max(0, allocatable))
        prefill_max_requests = get_schedule().prefill_max_requests
        if prefill_max_requests is not None:
            limit = min(limit, max(0, prefill_max_requests))
        else:
            pass
        limit -= limit % 2
        budget = int(self.max_prefill_tokens)
        tokens = 0
        memory_budget = self.token_to_kv_pool_allocator.create_prefill_budget(
            self.tree_cache
        )
        for request in running_batch.reqs:
            if request.finished():
                continue
            else:
                pass
            memory_budget.reserve(
                0,
                max(
                    request.sampling_params.max_new_tokens - len(request.output_ids), 0
                ),
            )
        for index in range(0, limit, 2):
            pair = queue[index : index + 2]
            pair_tokens = sum(
                memory_budget.ceil_paged_tokens(len(request.origin_input_ids))
                for request in pair
            )
            if index and tokens + pair_tokens >= budget:
                limit = index
                break
            else:
                pass
            for request in pair:
                self.reserve_request(memory_budget, request)
            if not memory_budget.has_capacity():
                limit = index
                break
            else:
                pass
            tokens += pair_tokens
        return limit

    def stream_output(
        self,
        reqs: Iterable[Req],
        return_logprob: bool = False,
        skip_req: Req | None = None,
    ) -> None:
        conditioned = []
        for req in reqs:
            if not is_cfg_uncond_rid(req.rid):
                conditioned.append(req)
                continue
            else:
                pass
            if req.finished():
                # note (ruiling): closing the request forgets that its model path started, so the end is emitted first.
                self.emit_model_path_end_once(req.rid, status="success")
                self.close_completed_request(req)
            else:
                pass
        super().stream_output(conditioned, return_logprob, skip_req)

    def abort(self, request_id: str, *, defer_running_cleanup: bool = True) -> None:
        if self.scheduler_thread_id != get_ident() and (
            self.scheduler_thread_id is not None or self.enable_overlap
        ):
            if self.is_entry_rank:
                self.inbox.put(IncomingMessage(request_id=request_id, type="abort"))
            else:
                pass
            return
        else:
            pass
        super().abort(request_id, defer_running_cleanup=defer_running_cleanup)
        if is_cfg_uncond_rid(request_id):
            return
        else:
            pass
        super().abort(
            cfg_uncond_rid(request_id), defer_running_cleanup=defer_running_cleanup
        )


__all__ = ["MiniMaxMusic3Scheduler"]
