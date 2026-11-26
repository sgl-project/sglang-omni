# SPDX-License-Identifier: Apache-2.0
"""Schedule both CFG rows of each Music3 request as one song."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from threading import get_ident
from typing import Literal

from sglang.srt.managers.schedule_batch import NextBatchPlan, Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import CacheAgnosticPolicy, SchedulePolicy
from sglang.srt.mem_cache.prefill_budget import PrefillBudget
from sglang.srt.runtime_context import get_schedule
from typing_extensions import Unpack

from sglang_omni.models.minimax_music3.serial_offload import get_coordinator
from sglang_omni.models.minimax_music3.sglang_request_builder import (
    MiniMaxMusic3SGLangRequestData,
    cfg_primary_rid,
    cfg_uncond_rid,
    is_cfg_uncond_rid,
)
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.omni_scheduler import (
    AdminActionResult,
    OmniScheduler,
    OmniSchedulerArguments,
)

logger = logging.getLogger(__name__)


def has_complete_cfg_pairs(rows: list[Req]) -> bool:
    return len(rows) % 2 == 0 and all(
        not is_cfg_uncond_rid(rows[index].rid)
        and rows[index + 1].rid == cfg_uncond_rid(rows[index].rid)
        for index in range(0, len(rows), 2)
    )


def order_cfg_rows(waiting_queue: list[Req]) -> list[Req]:
    # note (ruiling): row-based policies can separate CFG twins.
    rows_by_id = {row.rid: row for row in waiting_queue}
    ordered_rows: list[Req] = []
    for row in waiting_queue:
        if is_cfg_uncond_rid(row.rid) and cfg_primary_rid(row.rid) in rows_by_id:
            continue
        else:
            ordered_rows.append(row)
        twin = rows_by_id.get(cfg_uncond_rid(row.rid))
        if twin is not None:
            ordered_rows.append(twin)
        else:
            pass
    return ordered_rows


@dataclass(kw_only=True)
class PrefillSelection:
    deferred_rows: list[Req] = field(default_factory=list)
    new_owner_request_id: str | None = None


class MiniMaxMusic3SchedulePolicy(SchedulePolicy):
    """Keep the order of primary rows chosen by the configured policy."""

    def __init__(self, scheduler: MiniMaxMusic3Scheduler) -> None:
        original_policy = scheduler.policy
        super().__init__(
            scheduler.schedule_policy,
            original_policy.tree_cache,
            original_policy.enable_hierarchical_cache,
            original_policy.enable_priority_scheduling,
            original_policy.schedule_low_priority_values_first,
        )
        self.scheduler: MiniMaxMusic3Scheduler = scheduler

    def calc_priority(
        self,
        waiting_queue: list[Req],
        running_batch: ScheduleBatch | None = None,
        processed_tokens: int = 0,
    ) -> None:
        super().calc_priority(waiting_queue, running_batch, processed_tokens)
        policy_can_reorder = (
            self.policy != CacheAgnosticPolicy.FCFS or self.enable_priority_scheduling
        )
        if policy_can_reorder:
            waiting_queue[:] = order_cfg_rows(waiting_queue)
        else:
            pass
        if running_batch is not None and self.scheduler.prefill_selection is not None:
            self.scheduler.select_prefill_rows(waiting_queue, running_batch)
        else:
            pass


class MiniMaxMusic3Scheduler(OmniScheduler[MiniMaxMusic3SGLangRequestData]):
    def __init__(
        self, **kwargs: Unpack[OmniSchedulerArguments[MiniMaxMusic3SGLangRequestData]]
    ) -> None:
        super().__init__(**kwargs)
        self.policy = MiniMaxMusic3SchedulePolicy(self)
        self.prefill_selection: PrefillSelection | None = None
        self.enable_priority_preemption = False

    def built_request_rows(
        self, request_data: MiniMaxMusic3SGLangRequestData
    ) -> tuple[MiniMaxMusic3SGLangRequestData, ...]:
        twin = request_data.cfg_uncond
        if twin is None:
            raise ValueError(
                "MiniMax Music 3 needs both CFG rows before queue admission."
            )
        else:
            return request_data, twin

    def get_new_batch_prefill(self, running_batch: ScheduleBatch) -> NextBatchPlan:
        if not has_complete_cfg_pairs(self.waiting_queue):
            row_ids = {row.rid for row in self.waiting_queue}
            request_ids = dict.fromkeys(
                cfg_primary_rid(row.rid) for row in self.waiting_queue
            )
            for request_id in request_ids:
                if (
                    request_id not in row_ids
                    or cfg_uncond_rid(request_id) not in row_ids
                ):
                    self.emit_request_error(
                        request_id,
                        RuntimeError(
                            "MiniMax Music 3 found an incomplete CFG pair in the waiting queue."
                        ),
                    )
                    self.abort(request_id)
                else:
                    pass
            self.waiting_queue[:] = order_cfg_rows(self.waiting_queue)
        else:
            pass

        input_budget_tokens = self.max_prefill_tokens
        selection = PrefillSelection()
        self.prefill_selection = selection
        plan: NextBatchPlan | None = None
        invalid_batch: ScheduleBatch | None = None
        try:
            # A failed native allocation can leave slots or KV partially written.
            plan = super().get_new_batch_prefill(running_batch)
            if plan.batch_to_run is not None and not has_complete_cfg_pairs(
                plan.batch_to_run.reqs
            ):
                invalid_batch = plan.batch_to_run
                plan.running_batch.batch_is_full = False
                plan = NextBatchPlan(
                    batch_to_run=None, running_batch=plan.running_batch
                )
            else:
                pass
        finally:
            self.prefill_selection = None
            self.max_prefill_tokens = input_budget_tokens
            if selection.deferred_rows:
                self.waiting_queue.extend(selection.deferred_rows)
            else:
                pass
            owner = selection.new_owner_request_id
            batch = plan.batch_to_run if plan is not None else None
            if owner is not None and (
                batch is None or not batch.reqs or batch.reqs[0].rid != owner
            ):
                get_coordinator().cancel_ar(owner)
            else:
                pass

        if invalid_batch is not None:
            self.handle_batch_failure(
                invalid_batch,
                RuntimeError(
                    "MiniMax Music 3 prefill admitted an incomplete CFG pair."
                ),
            )
        else:
            pass
        return plan

    def select_prefill_rows(
        self, waiting_queue: list[Req], running_batch: ScheduleBatch
    ) -> None:
        selection = self.prefill_selection
        assert selection is not None
        admitted_rows = self.pair_admission_limit(waiting_queue, running_batch)
        coordinator = get_coordinator()
        if coordinator.enabled and admitted_rows:
            request_id = waiting_queue[0].rid
            previous_owner = coordinator.owner_request_id
            if coordinator.try_acquire_ar(request_id):
                admitted_rows = 2
                if previous_owner != request_id:
                    selection.new_owner_request_id = request_id
                else:
                    pass
            else:
                admitted_rows = 0
        else:
            pass
        if admitted_rows:
            first_pair_tokens = sum(
                -(-len(row.origin_input_ids) // self.page_size) * self.page_size
                for row in waiting_queue[:2]
            )
            # The adder allows the first row to exceed the input budget.
            self.max_prefill_tokens = max(
                self.max_prefill_tokens, first_pair_tokens + 1
            )
        else:
            pass
        if admitted_rows < len(waiting_queue):
            selection.deferred_rows = waiting_queue[admitted_rows:]
            del waiting_queue[admitted_rows:]
        else:
            pass

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

    def update_running_batch(self, batch: ScheduleBatch) -> ScheduleBatch | None:
        previous_row_count = batch.batch_size()
        finishing_rows = self._reqs_finishing_inflight(
            batch
        )  # noqa: leading-underscore  # scheduler hook
        batch.filter_batch(reqs_to_exclude=finishing_rows)
        if batch.batch_size() < previous_row_count:
            batch.batch_is_full = False
        else:
            pass
        if batch.is_empty():
            batch.batch_is_full = False
            return batch
        elif not has_complete_cfg_pairs(batch.reqs):
            error = RuntimeError(
                "MiniMax Music 3 found an incomplete CFG pair in the decode batch."
            )
        elif batch.check_decode_mem():
            self.new_token_ratio_tracker.decay_step()
            batch.prepare_for_decode()
            return batch
        elif finishing_rows:
            batch.batch_is_full = False
            return None
        else:
            error = RuntimeError(
                "MiniMax Music 3 ran out of KV capacity during decode."
            )
        self.handle_batch_failure(batch, error)
        batch.filter_batch()
        batch.batch_is_full = False
        return batch

    def admin_pause_generation(self, payload: dict[str, object]) -> AdminActionResult:
        if payload.get("mode") == "retract":
            message = (
                "MiniMax Music 3 cannot replay audio feedback. Use in_place or abort."
            )
            return {"success": False, "message": message, "error": message}
        else:
            return super().admin_pause_generation(payload)

    def handle_batch_failure(self, batch: ScheduleBatch, error: Exception) -> None:
        request_ids = dict.fromkeys(cfg_primary_rid(row.rid) for row in batch.reqs)
        logger.error(
            f"MiniMax Music 3 batch failed for {list(request_ids)}: {error}",
            exc_info=error if error.__traceback__ is not None else None,
        )
        for row in batch.reqs:
            row.skip_radix_cache_insert = True
            self.emit_model_path_end_once(row.rid, status="error")
        for request_id in request_ids:
            self.abort_cfg_pair(
                request_id, defer_running_cleanup=False, batch_to_abort=batch
            )
            self.emit_request_error(request_id, error)

    def request_id_for_row(self, row_id: str) -> str:
        return cfg_primary_rid(row_id)

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
        if is_cfg_uncond_rid(request_id):
            # Public cancellations can carry a rejected id with the reserved suffix.
            return
        else:
            pass
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

        self.abort_cfg_pair(request_id, defer_running_cleanup=defer_running_cleanup)

    def abort_cfg_pair(
        self,
        request_id: str,
        *,
        defer_running_cleanup: bool,
        batch_to_abort: ScheduleBatch | None = None,
    ) -> None:
        for row_id in (request_id, cfg_uncond_rid(request_id)):
            super().abort(
                row_id,
                defer_running_cleanup=defer_running_cleanup,
                batch_to_abort=batch_to_abort,
            )


__all__ = ["MiniMaxMusic3Scheduler"]
