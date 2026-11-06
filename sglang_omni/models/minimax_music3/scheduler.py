# SPDX-License-Identifier: Apache-2.0
"""OmniScheduler specialization that keeps MiniMax Music 3 CFG rows paired."""

from __future__ import annotations

from collections.abc import Iterable

from sglang.srt.managers.schedule_batch import NextBatchPlan, Req, ScheduleBatch
from sglang.srt.mem_cache.prefill_budget import PrefillBudget

from sglang_omni.models.minimax_music3.serial_offload import get_coordinator
from sglang_omni.models.minimax_music3.sglang_request_builder import (
    MiniMaxMusic3SGLangRequestData,
    cfg_uncond_rid,
    is_cfg_uncond_rid,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


class MiniMaxMusic3Scheduler(OmniScheduler["MiniMaxMusic3SGLangRequestData"]):
    """Admit, decode and retire every request as a CFG row pair."""

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
        req._coalesce_enqueue_t = (
            cond_req._coalesce_enqueue_t
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        req._omni_terminal_claimed = False  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        req.omni_data = uncond  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        self.waiting_queue.append(req)

    def get_new_batch_prefill(self, running_batch: ScheduleBatch) -> NextBatchPlan:
        queue = self.waiting_queue
        limit = self.pair_admission_limit(queue, running_batch)
        input_budget = self.max_prefill_tokens
        if limit:
            # note (Codex): The first-input exception must cover both CFG rows.
            first_pair_tokens = sum(
                -(-len(request.origin_input_ids) // self.page_size) * self.page_size
                for request in queue[:2]
            )
            self.max_prefill_tokens = max(input_budget, first_pair_tokens + 1)
        else:
            pass
        deferred = queue[limit:]
        del queue[limit:]
        try:
            return super().get_new_batch_prefill(running_batch)
        finally:
            self.max_prefill_tokens = input_budget
            self.waiting_queue.extend(deferred)

    def request_kv_capacity_error(
        self, req: Req, *, session_capacity_checked: bool = False
    ) -> str | None:
        if session_capacity_checked:
            error = super().request_kv_capacity_error(
                req, session_capacity_checked=True
            )
        else:
            error = super().request_kv_capacity_error(req)
        if error is not None:
            return error
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
        """How many leading queue entries the adder may see, always whole pairs."""
        allocatable = int(self.get_num_allocatable_reqs(len(running_batch.reqs)))
        limit = min(len(queue), max(0, allocatable))
        limit -= limit % 2
        budget = int(self.max_prefill_tokens)
        tokens = 0
        memory_budget = self.token_to_kv_pool_allocator.create_prefill_budget(
            self.tree_cache
        )
        # note (Codex): Music3 cannot replay a retracted row's audio feedback.
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
        coordinator = get_coordinator()
        if coordinator.enabled:
            if limit < 2 or not coordinator.try_acquire_ar(queue[0].rid):
                return 0
            else:
                return 2
        else:
            pass
        return limit

    def stream_output(
        self,
        reqs: Iterable[Req],
        return_logprob: bool = False,
        skip_req: Req | None = None,
    ) -> None:
        conditioned = []
        for req in reqs:
            if not self.is_cfg_uncond(req):
                conditioned.append(req)
                continue
            else:
                pass
            if req.finished():
                self.close_completed_request(req)
            else:
                pass
        super().stream_output(conditioned, return_logprob, skip_req)

    def abort(self, request_id: str, *, defer_running_cleanup: bool = True) -> None:
        super().abort(request_id, defer_running_cleanup=defer_running_cleanup)
        if is_cfg_uncond_rid(request_id):
            return
        else:
            pass
        super().abort(
            cfg_uncond_rid(request_id), defer_running_cleanup=defer_running_cleanup
        )

    @staticmethod
    def is_cfg_uncond(req: Req) -> bool:
        data = getattr(
            req, "omni_data", None
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        return data is not None and data.is_cfg_uncond


__all__ = ["MiniMaxMusic3Scheduler"]
