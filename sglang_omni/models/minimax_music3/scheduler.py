# SPDX-License-Identifier: Apache-2.0
"""OmniScheduler specialization that keeps MiniMax Music 3 CFG rows paired."""

from __future__ import annotations

from collections.abc import Iterable

from sglang.srt.managers.schedule_batch import NextBatchPlan, Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import SchedulePolicy
from typing_extensions import Unpack

from sglang_omni.models.minimax_music3.serial_offload import get_coordinator
from sglang_omni.models.minimax_music3.sglang_request_builder import (
    MiniMaxMusic3SGLangRequestData,
    cfg_uncond_rid,
    is_cfg_uncond_rid,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler, OmniSchedulerArguments


class MiniMaxMusic3SchedulePolicy(SchedulePolicy):
    """Order the waiting queue by the configured policy, keeping each CFG pair together."""

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


class MiniMaxMusic3Scheduler(OmniScheduler["MiniMaxMusic3SGLangRequestData"]):
    """Admit, decode and retire every request as a CFG row pair."""

    def __init__(
        self,
        **kwargs: Unpack[OmniSchedulerArguments[MiniMaxMusic3SGLangRequestData]],
    ) -> None:
        super().__init__(**kwargs)
        policy = self.policy
        self.policy = MiniMaxMusic3SchedulePolicy(
            self.schedule_policy,
            policy.tree_cache,
            policy.enable_hierarchical_cache,
            policy.enable_priority_scheduling,
            policy.schedule_low_priority_values_first,
        )

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
        req.time_stats.wait_queue_entry_time = cond_req.time_stats.wait_queue_entry_time
        req._coalesce_enqueue_t = (
            cond_req._coalesce_enqueue_t
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        req._omni_terminal_claimed = False  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        req.omni_data = uncond  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        self.waiting_queue.append(req)

    def get_new_batch_prefill(self, running_batch: ScheduleBatch) -> NextBatchPlan:
        queue = self.waiting_queue
        limit = self.pair_admission_limit(queue, running_batch)
        if limit >= len(queue):
            return super().get_new_batch_prefill(running_batch)
        else:
            pass
        deferred = queue[limit:]
        del queue[limit:]
        try:
            return super().get_new_batch_prefill(running_batch)
        finally:
            self.waiting_queue.extend(deferred)

    def pair_admission_limit(
        self, queue: list[Req], running_batch: ScheduleBatch
    ) -> int:
        """How many leading queue entries the adder may see, always whole pairs."""
        allocatable = int(self.get_num_allocatable_reqs(len(running_batch.reqs)))
        limit = min(len(queue), max(0, allocatable))
        limit -= limit % 2
        budget = int(self.max_prefill_tokens)
        tokens = 0
        for index in range(0, limit, 2):
            pair_tokens = len(queue[index].origin_input_ids) + len(
                queue[index + 1].origin_input_ids
            )
            if index and tokens + pair_tokens > budget:
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
                # note (ruiling): closing the request forgets that its model path started, so the end is emitted first.
                self.emit_model_path_end_once(req.rid, status="success")
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
