# SPDX-License-Identifier: Apache-2.0
"""OmniScheduler specialization that keeps MiniMax Music 3 CFG rows paired."""

from __future__ import annotations

from typing import Any

from sglang_omni.scheduling.omni_scheduler import OmniScheduler

from .sglang_request_builder import cfg_uncond_rid, is_cfg_uncond_rid


class MiniMaxMusic3Scheduler(OmniScheduler):
    """Admit, decode and retire every request as a CFG row pair."""

    def _enqueue_built_request(
        self,
        payload: Any,
        pending_stream_done: bool,
        req_data: Any,
        *,
        request_admission_lock_held: bool = False,
    ) -> None:
        super()._enqueue_built_request(
            payload,
            pending_stream_done,
            req_data,
            request_admission_lock_held=request_admission_lock_held,
        )
        uncond = req_data.cfg_uncond
        if uncond is None:
            return
        if request_admission_lock_held:
            self._enqueue_cfg_uncond(req_data, uncond)
            return
        with self._request_admission_lock:
            self._enqueue_cfg_uncond(req_data, uncond)

    def _enqueue_cfg_uncond(self, req_data: Any, uncond: Any) -> None:
        cond_req = req_data.req
        if not self.waiting_queue or self.waiting_queue[-1] is not cond_req:
            return
        req = uncond.req
        self._normalize_req_token_arrays(req)
        req._coalesce_enqueue_t = cond_req._coalesce_enqueue_t
        req._omni_terminal_claimed = False
        req._omni_data = uncond
        self.waiting_queue.append(req)

    def get_new_batch_prefill(self, running_batch: Any) -> Any:
        queue = self.waiting_queue
        limit = self._pair_admission_limit(queue, running_batch)
        if limit >= len(queue):
            return super().get_new_batch_prefill(running_batch)
        deferred = queue[limit:]
        del queue[limit:]
        try:
            return super().get_new_batch_prefill(running_batch)
        finally:
            self.waiting_queue.extend(deferred)

    def _pair_admission_limit(self, queue: list, running_batch: Any) -> int:
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
                return index
            tokens += pair_tokens
        return limit

    def stream_output(
        self, reqs: Any, return_logprob: bool = False, skip_req: Any = None
    ) -> None:
        conditioned = []
        for req in reqs:
            if not self._is_cfg_uncond(req):
                conditioned.append(req)
                continue
            if req.finished():
                self._close_completed_request(req)
        super().stream_output(conditioned, return_logprob, skip_req)

    def abort(self, request_id: str, *, defer_running_cleanup: bool = True) -> None:
        super().abort(request_id, defer_running_cleanup=defer_running_cleanup)
        if is_cfg_uncond_rid(request_id):
            return
        super().abort(
            cfg_uncond_rid(request_id), defer_running_cleanup=defer_running_cleanup
        )

    @staticmethod
    def _is_cfg_uncond(req: Any) -> bool:
        data = getattr(req, "_omni_data", None)
        return data is not None and data.is_cfg_uncond


__all__ = ["MiniMaxMusic3Scheduler"]
