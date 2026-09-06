# SPDX-License-Identifier: Apache-2.0
"""Atomic multi-request CFG admission for Breeze-TTS-2.

Every logical request owns adjacent conditional/unconditional rows. Admission
exposes only complete pairs to SGLang and reserves enough KV for every configured
row's bounded lifetime, so continuous batching never splits or retracts a pair.
"""

from sglang.srt.managers.schedule_batch import NextBatchPlan

from sglang_omni.scheduling.omni_scheduler import OmniScheduler

from .frontend import CONTEXT_LENGTH
from .request_builders import CFG_SUFFIX


class BreezeScheduler(OmniScheduler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.max_running_requests < 2 or self.max_running_requests % 2:
            raise ValueError(
                "Breeze max_running_requests must contain complete CFG row pairs"
            )
        # PrefillAdder page-rounds every prompt and reserves another page per
        # row. The final +1 covers its strict `needed >= remaining` rejection.
        needed = self.max_running_requests * (CONTEXT_LENGTH + 2 * self.page_size) + 1
        available = self.token_to_kv_pool_allocator.available_size()
        if available < needed:
            raise ValueError(
                "Breeze needs at least "
                f"{needed} KV token slots for {self.max_running_requests // 2} "
                f"atomic CFG pairs, but only {available} are available"
            )

    def _enqueue_built_request(
        self,
        payload,
        pending_stream_done,
        req_data,
        *,
        request_admission_lock_held=False,
    ):
        if not request_admission_lock_held:
            with self._request_admission_lock:
                return self._enqueue_built_request(
                    payload,
                    pending_stream_done,
                    req_data,
                    request_admission_lock_held=True,
                )
        super()._enqueue_built_request(
            payload, pending_stream_done, req_data, request_admission_lock_held=True
        )
        if not self.waiting_queue or self.waiting_queue[-1] is not req_data.req:
            return
        twin = req_data.cfg_uncond
        req = twin.req
        self._normalize_req_token_arrays(req)
        req._coalesce_enqueue_t = req_data.req._coalesce_enqueue_t
        req._omni_terminal_claimed = False
        req._omni_data = twin
        self.waiting_queue.append(req)

    def get_new_batch_prefill(self, running_batch):
        with self._request_admission_lock:
            return self._get_new_pair(running_batch)

    def _get_new_pair(self, running_batch):
        if not self.waiting_queue:
            return NextBatchPlan(batch_to_run=None, running_batch=running_batch)
        limit = self._pair_admission_limit(running_batch)
        if limit == 0:
            return NextBatchPlan(batch_to_run=None, running_batch=running_batch)

        queue = self.waiting_queue
        original = list(queue)
        deferred = queue[limit:]
        del queue[limit:]
        try:
            plan = super().get_new_batch_prefill(running_batch)
        except Exception:
            self.waiting_queue = original
            raise
        # Upstream replaces self.waiting_queue after removing admitted rows;
        # append to that new list rather than the temporary exposed prefix.
        self.waiting_queue.extend(deferred)

        batch = plan.batch_to_run
        if batch is not None:
            if len(batch.reqs) % 2:
                raise RuntimeError("Breeze CFG pair was split by prefill admission")
            self._validate_pairs(batch.reqs)
        return plan

    def _pair_admission_limit(self, running_batch):
        """Return a prefix containing only pairs SGLang can admit atomically."""
        queue = self.waiting_queue
        if len(queue) % 2:
            raise RuntimeError("Breeze admission found an incomplete CFG pair")
        self._validate_pairs(queue)

        allocatable = int(self.get_num_allocatable_reqs(len(running_batch.reqs)))
        limit = min(len(queue), max(0, allocatable))
        limit -= limit % 2

        # Without chunking, PrefillAdder rounds each input to a page and rejects
        # every non-first row whose input is >= the remaining prompt budget.
        # Applying the same strict comparison to each complete pair prevents it
        # from accepting the conditional row and deferring its twin.
        budget = int(self.max_prefill_tokens)
        tokens = 0
        for index in range(0, limit, 2):
            pair_tokens = sum(
                self._paged_input_tokens(req) for req in queue[index : index + 2]
            )
            if tokens + pair_tokens >= budget:
                return index
            tokens += pair_tokens
        return limit

    def _paged_input_tokens(self, req):
        tokens = len(req.origin_input_ids)
        return -(-tokens // self.page_size) * self.page_size

    @staticmethod
    def _validate_pairs(reqs):
        for index in range(0, len(reqs), 2):
            cond, uncond = reqs[index : index + 2]
            data = getattr(cond, "_omni_data", None)
            twin = getattr(uncond, "_omni_data", None)
            if (
                twin is None
                or not uncond.rid.endswith(CFG_SUFFIX)
                or getattr(data, "cfg_uncond", None) is not twin
            ):
                raise RuntimeError("Breeze CFG rows are not adjacent pairs")

    def stream_output(self, reqs, return_logprob=False, skip_req=None):
        conditioned = []
        for req in reqs:
            if req.rid.endswith(CFG_SUFFIX):
                if req.finished():
                    self._close_completed_request(req)
            else:
                conditioned.append(req)
        super().stream_output(conditioned, return_logprob, skip_req)

    def abort(self, request_id, *, defer_running_cleanup=True):
        parent = request_id.removesuffix(CFG_SUFFIX)
        with self._request_admission_lock:
            for rid in (parent, parent + CFG_SUFFIX):
                super().abort(rid, defer_running_cleanup=defer_running_cleanup)
