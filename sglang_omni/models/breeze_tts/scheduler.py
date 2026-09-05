# SPDX-License-Identifier: Apache-2.0
"""Atomic CFG admission, modeled on the MiniMax Music 3 scheduler.

Only one logical request runs at a time in this baseline. Reserving enough KV
for both complete branches avoids splitting a CFG pair or retracting feedback
state. Preprocessing and vocoder stages still overlap AR execution.
"""

from sglang_omni.scheduling.omni_scheduler import OmniScheduler

from .frontend import CONTEXT_LENGTH
from .request_builders import CFG_SUFFIX


class BreezeScheduler(OmniScheduler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        needed = 2 * (CONTEXT_LENGTH + self.page_size)
        if self.token_to_kv_pool_allocator.available_size() < needed:
            raise ValueError(
                f"Breeze needs at least {needed} KV token slots for an atomic CFG pair"
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
        if running_batch.reqs or not self.waiting_queue:
            return None
        if len(self.waiting_queue) < 2:
            raise RuntimeError("Breeze admission found an incomplete CFG pair")
        # Upstream must see only the first complete pair, never another prompt
        # mixed into the two rows reserved for conditional/unconditional decode.
        deferred = self.waiting_queue[2:]
        del self.waiting_queue[2:]
        try:
            batch = super().get_new_batch_prefill(running_batch)
            if batch is not None and len(batch.reqs) != 2:
                raise RuntimeError("Breeze CFG pair was split by prefill admission")
            return batch
        finally:
            self.waiting_queue.extend(deferred)

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
