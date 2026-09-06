# SPDX-License-Identifier: Apache-2.0
"""OmniScheduler specialization that keeps MiniMax Music 3 CFG rows paired."""

from __future__ import annotations

import logging
from typing import Any

from sglang.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS
from sglang.srt.managers.scheduler import TEST_RETRACT, TEST_RETRACT_INTERVAL
from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
    NewTokenRatioTracker,
)

from sglang_omni.scheduling.omni_scheduler import OmniScheduler

from .sglang_request_builder import cfg_uncond_rid, is_cfg_uncond_rid

logger = logging.getLogger(__name__)


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
        prefill_budget = self.max_prefill_tokens
        expanded_pair_budget = False
        if len(queue) >= 2:
            assert queue[0]._omni_data.cfg_uncond is queue[1]._omni_data
            assert queue[0].is_retracted == queue[1].is_retracted
            page_size = int(self.page_size)
            pair_input_tokens = 0
            for req in queue[:2]:
                input_length = len(req.origin_input_ids) + len(req.output_ids)
                pair_input_tokens += -(-input_length // page_size) * page_size
            if pair_input_tokens >= prefill_budget:
                self.max_prefill_tokens = pair_input_tokens + 1
                expanded_pair_budget = True

        limit = self._pair_admission_limit(queue, running_batch)
        if expanded_pair_budget:
            limit = min(limit, 2)
        elif limit >= len(queue):
            return super().get_new_batch_prefill(running_batch)
        deferred = queue[limit:]
        del queue[limit:]
        try:
            return super().get_new_batch_prefill(running_batch)
        finally:
            self.waiting_queue.extend(deferred)
            self.max_prefill_tokens = prefill_budget

    def _pair_admission_limit(self, queue: list, running_batch: Any) -> int:
        """How many leading queue entries the adder may see, always whole pairs."""
        allocatable = int(self.get_num_allocatable_reqs(len(running_batch.reqs)))
        limit = min(len(queue), max(0, allocatable))
        limit -= limit % 2

        page_size = int(self.page_size)
        remaining_input_tokens = int(self.max_prefill_tokens)
        running_token_reserve = sum(
            min(
                req.sampling_params.max_new_tokens - len(req.output_ids),
                CLIP_MAX_NEW_TOKENS,
            )
            * self.new_token_ratio_tracker.current
            for req in running_batch.reqs
        )
        remaining_total_tokens = (
            self.token_to_kv_pool_allocator.available_size()
            + self.tree_cache.evictable_size()
            - running_token_reserve
        )
        for index in range(0, limit, 2):
            cond, uncond = queue[index : index + 2]
            pair_input_tokens = 0
            pair_total_tokens = 0
            for req in (cond, uncond):
                input_length = len(req.origin_input_ids) + len(req.output_ids)
                input_tokens = -(-input_length // page_size) * page_size
                new_tokens = min(
                    req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS
                )
                pair_input_tokens += input_tokens
                pair_total_tokens += input_tokens + new_tokens + page_size
            if (
                pair_input_tokens >= remaining_input_tokens
                or pair_total_tokens >= remaining_total_tokens
            ):
                return index
            remaining_input_tokens -= pair_input_tokens
            remaining_total_tokens -= pair_total_tokens
        return limit

    def update_running_batch(self, batch: Any) -> Any:
        """Apply SGLang's decode update to complete CFG pairs."""
        initial_size = len(batch.reqs)
        batch.filter_batch()
        if batch.is_empty():
            batch.batch_is_full = False
            return batch

        if self.enable_hierarchical_cache:
            self.tree_cache.flush_write_through_acks()

        kv_cache_full = not batch.check_decode_mem()
        test_retraction = TEST_RETRACT and self.forward_ct % TEST_RETRACT_INTERVAL == 0
        if kv_cache_full or test_retraction:
            old_available_tokens = self.token_to_kv_pool_allocator.available_size()
            old_ratio = self.new_token_ratio_tracker.current
            retracted_pairs, aborted_pair = self._retract_decode_pairs(batch)
            retracted_reqs = [req for pair in retracted_pairs for req in pair]
            new_available_tokens = self.token_to_kv_pool_allocator.available_size()

            self.metrics_reporter.num_retracted_reqs = len(retracted_reqs)
            if self.metrics_reporter.enable_metrics and retracted_reqs:
                self.metrics_reporter.metrics_collector.increment_retracted_reqs(
                    num_retracted_reqs=len(retracted_reqs),
                    num_retracted_input_tokens=sum(
                        len(req.origin_input_ids) for req in retracted_reqs
                    ),
                    num_retracted_output_tokens=sum(
                        len(req.output_ids) for req in retracted_reqs
                    ),
                )
            self.new_token_ratio_tracker.current = (
                NewTokenRatioTracker.estimate_new_token_ratio_after_retract(batch.reqs)
            )

            message = (
                "KV cache pool is full. Retract requests. "
                if kv_cache_full
                else "Testing retraction. "
            )
            details = (
                f"#retracted_reqs: {len(retracted_reqs)}, "
                f"#new_tokens_gained: {new_available_tokens - old_available_tokens}"
            )
            if kv_cache_full:
                details += (
                    f", #new_token_ratio: {old_ratio:.4f} -> "
                    f"{self.new_token_ratio_tracker.current:.4f}"
                )
            logger.warning(message + details)

            for cond, uncond in retracted_pairs:
                self._add_request_to_queue(cond, is_retracted=True)
                self._add_request_to_queue(uncond, is_retracted=True)

            if aborted_pair is not None:
                cond, uncond = aborted_pair
                error = RuntimeError(
                    "MiniMax Music 3 request cannot allocate its next KV-cache "
                    "page after all other CFG pairs were retracted"
                )
                self._emit_request_error(cond.rid, error)
                self.abort(cond.rid, defer_running_cleanup=False)
                cond._omni_data = None
                uncond._omni_data = None
        else:
            self.new_token_ratio_tracker.decay_step()

        if len(batch.reqs) < initial_size:
            batch.batch_is_full = False
        if not batch.is_empty():
            batch.prepare_for_decode()
        return batch

    def _retract_decode_pairs(
        self, batch: Any
    ) -> tuple[list[tuple[Any, Any]], tuple[Any, Any] | None]:
        assert len(batch.reqs) >= 2 and len(batch.reqs) % 2 == 0
        for cond, uncond in zip(batch.reqs[0::2], batch.reqs[1::2], strict=True):
            assert cond._omni_data.cfg_uncond is uncond._omni_data
            assert len(cond.output_ids) == len(uncond.output_ids)

        row_order = batch._get_decode_retraction_order(
            batch.reqs,
            self.server_args,
            allow_policy_sort=(
                batch.spec_algorithm is None or batch.spec_algorithm.is_none()
            ),
        )
        pair_order = []
        seen_pairs = set()
        for row_index in row_order:
            pair_index = row_index // 2
            if pair_index not in seen_pairs:
                seen_pairs.add(pair_index)
                pair_order.append(pair_index)

        keep_indices = list(range(len(batch.reqs)))
        retracted_pairs = []
        first_iteration = True
        while first_iteration or not batch.check_decode_mem(
            selected_indices=keep_indices
        ):
            if len(pair_order) == 1:
                break
            first_iteration = False
            pair_index = pair_order.pop()
            row_indices = (2 * pair_index, 2 * pair_index + 1)
            retracted_pairs.append(
                (batch.reqs[row_indices[0]], batch.reqs[row_indices[1]])
            )
            keep_indices = [index for index in keep_indices if index not in row_indices]
            for offset, row_index in enumerate(row_indices):
                remaining_reqs = len(keep_indices) + 1 - offset
                batch.release_req(row_index, remaining_reqs, self.server_args)

        aborted_pair = None
        if not batch.check_decode_mem(selected_indices=keep_indices):
            assert len(pair_order) == 1
            pair_index = pair_order.pop()
            row_indices = (2 * pair_index, 2 * pair_index + 1)
            aborted_pair = (
                batch.reqs[row_indices[0]],
                batch.reqs[row_indices[1]],
            )
            keep_indices = []
            for remaining_reqs, row_index in zip((1, 0), row_indices, strict=True):
                batch.release_req(row_index, remaining_reqs, self.server_args)

        batch.filter_batch(keep_indices=keep_indices)
        return retracted_pairs, aborted_pair

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
