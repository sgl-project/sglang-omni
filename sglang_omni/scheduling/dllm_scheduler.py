# SPDX-License-Identifier: Apache-2.0
"""DllmScheduler — stage-facing scheduler for Diffusion LLM stages.

Provides the same public contract (inbox, outbox, start, stop, abort)
as OmniScheduler so it is interchangeable from the Stage's perspective.
"""

from __future__ import annotations

import logging
import queue as _queue_mod
import threading
import time
from typing import Any, Callable

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)


class DllmScheduler:
    """Stage-facing scheduler for Diffusion LLM stages.

    Public contract (used by Stage):
        ``inbox``, ``outbox``, ``start()``, ``stop()``, ``abort(request_id)``
    """

    def __init__(
        self,
        tp_worker: Any,
        tree_cache: Any,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        server_args: Any,
        model_config: Any,
        dllm_config: Any,
        *,
        request_builder: Callable,
        result_adapter: Callable,
    ):
        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()

        self._request_builder = request_builder
        self._result_adapter = result_adapter

        self.tp_worker = tp_worker
        self.tree_cache = tree_cache
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.server_args = server_args
        self.model_config = model_config
        self.dllm_config = dllm_config

        self._running = False
        self._abort_lock = threading.Lock()
        self._aborted_request_ids: set[str] = set()
        self._rid_to_req_data: dict[str, Any] = {}
        self._waiting_queue: list[Req] = []
        self._staging_queue: list[Req] = []

    def start(self) -> None:
        self._running = True
        self._event_loop()

    def event_loop(self) -> None:
        self.start()

    def stop(self) -> None:
        self._running = False

    def abort(self, request_id: str) -> None:
        with self._abort_lock:
            self._aborted_request_ids.add(request_id)

    def _event_loop(self) -> None:
        while self._running:
            self._drain_and_purge()
            batch = self._schedule_next_batch()

            if batch is None:
                time.sleep(0.001)
                continue

            model_worker_batch = batch.get_model_worker_batch()
            forward_batch = ForwardBatch.init_new(
                model_worker_batch, self.tp_worker.model_runner
            )
            batch_result = self.tp_worker.forward_batch_generation(forward_batch)

            batch.output_ids = batch_result.next_token_ids
            self._apply_results(batch, batch_result)
            self._post_step(batch)

    def _drain_and_purge(self) -> None:
        with self._abort_lock:
            aborted = self._aborted_request_ids
            self._aborted_request_ids = set()

        while True:
            try:
                msg = self.inbox.get_nowait()
            except _queue_mod.Empty:
                break

            if msg.request_id in aborted:
                continue

            if msg.type == "new_request":
                req_data = self._request_builder(msg.data)
                req = req_data.req
                self._rid_to_req_data[req.rid] = req_data
                self._waiting_queue.append(req)
            else:
                logger.warning(
                    "DllmScheduler: unhandled message type %r for request %s",
                    msg.type,
                    msg.request_id,
                )

        self._waiting_queue = [
            r for r in self._waiting_queue if r.rid not in aborted and not r.finished()
        ]
        new_staging = []
        for req in self._staging_queue:
            if req.rid in aborted:
                release_kv_cache(req, self.tree_cache)
            elif not req.finished():
                new_staging.append(req)
        self._staging_queue = new_staging

        for rid in aborted:
            self._rid_to_req_data.pop(rid, None)

    def _schedule_next_batch(self) -> ScheduleBatch | None:
        if not self._waiting_queue and not self._staging_queue:
            return None

        adder = PrefillAdder(
            self.server_args.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            None,  # running_batch
            0.5,  # new_token_ratio
            self.server_args.max_prefill_tokens,
            self.server_args.chunked_prefill_size,
            dllm_config=self.dllm_config,
        )

        # Re-submit existing staging (chunked) requests.
        for req in self._staging_queue:
            req.init_next_round_input()
            adder.add_chunked_req(req)

        # Add new waiting requests.
        for req in self._waiting_queue:
            req.init_next_round_input(self.tree_cache)
            if (
                adder.add_one_req(
                    req,
                    has_chunked_req=bool(self._staging_queue),
                    truncation_align_size=None,
                )
                != AddReqResult.CONTINUE
            ):
                break

        if not adder.can_run_list:
            return None

        # Promote newly staged requests and remove them from waiting queue.
        staging_rids = {r.rid for r in self._staging_queue}
        for req in adder.dllm_staging_reqs:
            if req.rid not in staging_rids:
                self._staging_queue.append(req)
                staging_rids.add(req.rid)
        self._waiting_queue = [
            r for r in self._waiting_queue if r.rid not in staging_rids
        ]

        for req in self._staging_queue:
            req.is_chunked += 1

        new_batch = ScheduleBatch.init_new(
            reqs=adder.can_run_list,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            dllm_config=self.dllm_config,
        )
        new_batch.prepare_for_extend()
        return new_batch

    def _apply_results(self, batch: Any, batch_result: Any) -> None:
        next_token_ids_list = batch_result.next_token_ids

        if not next_token_ids_list:
            return

        for i, req in enumerate(batch.reqs):
            token_ids = next_token_ids_list[i].tolist()

            if token_ids:
                req.output_ids.extend(token_ids)
                req.check_finished(new_accepted_len=len(token_ids))

            if req.finished():
                req_data = self._rid_to_req_data.pop(req.rid, None)
                if req_data is None:
                    continue
                req_data.output_ids = list(req.output_ids_through_stop)
                finished_reason = req.finished_reason
                req_data.finish_reason = (
                    finished_reason.to_json().get("type")
                    if finished_reason is not None
                    else None
                )
                self.outbox.put(
                    OutgoingMessage(
                        request_id=req.rid,
                        type="result",
                        data=self._result_adapter(req_data),
                    )
                )

    def _post_step(self, batch: Any) -> None:
        exclude = set()
        for req in batch.reqs:
            if req.finished():
                release_kv_cache(req, self.tree_cache)
                exclude.add(req)

        new_staging = []
        for req in self._staging_queue:
            exclude.add(req)
            if req.finished():
                continue
            self.tree_cache.cache_unfinished_req(req, chunked=True)
            if req.req_pool_idx is not None:
                self.req_to_token_pool.free(req.req_pool_idx)
            new_staging.append(req)
        self._staging_queue = new_staging

        batch.filter_batch(chunked_req_to_exclude=list(exclude))
