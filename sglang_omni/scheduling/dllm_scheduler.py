# SPDX-License-Identifier: Apache-2.0
"""DllmScheduler — stage-facing scheduler for Diffusion LLM stages.

Unlike OmniScheduler (which delegates to SGLang's AR scheduling methods),
DLLM scheduling uses its own DllmManager for batch construction and drives
ForwardBatch directly.  This class provides the same public contract
(inbox, outbox, start, stop, abort) so it is interchangeable from the
Stage's perspective.
"""

from __future__ import annotations

import logging
import queue as _queue_mod
import time
from typing import Any, Callable

import torch
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.sglang_backend.dllm import DllmManager

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

        self._running = False
        self._aborted_request_ids: set[str] = set()
        self._aborted_cleanup_pending: set[str] = set()
        self._rid_to_req_data: dict[str, Any] = {}

        self._manager = DllmManager(
            server_args=server_args,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=model_config,
            dllm_config=dllm_config,
        )

    def start(self) -> None:
        self._running = True
        self._event_loop()

    def event_loop(self) -> None:
        self.start()

    def stop(self) -> None:
        self._running = False

    def abort(self, request_id: str) -> None:
        self._aborted_request_ids.add(request_id)

    def _event_loop(self) -> None:
        while self._running:
            self._drain_inbox()
            self._purge_aborted()
            self._manager.filter_finished_reqs()
            batch = self._manager.schedule_next_batch()

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
            self._post_step_operations(batch)

    def _drain_inbox(self) -> None:
        while True:
            try:
                msg = self.inbox.get_nowait()
            except _queue_mod.Empty:
                break

            if msg.request_id in self._aborted_request_ids:
                continue

            if msg.type == "new_request":
                payload = msg.data
                req_data = self._request_builder(payload)
                req = req_data.req
                self._rid_to_req_data[req.rid] = req_data
                self._manager.add_waiting_reqs(req)
            else:
                logger.warning(
                    "DllmScheduler: unhandled message type %r for request %s",
                    msg.type,
                    msg.request_id,
                )

    def _purge_aborted(self) -> None:
        if self._aborted_cleanup_pending:
            self._aborted_request_ids -= self._aborted_cleanup_pending
            self._aborted_cleanup_pending = set()

        if not self._aborted_request_ids:
            return
        self._manager.waiting_queue = [
            r
            for r in self._manager.waiting_queue
            if r.rid not in self._aborted_request_ids
        ]
        self._manager.staging_queue = [
            r
            for r in self._manager.staging_queue
            if r.rid not in self._aborted_request_ids
        ]
        for rid in list(self._rid_to_req_data):
            if rid in self._aborted_request_ids:
                self._rid_to_req_data.pop(rid, None)

        self._aborted_cleanup_pending = set(self._aborted_request_ids)

    def _apply_results(self, batch: Any, batch_result: Any) -> None:
        next_token_ids_list = batch_result.next_token_ids

        for i, req in enumerate(batch.reqs):
            if req.rid in self._aborted_request_ids:
                continue

            if next_token_ids_list and i < len(next_token_ids_list):
                token_ids = next_token_ids_list[i]
                if isinstance(token_ids, torch.Tensor):
                    token_ids = token_ids.tolist()
            else:
                token_ids = []

            if token_ids:
                req.output_ids.extend(token_ids)
                req.check_finished(new_accepted_len=len(token_ids))

            req_data = self._rid_to_req_data.get(req.rid)
            if req_data is None:
                continue

            if req.finished():
                self._rid_to_req_data.pop(req.rid, None)
                req_data.output_ids = list(req.output_ids_through_stop)
                finished_reason = req.finished_reason
                req_data.finish_reason = (
                    finished_reason.to_json().get("type")
                    if finished_reason is not None
                    else None
                )
                result_payload = self._result_adapter(req_data)
                self.outbox.put(
                    OutgoingMessage(
                        request_id=req.rid,
                        type="result",
                        data=result_payload,
                    )
                )

    def _post_step_operations(self, batch: Any) -> None:
        for req in batch.reqs:
            if req.finished():
                release_kv_cache(req, self.tree_cache)

        chunked_req_to_exclude = set()
        if self._manager.any_staging_reqs():
            for req in self._manager.staging_queue:
                if req.finished():
                    continue
                chunked_req_to_exclude.add(req)
                self._manager.tree_cache.cache_unfinished_req(req, chunked=True)
                if req.req_pool_idx is not None:
                    self._manager.req_to_token_pool.free(req.req_pool_idx)

        batch.filter_batch(chunked_req_to_exclude=list(chunked_req_to_exclude))
