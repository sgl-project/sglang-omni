# SPDX-License-Identifier: Apache-2.0
"""Fish-specific stage-facing scheduler.

Uses the old Fish request lifecycle semantics:
- model runner owns model-side decode and persistent buffers
- scheduler update/finish logic is Fish-step aware rather than generic token based
"""

from __future__ import annotations

import logging
import queue as _queue_mod
import time
from collections import deque
from typing import Any

from sglang.srt.mem_cache.common import release_kv_cache

from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.types import (
    ModelRunnerOutput,
    SchedulerOutput,
    SchedulerRequest,
    SchedulerStatus,
)

logger = logging.getLogger(__name__)


class FishBatchPlanner:
    """SGLang-backed batch planner ported from the old Fish runtime."""

    def __init__(self, prefill_manager: Any, decode_manager: Any, server_args: Any):
        self.prefill_manager = prefill_manager
        self.decode_manager = decode_manager
        self.server_args = server_args
        self.last_batch: Any | None = None
        self.forward_ct: int = 0
        self.req_id_map: dict[str, SchedulerRequest] = {}
        self._cached_schedule_batch: Any | None = None

    def select_requests(
        self,
        waiting: list[SchedulerRequest],
        running: list[SchedulerRequest],
    ) -> list[SchedulerRequest]:
        self._post_step_operations()
        active_request_ids = {req.request_id for req in waiting}
        active_request_ids.update(req.request_id for req in running)
        self._prune_inactive_state(active_request_ids)

        for sched_req in waiting:
            data = sched_req.data
            if not data.synced:
                self.prefill_manager.add_one_request(data.req)
                data.synced = True
                self.req_id_map[data.req.rid] = sched_req

        running_batch = self.decode_manager.running_batch
        running_bs = running_batch.batch_size()
        num_allocatable_reqs = max(
            self.server_args.max_running_requests - running_bs, 0
        )

        running_batch_for_prefill = self.decode_manager.running_batch
        if (
            running_batch_for_prefill is not None
            and running_batch_for_prefill.is_empty()
        ):
            running_batch_for_prefill = None

        schedule_batch = self.prefill_manager.schedule_next_batch(
            running_batch_for_prefill,
            num_allocatable_reqs,
            new_token_ratio=self.decode_manager.new_token_ratio,
        )

        if schedule_batch is None and self.decode_manager.runnable:
            schedule_batch = self.decode_manager.schedule_next_batch(self.forward_ct)

        if schedule_batch is None:
            self._cached_schedule_batch = None
            return []

        self._cached_schedule_batch = schedule_batch
        self.forward_ct += 1

        selected: list[SchedulerRequest] = []
        keep_indices: list[int] = []
        for i, req in enumerate(schedule_batch.reqs):
            sched_req = self.req_id_map.get(req.rid)
            if sched_req is None:
                logger.warning("Fish SGLang req %s not found in req_id_map", req.rid)
                continue
            if sched_req.status in (SchedulerStatus.FINISHED, SchedulerStatus.ABORTED):
                continue
            keep_indices.append(i)
            selected.append(sched_req)

        if len(keep_indices) != len(schedule_batch.reqs):
            if keep_indices:
                schedule_batch.filter_batch(keep_indices=keep_indices)
                self._cached_schedule_batch = schedule_batch
            else:
                self._cached_schedule_batch = None
                return []

        return selected

    def build_batch(self, requests: list[SchedulerRequest]) -> Any:
        del requests
        return self._cached_schedule_batch

    def record_last_batch(self, schedule_batch: Any) -> None:
        self.last_batch = schedule_batch

    def _post_step_operations(self) -> None:
        chunked_req_to_exclude = set()
        active_chunked_req = self.prefill_manager.chunked_req
        if active_chunked_req is not None:
            chunked_req_to_exclude.add(active_chunked_req)
            self.prefill_manager.tree_cache.cache_unfinished_req(
                active_chunked_req, chunked=True
            )
            if active_chunked_req.req_pool_idx is not None:
                self.prefill_manager.req_to_token_pool.free(
                    active_chunked_req.req_pool_idx
                )

        if self.last_batch is None:
            return

        if self.last_batch.forward_mode.is_extend():
            if self.last_batch.chunked_req is not None:
                chunked_req_to_exclude.add(self.last_batch.chunked_req)

            last_bs = self.last_batch.batch_size()
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
            if self.last_batch.batch_size() < last_bs:
                self.decode_manager.running_batch.batch_is_full = False

            if not self.last_batch.is_empty() and not self.last_batch.is_prefill_only:
                if self.decode_manager.running_batch.is_empty():
                    self.decode_manager.running_batch = self.last_batch
                else:
                    self.decode_manager.running_batch.merge_batch(self.last_batch)

        if not self.decode_manager.running_batch.is_empty():
            finished_indices = []
            for i, req in enumerate(self.decode_manager.running_batch.reqs):
                sched_req = self.req_id_map.get(req.rid)
                if req.finished() or (
                    sched_req is not None
                    and sched_req.status
                    in (SchedulerStatus.FINISHED, SchedulerStatus.ABORTED)
                ):
                    finished_indices.append(i)

            if finished_indices:
                keep = [
                    i
                    for i in range(len(self.decode_manager.running_batch.reqs))
                    if i not in finished_indices
                ]
                if keep:
                    self.decode_manager.running_batch.filter_batch(keep_indices=keep)
                else:
                    from sglang.srt.managers.schedule_batch import ScheduleBatch

                    self.decode_manager.running_batch = ScheduleBatch(
                        reqs=[], batch_is_full=False
                    )

        self.last_batch = None

    def _prune_inactive_state(self, active_request_ids: set[str]) -> None:
        inactive_rids: set[str] = set()
        for rid, sched_req in list(self.req_id_map.items()):
            if sched_req.request_id not in active_request_ids or sched_req.status in (
                SchedulerStatus.FINISHED,
                SchedulerStatus.ABORTED,
            ):
                inactive_rids.add(rid)
                del self.req_id_map[rid]

        if not inactive_rids:
            return

        if self.prefill_manager.waiting_queue:
            self.prefill_manager.waiting_queue = [
                req
                for req in self.prefill_manager.waiting_queue
                if req.rid not in inactive_rids
            ]

        running_batch = self.decode_manager.running_batch
        if running_batch is None or running_batch.is_empty():
            return
        keep_indices = [
            i
            for i, req in enumerate(running_batch.reqs)
            if req.rid not in inactive_rids
        ]
        if len(keep_indices) == len(running_batch.reqs):
            return
        if keep_indices:
            running_batch.filter_batch(keep_indices=keep_indices)
        else:
            from sglang.srt.managers.schedule_batch import ScheduleBatch

            self.decode_manager.running_batch = ScheduleBatch(
                reqs=[], batch_is_full=False
            )


class FishResourceManager:
    def __init__(
        self, token_to_kv_pool_allocator: Any, req_to_token_pool: Any, tree_cache: Any
    ):
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.req_to_token_pool = req_to_token_pool
        self.tree_cache = tree_cache

    def free(self, request: SchedulerRequest) -> None:
        data = request.data
        if data.req is not None:
            release_kv_cache(data.req, self.tree_cache)
        data.previous_semantic_tokens.clear()
        data.last_codebook_values = None


class FishIterationController:
    def __init__(
        self, tree_cache: Any, im_end_token_id: int, max_new_tokens: int = 2048
    ):
        self.tree_cache = tree_cache
        self._im_end_token_id = int(im_end_token_id)
        self._max_new_tokens = int(max_new_tokens)

    def update_request(
        self, request: SchedulerRequest, output_token_id: int | None
    ) -> None:
        data = request.data
        req = data.req

        if req.is_chunked > 0:
            req.is_chunked -= 1
            return

        if output_token_id is not None:
            semantic_token = int(output_token_id)
            req.output_ids.append(semantic_token)
            # Skip caching the terminal slow-AR EOS regardless of req.finished()
            # semantics: it is not an audio timestep and has no KV to preserve.
            if semantic_token == self._im_end_token_id:
                return
            if not req.finished() and req.decode_batch_idx == 0:
                self.tree_cache.cache_unfinished_req(req)

    def is_finished(
        self, request: SchedulerRequest, output_token_id: int | None
    ) -> bool:
        data = request.data
        if data.req.is_chunked > 0:
            return False

        semantic_token = output_token_id
        if semantic_token is None and data.previous_semantic_tokens:
            semantic_token = int(data.previous_semantic_tokens[-1])

        if semantic_token == self._im_end_token_id:
            return True

        max_tok = data.max_new_tokens or self._max_new_tokens
        if len(data.output_codes) >= max_tok:
            return True

        return False


class FishScheduler:
    """Stage-facing scheduler for Fish TTS with Fish-specific finish semantics."""

    def __init__(
        self,
        *,
        tree_cache: Any,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        prefill_manager: Any,
        decode_manager: Any,
        server_args: Any,
        model_runner: Any,
        request_builder: Any,
        result_adapter: Any,
        im_end_token_id: int,
        max_new_tokens: int,
    ):
        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()

        self._request_builder = request_builder
        self._result_adapter = result_adapter
        self._model_runner = model_runner

        self._running = False
        self._aborted_request_ids: set[str] = set()

        self._requests: dict[str, SchedulerRequest] = {}
        self._waiting: deque[str] = deque()
        self._running_ids: list[str] = []
        self._submit_times: dict[str, float] = {}
        self._step_id = 0

        self.batch_planner = FishBatchPlanner(
            prefill_manager, decode_manager, server_args
        )
        self.resource_manager = FishResourceManager(
            token_to_kv_pool_allocator,
            req_to_token_pool,
            tree_cache,
        )
        self.iteration_controller = FishIterationController(
            tree_cache=tree_cache,
            im_end_token_id=im_end_token_id,
            max_new_tokens=max_new_tokens,
        )

    def recv_requests(self) -> list[Any]:
        new_reqs: list[Any] = []
        while True:
            try:
                msg = self.inbox.get_nowait()
            except _queue_mod.Empty:
                break

            if msg.request_id in self._aborted_request_ids:
                continue
            if msg.type == "new_request":
                new_reqs.append(msg.data)
        return new_reqs

    def process_input_requests(self, recv_reqs: list[Any]) -> None:
        for payload in recv_reqs:
            req_id = payload.request_id
            req_data = self._request_builder(payload)
            sched_req = SchedulerRequest(
                request_id=req_id,
                data=req_data,
                arrival_time=time.time(),
            )
            self._submit_times[req_id] = time.perf_counter()
            self._requests[req_id] = sched_req
            self._waiting.append(req_id)

    def schedule(self) -> SchedulerOutput | None:
        if not self._waiting and not self._running_ids:
            return None

        self._step_id += 1
        waiting = [self._requests[req_id] for req_id in self._waiting]
        running = [self._requests[req_id] for req_id in self._running_ids]
        selected = self.batch_planner.select_requests(waiting, running)
        if not selected:
            return None

        for request in selected:
            if request.request_id in self._waiting:
                self._waiting.remove(request.request_id)
                self._running_ids.append(request.request_id)
                request.status = SchedulerStatus.RUNNING

        batch_data = self.batch_planner.build_batch(selected)
        return SchedulerOutput(
            requests=selected, batch_data=batch_data, step_id=self._step_id
        )

    def update(
        self,
        scheduler_output: SchedulerOutput,
        model_output: ModelRunnerOutput,
    ) -> list[SchedulerRequest]:
        finished: list[SchedulerRequest] = []

        for request in scheduler_output.requests:
            if request.request_id in self._aborted_request_ids:
                continue
            if request.status != SchedulerStatus.RUNNING:
                continue

            output = model_output.outputs.get(request.request_id)
            if output is None:
                logger.warning(
                    "Missing Fish output for request_id=%s", request.request_id
                )
                continue

            self.iteration_controller.update_request(request, output.data)
            if self.iteration_controller.is_finished(request, output.data):
                self._finish_request(request)
                finished.append(request)

        return finished

    def _finish_request(self, request: SchedulerRequest) -> None:
        request.status = SchedulerStatus.FINISHED
        request.finish_time = time.time()
        self.resource_manager.free(request)
        if request.request_id in self._running_ids:
            self._running_ids.remove(request.request_id)
        if request.request_id in self._waiting:
            self._waiting.remove(request.request_id)
        self._aborted_request_ids.discard(request.request_id)

    def run_batch(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        model_output = self._model_runner.execute(scheduler_output)
        self.batch_planner.record_last_batch(scheduler_output.batch_data)
        return model_output

    def emit_finished(self, finished: list[SchedulerRequest]) -> None:
        for request in finished:
            data = request.data
            data.output_ids = list(data.req.output_ids)
            t_submit = self._submit_times.pop(request.request_id, None)
            if not data.output_codes:
                self.outbox.put(
                    OutgoingMessage(
                        request_id=request.request_id,
                        type="error",
                        data=ValueError(
                            f"Request {request.request_id}: "
                            "S2-Pro generated no audio codec tokens"
                        ),
                    )
                )
                continue
            result = self._result_adapter(data)
            if t_submit is not None and isinstance(result.data, dict):
                result.data["engine_time_s"] = time.perf_counter() - t_submit
            self.outbox.put(
                OutgoingMessage(
                    request_id=request.request_id,
                    type="result",
                    data=result,
                )
            )

    def start(self) -> None:
        self._running = True
        while self._running:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            batch = self.schedule()
            if batch is None:
                time.sleep(0.001)
                continue

            try:
                result = self.run_batch(batch)
                finished = self.update(batch, result)
                self.emit_finished(finished)
            except Exception as exc:
                logger.exception("FishScheduler batch failed")
                for request in batch.requests:
                    self.outbox.put(
                        OutgoingMessage(
                            request_id=request.request_id,
                            type="error",
                            data=exc,
                        )
                    )
                    self.abort(request.request_id)

    def stop(self) -> None:
        self._running = False

    def abort(self, request_id: str) -> None:
        self._aborted_request_ids.add(request_id)
        self._requests.pop(request_id, None)
        self._submit_times.pop(request_id, None)
        self._waiting = deque(
            req_id for req_id in self._waiting if req_id != request_id
        )
        if request_id in self._running_ids:
            self._running_ids.remove(request_id)
