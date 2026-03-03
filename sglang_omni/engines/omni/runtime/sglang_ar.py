# SPDX-License-Identifier: Apache-2.0
"""SGLang AR backend integration into OmniEngine's Protocol system.

Provides BatchPlanner, ResourceManager, OutputProcessor, IterationController,
and ModelRunner implementations that delegate to SGLang's PrefillManager,
DecodeManager, and ModelWorker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from sglang.srt.mem_cache.common import release_kv_cache

from ..types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
    SchedulerStatus,
)
from .ar import ARRequestData

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

    from sglang_omni.engines.ar.sglang_backend.model_worker import ModelWorker
    from sglang_omni.engines.ar.sglang_backend.scheduler.decode import DecodeManager
    from sglang_omni.engines.ar.sglang_backend.scheduler.prefill import PrefillManager

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------


@dataclass
class SGLangARRequestData(ARRequestData):
    """SGLang-specific AR request data.

    Subclasses ARRequestData so isinstance(result, ARRequestData) passes
    and all existing field accesses (.output_ids, .extra_model_outputs) work.
    """

    req: Any = None
    synced: bool = False


# -----------------------------------------------------------------------------
# BatchPlanner
# -----------------------------------------------------------------------------


class SGLangBatchPlanner:
    """Batch planner that delegates to SGLang's PrefillManager and DecodeManager.

    Implements the BatchPlanner protocol (select_requests, build_batch).
    """

    def __init__(
        self,
        prefill_manager: "PrefillManager",
        decode_manager: "DecodeManager",
        server_args: "ServerArgs",
    ):
        self.prefill_manager = prefill_manager
        self.decode_manager = decode_manager
        self.server_args = server_args
        self.last_batch: Any | None = None
        self.forward_ct: int = 0
        # Maps SGLang req.rid -> OmniEngine SchedulerRequest
        self.req_id_map: dict[str, SchedulerRequest] = {}
        self._cached_schedule_batch: Any | None = None

    def select_requests(
        self,
        waiting: list[SchedulerRequest],
        running: list[SchedulerRequest],
        resource_manager: Any,
    ) -> list[SchedulerRequest]:
        # 1. Post-step from previous batch
        self._post_step_operations()
        active_request_ids = {req.request_id for req in waiting}
        active_request_ids.update(req.request_id for req in running)
        self._prune_inactive_state(active_request_ids)

        # 2. Sync new SchedulerRequests to PrefillManager
        for sched_req in waiting:
            data: SGLangARRequestData = sched_req.data
            if not data.synced:
                self.prefill_manager.add_one_request(data.req)
                data.synced = True
                self.req_id_map[data.req.rid] = sched_req

        # 3. Try prefill first
        running_batch = self.decode_manager.running_batch
        if hasattr(running_batch, "batch_size"):
            running_bs = running_batch.batch_size()
        else:
            running_bs = len(getattr(running_batch, "reqs", []))
        num_allocatable_reqs = max(
            self.server_args.max_running_requests - running_bs, 0
        )

        running_batch_for_prefill = self.decode_manager.running_batch
        if (
            running_batch_for_prefill is not None
            and hasattr(running_batch_for_prefill, "is_empty")
            and running_batch_for_prefill.is_empty()
        ):
            running_batch_for_prefill = None

        schedule_batch = self.prefill_manager.schedule_next_batch(
            running_batch_for_prefill,
            num_allocatable_reqs,
            new_token_ratio=self.decode_manager.new_token_ratio,
        )

        # 4. Fallback to decode
        if schedule_batch is None and self.decode_manager.runnable:
            schedule_batch = self.decode_manager.schedule_next_batch(self.forward_ct)

        if schedule_batch is None:
            self._cached_schedule_batch = None
            return []

        self._cached_schedule_batch = schedule_batch
        self.forward_ct += 1

        # Map ScheduleBatch reqs back to SchedulerRequests
        selected: list[SchedulerRequest] = []
        keep_indices: list[int] = []
        for i, req in enumerate(schedule_batch.reqs):
            sched_req = self.req_id_map.get(req.rid)
            if sched_req is not None and sched_req.status not in (
                SchedulerStatus.FINISHED,
                SchedulerStatus.ABORTED,
            ):
                keep_indices.append(i)
                selected.append(sched_req)
            elif sched_req is None:
                logger.warning("SGLang req %s not found in req_id_map", req.rid)

        if len(keep_indices) != len(schedule_batch.reqs):
            if keep_indices:
                schedule_batch.filter_batch(keep_indices=keep_indices)
                self._cached_schedule_batch = schedule_batch
            else:
                self._cached_schedule_batch = None
                return []

        return selected

    def build_batch(self, requests: list[SchedulerRequest]) -> Any:
        return self._cached_schedule_batch

    def _post_step_operations(self):
        """Handle post-step merging from last prefill batch into decode batch."""
        chunked_req_to_exclude = set()
        active_chunked_req = self.prefill_manager.chunked_req
        if active_chunked_req is not None:
            chunked_req_to_exclude.add(active_chunked_req)
            self.prefill_manager.tree_cache.cache_unfinished_req(
                active_chunked_req, chunked=True
            )
            # Reuse req_to_token slot across chunk rounds; otherwise long chunked
            # prompts can exhaust req_to_token_pool entries.
            if getattr(active_chunked_req, "req_pool_idx", None) is not None:
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

        # Filter finished requests from decode running_batch
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

    def record_last_batch(self, schedule_batch: Any):
        """Called after model execution to record the last batch for post-step."""
        self.last_batch = schedule_batch

    def _prune_inactive_state(self, active_request_ids: set[str]) -> None:
        """Prune stale request mappings and remove inactive reqs from SGLang queues."""
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


# -----------------------------------------------------------------------------
# ResourceManager
# -----------------------------------------------------------------------------


class SGLangResourceManager:
    """Resource manager that delegates KV cache management to SGLang internals.

    PrefillAdder handles real allocation checks internally, so can_allocate/allocate
    are pass-throughs. KV release happens only in free().
    """

    def __init__(self, token_to_kv_pool_allocator, req_to_token_pool, tree_cache):
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.req_to_token_pool = req_to_token_pool
        self.tree_cache = tree_cache

    def can_allocate(self, request: SchedulerRequest) -> bool:
        return True

    def allocate(self, request: SchedulerRequest) -> None:
        pass

    def free(self, request: SchedulerRequest) -> None:
        data: SGLangARRequestData = request.data
        if data.req is not None:
            release_kv_cache(data.req, self.tree_cache)


# -----------------------------------------------------------------------------
# OutputProcessor
# -----------------------------------------------------------------------------


class SGLangOutputProcessor:
    """Converts GenerationBatchResult to per-request RequestOutputs."""

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, RequestOutput]:
        token_list = (
            model_output.next_token_ids.tolist()
            if model_output.next_token_ids is not None
            else []
        )

        outputs = {}
        for i, sched_req in enumerate(scheduler_output.requests):
            token_id = token_list[i] if i < len(token_list) else None
            outputs[sched_req.request_id] = RequestOutput(
                request_id=sched_req.request_id,
                data=token_id,
                finished=False,
            )
        return outputs


# -----------------------------------------------------------------------------
# IterationController
# -----------------------------------------------------------------------------


class SGLangIterationController:
    """Handles per-request state updates with chunked prefill semantics.

    Chunked prefill: req.is_chunked > 0 means the request is still in
    chunked prefill — decrement counter, do NOT append token or check finish.
    """

    def __init__(self, tree_cache):
        self.tree_cache = tree_cache

    def update_request(self, request: SchedulerRequest, output: RequestOutput) -> None:
        data: SGLangARRequestData = request.data
        req = data.req

        if req.is_chunked > 0:
            # Suppress stream emission for this chunk. Scheduler emits stream
            # after update_request(), so marking output None here avoids races.
            output.data = None
            req.is_chunked -= 1
            return

        token_id = output.data
        if token_id is not None:
            # output_ids is shared: req.output_ids IS data.output_ids
            req.output_ids.append(token_id)
            req.check_finished()
            # Mirror upstream prefill behavior: unfinished prefill requests
            # should insert/update prefix cache after extend step.
            if not req.finished() and getattr(req, "decode_batch_idx", 0) == 0:
                self.tree_cache.cache_unfinished_req(req)

    def is_finished(self, request: SchedulerRequest, output: RequestOutput) -> bool:
        return request.data.req.finished()


# -----------------------------------------------------------------------------
# SGLangModelRunner (duck-typed replacement for ModelRunner)
# -----------------------------------------------------------------------------


class SGLangModelRunner:
    """Model runner that uses SGLang's ModelWorker for execution.

    Replaces the generic ModelRunner — handles ScheduleBatch → ForwardBatch
    conversion, forward pass, and conditional sampling.
    """

    def __init__(
        self,
        model_worker: "ModelWorker",
        output_processor: SGLangOutputProcessor,
        batch_planner: SGLangBatchPlanner | None = None,
    ):
        self.model_worker = model_worker
        self.output_processor = output_processor
        self.batch_planner = batch_planner

    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        schedule_batch = scheduler_output.batch_data

        if schedule_batch is None:
            return ModelRunnerOutput(outputs={}, req_ids=[], req_id_to_index={})

        # 1. Get ModelWorkerBatch from ScheduleBatch
        model_worker_batch = schedule_batch.get_model_worker_batch()

        # 2. Create ForwardBatch
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.model_worker.model_runner
        )

        # 3. Forward pass
        batch_result = self.model_worker.forward_batch_generation(forward_batch)

        # 4. Produce per-request next tokens and store them on ScheduleBatch.
        # Decode preparation relies on schedule_batch.output_ids from the
        # previous step; without this, the next decode step cannot run.
        if schedule_batch.is_prefill_only:
            batch_result.next_token_ids = torch.zeros(
                len(model_worker_batch.seq_lens),
                dtype=torch.long,
                device=model_worker_batch.input_ids.device,
            )
        else:
            batch_result.next_token_ids = self.model_worker.model_runner.sample(
                batch_result.logits_output, forward_batch
            )
        schedule_batch.output_ids = batch_result.next_token_ids

        # 5. Record last batch for post-step operations
        if self.batch_planner is not None:
            self.batch_planner.record_last_batch(schedule_batch)

        # 6. Process outputs
        outputs = self.output_processor.process(batch_result, scheduler_output)

        req_ids = [req.request_id for req in scheduler_output.requests]
        req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}

        return ModelRunnerOutput(
            outputs=outputs,
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
        )
