import asyncio
import logging
import time
from collections import deque
from typing import Any, AsyncIterator, Callable

from sglang_omni.vendor.sglang.core import ModelConfig, ScheduleBatch, ServerArgs, envs

from ..model_worker import ModelWorker, ModelWorkerConfig
from .cache import CacheManager
from .decode import DecodeManager
from .prefill import PrefillManager

from ....omni.types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
    SchedulerStatus,
)

logger = logging.getLogger(__name__)


class SGLangSchedulerConfig:
    server_args = ServerArgs
    model_config = ModelConfig
    device = int


def _conver_model_worker_config(config: SGLangSchedulerConfig) -> ModelWorkerConfig:
    return ModelWorkerConfig()


class SGLangScheduler:
    def __init__(
        self,
        config: SGLangSchedulerConfig,
    ):
        self.config = config
        self.server_args = self.config.server_args
        self.device = self.config.device

        # The current forward batch
        self.cur_batch = None
        # The last forward batch
        self.last_batch = None

        # Init memory pool and cache manager
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            self.tp_worker.get_memory_pool()
        )

        self.cache_manager = CacheManager()

        self.init_chunked_prefill()

        # Init prefill & decode manager
        self.prefill_manager = PrefillManager(
            page_size=self.server_args.page_size,
            chunked_prefill_size=self.chunked_prefill_size,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.cache_manager.tree_cache,
            model_config=self.tp_worker.model_config,
            enable_overlap=False,
        )
        self.decode_manager = DecodeManager(
            server_args=self.server_args,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
        )

        self.forward_ct = 0
        self.init_schedule_policy()

    def _init_model_worker(self):
        config = _conver_model_worker_config(self.config)
        self.tp_worker = ModelWorker(
            config=config,
            server_args=self.server_args,
            gpu_id=self.device,
            tp_rank=0,
        )

    def init_schedule_policy(self):
        # Init schedule policy and new token estimation
        # Enable preemption for priority scheduling.
        self.init_new_token_ratio = min(
            envs.SGLANG_INIT_NEW_TOKEN_RATIO.get()
            * self.server_args.schedule_conservativeness,
            1.0,
        )
        self.min_new_token_ratio = min(
            self.init_new_token_ratio * envs.SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR.get(),
            1.0,
        )
        self.new_token_ratio_decay = (
            self.init_new_token_ratio - self.min_new_token_ratio
        ) / envs.SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS.get()
        self.new_token_ratio = self.init_new_token_ratio

    def init_chunked_prefill(self):
        # TODO(ocss884): For simplicity, we disabled `dynamic chunking` and `mixed chunk` for now
        self.chunked_prefill_size = self.server_args.chunked_prefill_size


    # =========================================================================
    # SGLangEngine Public API - Request Lifecycle
    # =========================================================================

    def add_request(self, request_id: str, data: Any) -> None:
        """Add a new request with model-specific data.

        Args:
            request_id: Unique identifier for the request.
            data: Model-specific request data (e.g., token ids, generation
                  params). This is opaque to the scheduler.
        """
        request = SchedulerRequest(
            request_id=request_id,
            data=data,
            arrival_time=time.time(),
        )
        self.requests[request_id] = request
        self.waiting.append(request_id)

        # Also prepare SGLang-native request representation if needed
        sglang_req = self._convert_to_sglang_request(request)
        if sglang_req is not None:
            self._sglang_requests[request_id] = sglang_req

        logger.info(
            "sched.add_request id=%s waiting=%d running=%d total=%d",
            request_id,
            len(self.waiting),
            len(self.running),
            len(self.requests),
        )

    def abort_request(self, request_id: str) -> None:
        """Abort a request."""
        request = self.requests.get(request_id)
        if request is None:
            return
        self._aborted_this_step.add(request_id)
        self._finish_request(request, status=SchedulerStatus.ABORTED)

    def fail_request(self, request_id: str, error: Exception) -> None:
        """Fail a request with an error, propagating it to any waiting caller."""
        request = self.requests.get(request_id)
        if request is None:
            return
        self._aborted_this_step.add(request_id)
        self._finish_request(request, status=SchedulerStatus.ABORTED, error=error)

    def has_requests(self) -> bool:
        """Check if there are any requests to process."""
        return len(self.waiting) > 0 or len(self.running) > 0

    async def get_result(self, request_id: str) -> SchedulerRequest:
        """Wait for a request to complete and return the result."""
        if request_id not in self.requests:
            raise KeyError(f"Unknown request: {request_id}")
        loop = asyncio.get_running_loop()

        logger.info("sched.get_result wait id=%s", request_id)

        while True:
            request = self.requests[request_id]
            if request.status in (SchedulerStatus.FINISHED, SchedulerStatus.ABORTED):
                if request.error is not None:
                    raise request.error
                return request

            future = self._futures.get(request_id)
            if future is None or future.cancelled():
                future = loop.create_future()
                self._futures[request_id] = future
                logger.info("sched.get_result create_future id=%s", request_id)

            await asyncio.shield(future)

    async def stream(self, request_id: str) -> AsyncIterator[Any]:
        """Yield per-step stream data for a request."""
        queue = self._subscribe_stream(request_id)
        while True:
            item = await queue.get()
            if item is self._stream_done:
                return
            yield item

    def _subscribe_stream(self, request_id: str) -> asyncio.Queue[Any]:
        if request_id not in self.requests:
            raise KeyError(f"Unknown request: {request_id}")
        queue = self._stream_queues.get(request_id)
        if queue is None:
            queue = asyncio.Queue()
            self._stream_queues[request_id] = queue
        request = self.requests[request_id]
        if request.status in (SchedulerStatus.FINISHED, SchedulerStatus.ABORTED):
            queue.put_nowait(self._stream_done)
        return queue

    # =========================================================================
    # SGLangEngine Core Scheduling Protocol
    # =========================================================================

    def schedule(self) -> SchedulerOutput | None:
        """Schedule next batch of requests.

        Returns a SchedulerOutput containing the selected requests and
        SGLang's native ScheduleBatch as batch_data, or None if idle.
        """
        if not self.waiting and not self.running:
            return None

        self._step_id += 1
        self._aborted_this_step.clear()

        # Use SGLang's native scheduling to determine the next batch
        sglang_batch = self._get_next_batch_to_run()

        if sglang_batch is None:
            return None

        # Extract request_ids from the SGLang batch and map to SchedulerRequests
        batch_request_ids = self._extract_request_ids_from_batch(sglang_batch)
        selected: list[SchedulerRequest] = []

        for req_id in batch_request_ids:
            request = self.requests.get(req_id)
            if request is None:
                continue
            selected.append(request)

            # Move from waiting to running if necessary
            if request.request_id in self.waiting:
                self.waiting.remove(request.request_id)
                if request.request_id not in self.running:
                    self.running.append(request.request_id)
                request.status = SchedulerStatus.RUNNING

        if not selected:
            return None

        self.cur_batch = sglang_batch
        self.forward_ct += 1

        logger.info(
            "sched.schedule step=%d selected=%d waiting=%d running=%d",
            self._step_id,
            len(selected),
            len(self.waiting),
            len(self.running),
        )

        return SchedulerOutput(
            requests=selected,
            batch_data=sglang_batch,  # SGLang's ScheduleBatch, opaque to engine
            step_id=self._step_id,
        )

    def update(
        self,
        scheduler_output: SchedulerOutput,
        model_output: ModelRunnerOutput,
    ) -> list[SchedulerRequest]:
        """Update scheduler state from model execution results.

        This method:
        1. Processes each request's output from ModelRunnerOutput
        2. Updates SGLang-native state (KV cache, decode state, etc.)
        3. Determines which requests are finished
        4. Emits streaming data if applicable

        Args:
            scheduler_output: The SchedulerOutput from the schedule() call.
            model_output: The ModelRunnerOutput from model execution.

        Returns:
            List of finished SchedulerRequest objects.
        """
        finished: list[SchedulerRequest] = []
        sglang_batch = scheduler_output.batch_data  # The native ScheduleBatch

        for request in scheduler_output.requests:
            if request.request_id in self._aborted_this_step:
                continue

            output = model_output.outputs.get(request.request_id)
            if output is None:
                logger.warning(
                    "Missing output for request_id=%s", request.request_id
                )
                continue

            # Update request data with the output
            self._update_request_state(request, output)

            # Emit streaming data
            self._emit_stream(request, output)

            # Check if the request is finished
            if self._is_request_finished(request, output):
                self._finish_request(request)
                finished.append(request)

        # Update SGLang-native batch state (process_batch_result equivalent)
        if sglang_batch is not None:
            self._process_sglang_batch_result(sglang_batch, model_output)

        self.last_batch = sglang_batch

        return finished

    # =========================================================================
    # SGLang-native scheduling internals
    # =========================================================================

    def _get_next_batch_to_run(self) -> ScheduleBatch | None:
        """Determine next batch using SGLang's native scheduling logic.

        Wraps the original get_next_batch_to_run with proper state management.
        """
        chunked_req_to_exclude = set()

        if self.last_batch and self.last_batch.forward_mode.is_extend():
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

        running_bs = len(self.decode_manager.running_batch)
        num_allocatable_reqs = self._get_num_allocatable_reqs(running_bs)

        # Try prefill first
        next_batch = self.prefill_manager.schedule_next_batch(
            self.decode_manager.running_batch, num_allocatable_reqs
        )
        if next_batch is not None:
            return next_batch

        # Then try decode
        if self.decode_manager.runnable:
            ret = self.decode_manager.schedule_next_batch(self.forward_ct)
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_decay,
                self.min_new_token_ratio,
            )
            return ret

        return None

    def _get_num_allocatable_reqs(self, running_bs: int) -> int:
        return self.server_args.max_running_requests - running_bs


    def init_overlap(self):
        # TODO(ocss884): implement overlap scheduling
        raise NotImplementedError

    def recv_requests():
        pass

    def normal_loop(self) -> None:
        """A normal scheduler loop."""
        while True:
            recv_reqs = self.recv_requests()
            self.process_requess(recv_reqs)

            batch = self.get_next_batch_to_run(batch)
            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)

            self.last_batch = batch

    def process_batch_result(self, batch, result):
        # TODO(ocss884): implement result processing, including sending output to tokenizer and updating cache
        for i, req in enumerate(batch.reqs):
            if req.is_chunked > 0:
                req.is_chunked -= 1
                continue

            token_id = next_token_ids[i].item()
            req.output_ids.append(token_id)

            req.check_finished()

            if req.finished():
                tree_cache.cache_finished_req(req)
            else:
                tree_cache.cache_unfinished_req(req)

    def get_next_batch_to_run(self):
        # TODO(ocss884): maybe support more scheduling strategies

        chunked_req_to_exclude = set()

        if self.last_batch and self.last_batch.forward_mode.is_extend():
            if self.last_batch.chunked_req is not None:
                # In the context pipeline parallelism, after the last chunk, the current microbatch still track outdated chunked_req.
                # We need to discard it.
                chunked_req_to_exclude.add(self.last_batch.chunked_req)

            # Filter batch
            last_bs = self.last_batch.batch_size()
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
            if self.last_batch.batch_size() < last_bs:
                self.running_batch.batch_is_full = False

            # Merge the new batch into the running batch.
            # For prefill-only batch, we can avoid going through decoding step.
            if not self.last_batch.is_empty() and not self.last_batch.is_prefill_only:
                if self.running_batch.is_empty():
                    self.running_batch = self.last_batch
                else:
                    # Merge running_batch with prefill batch
                    self.running_batch.merge_batch(self.last_batch)

        running_bs = len(self.decode_manager.running_batch)
        num_allocatable_reqs = self.get_num_allocatable_reqs(running_bs)

        if (
            next_batch := self.prefill_manager.schedule_next_batch(
                self.decode_manager.running_batch, num_allocatable_reqs
            )
            is not None
        ):
            ret = next_batch
        elif self.decode_manager.runnable:
            ret = self.decode_manager.schedule_next_batch(self.forward_ct)
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_decay,
                self.min_new_token_ratio,
            )
        else:
            ret = None

        return ret

    def get_num_allocatable_reqs(self, running_bs):
        # NOTE(ocss884): cp from sglang but removed pp
        res = self.server_args.max_running_requests - running_bs
        return res

    # TODO(ocss884): fix it
    def run_batch(self, batch: ScheduleBatch):
        model_worker_batch = batch.get_model_worker_batch()

        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_worker.model_runner)



def launch_scheduler(self, config: SGLangSchedulerConfig):
    scheduler = SGLangScheduler(config)
    return scheduler
