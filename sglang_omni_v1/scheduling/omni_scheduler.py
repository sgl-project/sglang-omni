# SPDX-License-Identifier: Apache-2.0
"""OmniScheduler — stage-facing AR scheduler using composition.

Uses SGLang's batch selection and result processing logic via **unbound
method calls** on the upstream ``Scheduler`` class.  No inheritance.

When an upstream method (e.g. ``get_next_batch_to_run``) internally calls
``self.get_new_batch_prefill()``, Python finds it through
``OmniScheduler.__getattr__`` → looks it up on the upstream class → binds
it to this instance.  This gives us the full scheduling MRO without
inheriting from ``SGLangScheduler``.
"""
from __future__ import annotations

import logging
import queue as _queue_mod
import time
import types
from collections import deque
from typing import Any, Callable

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler as _Upstream
from sglang.srt.utils import broadcast_pyobj

from sglang_omni_v1.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stubs for upstream subsystems we don't use
# ---------------------------------------------------------------------------


class _NoOpSender:
    """Stub for send_to_detokenizer — stream_output handles emission."""

    def send_output(self, *args, **kwargs):
        pass


class _NoOpGrammarManager:
    """Stub — OmniScheduler never uses constrained decoding."""

    grammar_queue: list = []

    def has_waiting_grammars(self) -> bool:
        return False

    def get_ready_grammar_requests(self) -> list:
        return []

    def abort_requests(self, recv_req) -> None:
        pass

    def clear(self) -> None:
        pass

    def __len__(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# OmniScheduler
# ---------------------------------------------------------------------------


class OmniScheduler:
    """Stage-facing scheduler for AR stages.

    Public contract (used by Stage):
        ``inbox``, ``outbox``, ``start()``, ``stop()``, ``abort(request_id)``

    Composition strategy:
        SGLang scheduling methods (``get_next_batch_to_run``,
        ``process_batch_result``, …) are looked up on the upstream
        ``Scheduler`` *class* via ``__getattr__`` and called with this
        instance as ``self``.  Methods we override (``recv_requests``,
        ``process_input_requests``, ``run_batch``, ``send_to_tokenizer``)
        are defined directly on this class and take precedence.
    """

    def __init__(
        self,
        tp_worker: Any,
        tree_cache: Any,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        server_args: Any,
        model_config: Any,
        *,
        prefill_manager: Any = None,
        decode_manager: Any = None,
        model_runner: Any = None,
        request_builder: Callable | None = None,
        result_adapter: Callable | None = None,
        stream_output_builder: Callable | None = None,
        stream_chunk_handler: Callable | None = None,
        stream_done_handler: Callable | None = None,
        enable_overlap: bool = False,
    ):
        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()

        # --- Request builder: StagePayload → SGLangARRequestData ----------
        self._request_builder = request_builder
        self._result_adapter = result_adapter
        self._model_runner = model_runner
        self._stream_output_builder = stream_output_builder
        self._stream_chunk_handler = stream_chunk_handler
        self._stream_done_handler = stream_done_handler

        # --- Core scheduling state (read/written by upstream methods) -----
        self.server_args = server_args
        self.model_config = model_config
        self.gpu_id = tp_worker.gpu_id
        self.tp_rank = getattr(tp_worker, "tp_rank", 0)
        self.tp_size = server_args.tp_size
        self.pp_rank = 0
        self.pp_size = server_args.pp_size
        self.dp_rank = None
        self.dp_size = 1
        self.moe_ep_rank = 0
        self.moe_ep_size = 1
        self.page_size = server_args.page_size
        self.enable_overlap = enable_overlap

        # Token / memory info (upstream reads from tp_worker.get_worker_info)
        mr = tp_worker.model_runner
        self.max_total_num_tokens = mr.max_total_num_tokens
        self.max_prefill_tokens = server_args.max_prefill_tokens
        self.max_running_requests = server_args.max_running_requests
        self.max_req_len = min(
            server_args.context_length - 1,
            self.max_total_num_tokens - 1,
        )
        self.max_req_input_len = self.max_req_len - 1
        self.random_seed = tp_worker.random_seed
        self.device = tp_worker.device

        # Global server_args field upstream sets in its __init__
        from sglang.srt.server_args import get_global_server_args

        gsa = get_global_server_args()
        if gsa is not None and gsa.pp_max_micro_batch_size is None:
            gsa.pp_max_micro_batch_size = max(
                self.max_running_requests // self.pp_size,
                1,
            )

        # Workers
        self.tp_worker = tp_worker
        self.model_worker = tp_worker

        # Cache / memory management
        self.tree_cache = tree_cache
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.prefill_manager = prefill_manager
        self.decode_manager = decode_manager

        # Batch state
        self.waiting_queue: list = []
        self.running_batch = ScheduleBatch(reqs=[], batch_is_full=False)
        self.cur_batch = None
        self.last_batch = None
        self.forward_ct = 0
        self.return_health_check_ct = 0
        self.num_retracted_reqs = 0
        self.num_paused_reqs = 0
        self.sessions: dict = {}
        self.forward_sleep_time = None
        self._engine_paused = False

        # Chunked prefill
        self.chunked_prefill_size = server_args.chunked_prefill_size
        if self.chunked_prefill_size <= 0:
            self.chunked_prefill_size = None
        self.chunked_req = None
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None and server_args.enable_mixed_chunk
        )
        self.enable_dynamic_chunking = False

        # Schedule policy
        from sglang.srt.managers.schedule_policy import SchedulePolicy

        self.schedule_policy = server_args.schedule_policy
        self.policy = SchedulePolicy(
            self.schedule_policy,
            self.tree_cache,
            server_args.enable_hierarchical_cache,
            server_args.enable_priority_scheduling,
            server_args.schedule_low_priority_values_first,
        )
        self.enable_priority_scheduling = server_args.enable_priority_scheduling
        self.try_preemption = server_args.enable_priority_scheduling
        self.priority_scheduling_preemption_threshold = (
            server_args.priority_scheduling_preemption_threshold
        )
        self.schedule_low_priority_values_first = (
            server_args.schedule_low_priority_values_first
        )
        self.init_new_token_ratio = min(
            envs.SGLANG_INIT_NEW_TOKEN_RATIO.get()
            * server_args.schedule_conservativeness,
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
        self.prefill_delayer = None

        # Feature flags (all disabled)
        self.enable_lora = False
        self.enable_pdmux = False
        self.enable_metrics = server_args.enable_metrics
        self.enable_trace = False
        self.enable_hierarchical_cache = False
        self.enable_hicache_storage = False
        self.enable_kv_cache_events = False
        self.is_generation = True
        self.skip_tokenizer_init = True
        self.stream_interval = 1
        self.max_recv_per_poll = 64
        self.enable_lora_overlap_loading = False
        self.enable_metrics_for_all_schedulers = (
            server_args.enable_metrics_for_all_schedulers
        )
        self.current_scheduler_metrics_enabled = False

        # Speculative decoding (disabled)
        try:
            from sglang.srt.managers.scheduler import DllmStagingReqs
        except ImportError:

            class DllmStagingReqs:  # type: ignore[no-redef]
                def __init__(self, dllm_config=None):
                    self.dllm_config = dllm_config

        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        self.spec_algorithm = SpeculativeAlgorithm.NONE
        self.dllm_config = None
        self.dllm_staging_reqs = DllmStagingReqs(dllm_config=None)
        self.draft_worker = None

        # Subsystem stubs
        self.watchdog = None
        self.soft_watchdog = None
        self.recv_skipper = None
        self.idle_sleeper = None
        self.grammar_manager = _NoOpGrammarManager()
        self.grammar_queue = []
        self.grammar_backend = None
        self.require_mlp_sync = False
        self.abort_on_priority_when_disabled = False

        # Disaggregation / hybrid (disabled)
        from sglang.srt.disaggregation.utils import DisaggregationMode

        self.disaggregation_mode = DisaggregationMode.NULL
        self.is_hybrid_swa = False
        self.is_hybrid_ssm = False
        self.offload_tags: set = set()
        self.is_initializing = False
        self.truncation_align_size = None

        # Attention parallelism / TP ownership
        self.attn_tp_rank = self.tp_rank
        self.attn_tp_size = self.tp_size
        self.attn_dp_rank = 0
        self.tp_group = None
        self.tp_cpu_group = None
        self.attn_tp_group = None
        self.attn_tp_cpu_group = None
        self.cpu_group = None
        self.entry_rank = 0
        self.is_entry_rank = self.tp_rank == 0

        # Misc
        self.metrics_collector = None
        self.pad_input_ids_func = None
        self.decode_mem_cache_buf_multiplier = 0
        self.decode_offload_manager = None
        self.send_to_detokenizer = _NoOpSender()

        self._init_parallel_state(tp_worker)
        self.init_metrics(self.tp_rank, self.pp_rank, self.dp_rank)

        self._running = False
        self._aborted_request_ids: set[str] = set()
        self._pending_stream_chunks: dict[str, list[Any]] = {}
        self._pending_stream_done: set[str] = set()
        self._deferred_request_payloads: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Composition: delegate missing attributes to the upstream class
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        """Look up methods on the upstream SGLang Scheduler class.

        This gives us access to the full scheduling MRO (batch selection,
        result processing, memory checks, etc.) without inheriting.
        """
        if name == "grammar_queue":
            value = []
            self.__dict__[name] = value
            return value
        if name == "grammar_backend":
            self.__dict__[name] = None
            return None

        try:
            attr = getattr(_Upstream, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute {name!r}"
            ) from None

        # Bind unbound methods to this instance so they use our state
        if callable(attr):
            return types.MethodType(attr, self)
        return attr

    def _init_parallel_state(self, tp_worker: Any) -> None:
        enable_dp_attention = self.server_args.enable_dp_attention
        self.attn_tp_rank, self.attn_tp_size, self.attn_dp_rank = (
            compute_dp_attention_world_info(
                enable_dp_attention,
                self.tp_rank,
                self.tp_size,
                self.dp_size,
            )
        )

        self.tp_group = tp_worker.get_tp_group()
        self.tp_cpu_group = self.tp_group.cpu_group
        self.attn_tp_group = tp_worker.get_attention_tp_group()
        self.attn_tp_cpu_group = tp_worker.get_attention_tp_cpu_group()

        if enable_dp_attention:
            self.cpu_group = self.attn_tp_cpu_group
            self.entry_rank = self.attn_tp_group.first_rank
            self.is_entry_rank = self.attn_tp_rank == 0
        else:
            self.cpu_group = self.tp_cpu_group
            self.entry_rank = self.tp_group.first_rank
            self.is_entry_rank = self.tp_group.rank_in_group == 0

        self.pad_input_ids_func = tp_worker.get_pad_input_ids_func()

        self.current_scheduler_metrics_enabled = (
            self.attn_tp_rank == 0 or self.enable_metrics_for_all_schedulers
        )

    # ------------------------------------------------------------------
    # Overridden methods (take precedence over __getattr__)
    # ------------------------------------------------------------------

    def get_next_batch_to_run(self):
        batch = _Upstream.get_next_batch_to_run(self)
        if batch is not None and not self._is_batch_ready_to_run(batch):
            return None
        return batch

    def recv_requests(self):
        """Drain inbox on rank 0 and broadcast scheduler inputs to TP followers."""
        recv_msgs = self._recv_scheduler_messages()
        new_reqs: list = []
        for msg in recv_msgs:
            if msg.request_id in self._aborted_request_ids:
                continue

            if msg.type == "new_request":
                new_reqs.append(msg.data)
            elif msg.type == "stream_chunk":
                self._on_stream_chunk(msg.request_id, msg.data)
            elif msg.type == "stream_done":
                self._on_stream_done(msg.request_id)

        return new_reqs

    def _recv_scheduler_messages(self) -> list[IncomingMessage]:
        if self.tp_size == 1:
            return self._drain_local_inbox()

        recv_msgs = self._drain_local_inbox() if self.is_entry_rank else []
        return broadcast_pyobj(
            recv_msgs,
            self.tp_group.rank,
            self.tp_cpu_group,
            src=self.tp_group.ranks[0],
        )

    def _drain_local_inbox(self) -> list[IncomingMessage]:
        recv_msgs: list[IncomingMessage] = []
        while True:
            try:
                recv_msgs.append(self.inbox.get_nowait())
            except _queue_mod.Empty:
                break
        return recv_msgs

    def process_input_requests(self, recv_reqs):
        """Convert incoming payloads to SGLang Reqs and enqueue."""
        for payload in recv_reqs:
            req_id = payload.request_id
            buffered_chunks = self._pending_stream_chunks.pop(req_id, [])
            existing_chunks = list(getattr(payload, "prefetched_chunks", []) or [])
            if existing_chunks:
                existing_chunks.extend(buffered_chunks)
                payload.prefetched_chunks = existing_chunks
            else:
                payload.prefetched_chunks = buffered_chunks
            pending_stream_done = req_id in self._pending_stream_done
            payload.prefetched_stream_done = pending_stream_done
            if not self._is_request_build_ready(
                payload,
                pending_stream_done=pending_stream_done,
            ):
                self._deferred_request_payloads[req_id] = payload
                continue
            req_data = self._request_builder(payload)
            if pending_stream_done:
                self._pending_stream_done.discard(req_id)
            self._deferred_request_payloads.pop(req_id, None)
            req = req_data.req
            req._omni_data = req_data
            req_id = req.rid
            self._initialize_request_stream_state(req_data, payload)
            if req_id in self._aborted_request_ids:
                continue
            self.waiting_queue.append(req)

    def _take_deferred_request_payloads(self) -> list[Any]:
        if not self._deferred_request_payloads:
            return []
        deferred = list(self._deferred_request_payloads.values())
        self._deferred_request_payloads.clear()
        return deferred

    def _is_request_build_ready(
        self,
        payload: Any,
        *,
        pending_stream_done: bool,
    ) -> bool:
        del payload, pending_stream_done
        return True

    def _initialize_request_stream_state(self, req_data: Any, payload: Any) -> None:
        for chunk in getattr(payload, "prefetched_chunks", []) or []:
            self._append_stream_chunk(req_data, chunk)
        if bool(getattr(payload, "prefetched_stream_done", False)):
            self._mark_stream_done(req_data)

    def _is_batch_ready_to_run(self, batch: Any) -> bool:
        del batch
        return True

    def run_batch(self, batch, pp_proxy_tensors=None):
        """Run a batch through the model runner.

        The custom model runner (for example ThinkerModelRunner or a
        model-specific talker runner)
        accepts a ``SchedulerOutput`` wrapper and returns a
        ``ModelRunnerOutput``.  The upstream ``process_batch_result`` expects
        a ``GenerationBatchResult``.  We bridge the two formats here.
        """
        if self._model_runner is not None:
            from sglang.srt.managers.scheduler import GenerationBatchResult

            from sglang_omni_v1.scheduling.types import (
                SchedulerOutput,
                SchedulerRequest,
            )

            # Wrap ScheduleBatch → SchedulerOutput for the model runner
            sched_reqs = []
            for req in batch.reqs:
                rid = req.rid
                data = req._omni_data
                sched_reqs.append(SchedulerRequest(request_id=rid, data=data))
            sched_output = SchedulerOutput(requests=sched_reqs, batch_data=batch)

            mr_output = self._model_runner.execute(sched_output)

            if self._stream_output_builder is not None:
                for sched_req in sched_output.requests:
                    req_output = mr_output.outputs[sched_req.request_id]
                    stream_msg = self._stream_output_builder(
                        sched_req.request_id,
                        sched_req.data,
                        req_output,
                    )
                    if stream_msg is not None:
                        self.outbox.put(stream_msg)

            # Convert ModelRunnerOutput → GenerationBatchResult
            # The upstream process_batch_result reads .next_token_ids and
            # .logits_output from the result; both are already on batch via
            # the model runner's execute() (batch.output_ids is set there).
            return GenerationBatchResult(
                logits_output=None,
                next_token_ids=batch.output_ids,
                can_run_cuda_graph=mr_output.can_run_cuda_graph,
            )
        # Fallback: call upstream's run_batch (uses tp_worker directly)
        return _Upstream.run_batch(self, batch, pp_proxy_tensors)

    def stream_output(self, reqs, return_logprob=False, skip_req=None):
        """Intercept finished requests and emit to outbox.

        Upstream calls this after process_batch_result to send results
        to the detokenizer via ZMQ.  We capture finished requests here
        and put them in the outbox so Stage can route them downstream.
        """
        for req in reqs:
            if skip_req is not None and req is skip_req:
                continue
            if not req.finished():
                continue

            rid = req.rid

            # Build result payload from the Req
            data = req._omni_data
            data.output_ids = list(req.output_ids)
            finished_reason = req.finished_reason
            data.finish_reason = (
                finished_reason.to_json().get("type")
                if finished_reason is not None
                else None
            )
            result = self._result_adapter(data)

            self.outbox.put(
                OutgoingMessage(
                    request_id=rid,
                    type="result",
                    data=result,
                )
            )

    def send_to_tokenizer(self):
        """No-op — results are routed through the stage outbox."""
        return None

    # ------------------------------------------------------------------
    # Stream chunk handling
    # ------------------------------------------------------------------

    def _on_stream_chunk(self, request_id: str, chunk: Any) -> None:
        req_data = self._find_request_data(request_id)
        if req_data is not None:
            self._append_stream_chunk(req_data, chunk)
            return
        self._pending_stream_chunks.setdefault(request_id, []).append(chunk)

    def _on_stream_done(self, request_id: str) -> None:
        req_data = self._find_request_data(request_id)
        if req_data is not None:
            self._mark_stream_done(req_data)
            return
        self._pending_stream_done.add(request_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        if self.enable_overlap:
            self._event_loop_overlap()
        else:
            self._event_loop_normal()

    def event_loop(self) -> None:
        self.start()

    def stop(self) -> None:
        self._running = False

    def abort(self, request_id: str) -> None:
        self._aborted_request_ids.add(request_id)
        self._pending_stream_chunks.pop(request_id, None)
        self._pending_stream_done.discard(request_id)
        self._deferred_request_payloads.pop(request_id, None)
        self.waiting_queue = [
            req for req in self.waiting_queue if req.rid != request_id
        ]
        _remove_from_batch(self.running_batch, request_id)
        _remove_from_batch(self.cur_batch, request_id)
        _remove_from_batch(self.last_batch, request_id)
        self._drain_inbox_for_request(request_id)

    # ------------------------------------------------------------------
    # Event loops
    # ------------------------------------------------------------------

    def _event_loop_normal(self) -> None:
        # Note (Chenyang): yield the GIL when idle so co-located non-AR stages
        # (encoders, preprocessor) running in sibling threads aren't starved
        # of Python execution. Without this, in single-process mode the busy
        # AR scheduler loop pins the GIL and the audio_encoder forward pass
        # (which is mostly Python-side dispatch into many small CUDA kernels)
        # slows ~600x, dropping audio QPS from >10 to <0.5.
        while self._running:
            recv_reqs = self.recv_requests()
            recv_reqs.extend(self._take_deferred_request_payloads())
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                time.sleep(0.001)
                continue

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                self.self_check_during_idle()
                time.sleep(0.001)

            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.self_check_during_busy()

    def _event_loop_overlap(self) -> None:
        self.result_queue = deque()

        def pop_and_process():
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        while self._running:
            recv_reqs = self.recv_requests()
            recv_reqs.extend(self._take_deferred_request_payloads())
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch
            disable_overlap_for_batch = self.is_disable_overlap_for_batch(batch)

            if disable_overlap_for_batch and self.result_queue:
                pop_and_process()

            if batch:
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            if self.last_batch:
                if not disable_overlap_for_batch and self.result_queue:
                    pop_and_process()
            elif batch is None:
                self.self_check_during_idle()

            if self.is_generation:
                self.launch_batch_sample_if_needed(batch_result)

            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.self_check_during_busy()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _drain_inbox_for_request(self, request_id: str) -> None:
        retained: list[IncomingMessage] = []
        while True:
            try:
                msg = self.inbox.get_nowait()
            except _queue_mod.Empty:
                break
            if msg.request_id != request_id:
                retained.append(msg)
        for msg in retained:
            self.inbox.put(msg)

    def _find_request_data(self, request_id: str) -> Any | None:
        for req in self.running_batch.reqs:
            if req.rid == request_id:
                return req._omni_data
        for req in self.waiting_queue:
            if req.rid == request_id:
                return req._omni_data
        return None

    @staticmethod
    def _append_stream_chunk_default(req_data: Any, chunk: Any) -> None:
        stream_chunks = getattr(req_data, "stream_chunks", None)
        if stream_chunks is None:
            stream_chunks = deque()
            req_data.stream_chunks = stream_chunks
        stream_chunks.append(chunk)

    def _append_stream_chunk(self, req_data: Any, chunk: Any) -> None:
        if self._stream_chunk_handler is None:
            self._append_stream_chunk_default(req_data, chunk)
            return
        self._stream_chunk_handler(req_data, chunk)

    def _mark_stream_done(self, req_data: Any) -> None:
        if self._stream_done_handler is None:
            req_data.stream_done = True
            return
        self._stream_done_handler(req_data)


def _remove_from_batch(batch: Any, request_id: str) -> None:
    if batch is None:
        return
    batch.reqs = [req for req in batch.reqs if req.rid != request_id]
    if not batch.reqs:
        batch.batch_is_full = False
