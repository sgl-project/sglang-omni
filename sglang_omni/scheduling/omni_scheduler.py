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
import threading
import time
import types
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

import torch
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler as _Upstream
from sglang.srt.managers.scheduler import validate_input_length
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.utils import broadcast_pyobj

from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.profiler.event_recorder import get_active_stage as _get_active_stage
from sglang_omni.proto.admin import (
    ADMIN_CONTINUE_GENERATION,
    ADMIN_DESTROY_WEIGHTS_UPDATE_GROUP,
    ADMIN_INIT_WEIGHTS_UPDATE_GROUP,
    ADMIN_MODEL_INFO,
    ADMIN_PAUSE_GENERATION,
    ADMIN_UPDATE_WEIGHTS_FROM_DISK,
    ADMIN_UPDATE_WEIGHTS_FROM_DISTRIBUTED,
    ADMIN_UPDATE_WEIGHTS_FROM_TENSOR,
    ADMIN_WEIGHTS_CHECKER,
)
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

_FAILED_BATCH_RESULT = object()

_ABORTED_REQUEST_ID_LIMIT = 10000
_ABORTED_REQUEST_ID_RETAINED = 5000


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
        abort_callback: Callable[[str], None] | None = None,
        enable_overlap: bool = False,
        enable_async_decode: bool = False,
        async_decode_min_batch_size: int = 2,
        request_build_max_workers: int = 1,
        request_build_max_pending: int | None = None,
    ):
        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()
        self.requires_tp_work_fanout: bool = False

        # --- Request builder: StagePayload → SGLangARRequestData ----------
        self._request_builder = request_builder
        self._result_adapter = result_adapter
        self._model_runner = model_runner
        self._stream_output_builder = stream_output_builder
        self._stream_chunk_handler = stream_chunk_handler
        self._stream_done_handler = stream_done_handler
        self._abort_callback = abort_callback
        self._request_admission_lock = threading.RLock()
        self.request_build_max_workers = max(1, int(request_build_max_workers))
        if self.request_build_max_workers > 1 and int(server_args.tp_size) > 1:
            logger.warning(
                "OmniScheduler request-build workers are disabled for "
                f"tp_size={server_args.tp_size} to preserve identical request "
                "admission order on every TP rank"
            )
            self.request_build_max_workers = 1
        if self.request_build_max_workers > 1:
            max_pending = (
                self.request_build_max_workers
                if request_build_max_pending is None
                else int(request_build_max_pending)
            )
            self.request_build_max_pending = max(1, max_pending)
            self._request_build_backlog_limit = max(
                self.request_build_max_pending,
                int(server_args.max_queued_requests or 0),
            )
            self._request_build_executor: ThreadPoolExecutor | None = (
                ThreadPoolExecutor(
                    max_workers=self.request_build_max_workers,
                    thread_name_prefix="omni-request-build",
                )
            )
        else:
            self.request_build_max_pending = 0
            self._request_build_backlog_limit = 0
            self._request_build_executor = None
        self._pending_request_builds: dict[str, tuple[Any, bool, Future]] = {}
        self._backlogged_request_build_payloads: deque[Any] = deque()
        self._request_build_max_pending_observed = 0

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
        # One-step-lookahead async decode (single stream + CUDA event). Only
        # safe for model runners that implement post_decode_launch/resolve.
        self.enable_async_decode = enable_async_decode
        # Below this decode batch size the lookahead is bypassed for a plain
        # synchronous step: at low concurrency the per-step collect is too small
        # to overlap, so the lookahead's fixed overhead is a net loss (the bs=1
        # regression — see benchmark_results.md / stall_analysis.md). Default 2
        # = only bs=1 takes the fast path.
        self.async_decode_min_batch_size = int(async_decode_min_batch_size)
        if model_runner is not None:
            model_runner._async_enabled = enable_async_decode

        # Token / memory info (upstream reads from tp_worker.get_worker_info)
        mr = tp_worker.model_runner
        self.max_total_num_tokens = mr.max_total_num_tokens
        self.max_prefill_tokens = server_args.max_prefill_tokens
        self.max_running_requests = server_args.max_running_requests
        self.max_queued_requests = server_args.max_queued_requests
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
        # Async decode (one-step lookahead): the launched-but-not-resolved
        # decode batch, or None. Tracked here (not just a loop local) so abort
        # can reach the in-flight step. See _event_loop_async_decode.
        self._async_pending = None
        self.forward_ct = 0
        self.return_health_check_ct = 0
        self.num_retracted_reqs = 0
        self.num_paused_reqs = 0
        self.sessions: dict = {}
        self.forward_sleep_time = None
        self._engine_paused = False
        self._admin_lock = threading.Lock()
        self._admin_queue = _queue_mod.Queue()
        self._scheduler_thread_id: int | None = None
        self._last_pause_mode: str | None = None

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
        self._init_upstream_compat_flags(server_args)
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
        self._aborted_request_id_order: deque[str] = deque()
        self._pending_stream_chunks: dict[str, list[Any]] = {}
        self._pending_stream_done: set[str] = set()
        self._deferred_request_payloads: dict[str, Any] = {}
        self._dirty_deferred_request_ids: set[str] = set()
        self._first_emit_done: set[str] = set()
        self._prefill_start_done: set[str] = set()

    def _init_upstream_compat_flags(self, server_args: Any) -> None:
        self.enable_hisparse = bool(server_args.enable_hisparse)
        self.hisparse_coordinator = None
        self.enable_priority_preemption = bool(
            server_args.enable_priority_scheduling
            and not server_args.disable_priority_preemption
        )
        # High-water mark, not a cap. Mirrors upstream Scheduler.__init__ (sglang/srt/managers/scheduler.py).
        self.max_prefill_bs = 0
        self.use_ngram_embedding = False
        self.return_health_check_ipcs = []
        self.enable_overlap_mlx = False
        # Upstream scheduler_runtime_checker_mixin._streaming_session_count
        # iterates ``self.session_controller.sessions.values()`` during
        # report_decode_stats. We don't host SGLang's interactive-session
        # feature, so a stub with an empty sessions dict is sufficient.
        from types import SimpleNamespace

        self.session_controller = SimpleNamespace(sessions={})
        self.dllm_manager = SimpleNamespace(any_staging_reqs=lambda: False)
        device = getattr(self, "device", None)
        self.device_module = (
            torch.get_device_module(device)
            if device is not None
            else torch.get_device_module()
        )

    def self_check_during_idle(self) -> None:
        self.new_token_ratio = self.init_new_token_ratio
        idle_sleeper = self.__dict__.get("idle_sleeper")
        if idle_sleeper is not None:
            idle_sleeper.maybe_sleep()

    def self_check_during_busy(self) -> None:
        return None

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
        self._drain_request_build_results()
        recv_reqs, rejected_reqs = self._stage_request_build_payloads(recv_reqs)
        for payload in rejected_reqs:
            self._reject_request_build_backlog_overflow(payload)
        for payload in recv_reqs:
            req_id = payload.request_id
            with self._request_admission_lock:
                if (
                    req_id in self._aborted_request_ids
                    or req_id in self._pending_request_builds
                ):
                    continue
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
            active_stage = _get_active_stage()
            request_build_executor = self._request_build_executor
            if request_build_executor is not None:
                with self._request_admission_lock:
                    if (
                        req_id in self._aborted_request_ids
                        or req_id in self._pending_request_builds
                    ):
                        continue
                    future = request_build_executor.submit(
                        self._run_request_builder, payload, active_stage
                    )
                    self._pending_request_builds[req_id] = (
                        payload,
                        pending_stream_done,
                        future,
                    )
                    self._request_build_max_pending_observed = max(
                        self._request_build_max_pending_observed,
                        len(self._pending_request_builds),
                    )
                continue
            try:
                req_data = self._run_request_builder(payload, active_stage)
            except Exception as exc:
                logger.exception(f"OmniScheduler: request builder failed for {req_id}")
                self._emit_request_error(req_id, exc)
                self.abort(req_id)
                continue
            self._enqueue_built_request(payload, pending_stream_done, req_data)
        self._drain_request_build_results()

    def _run_request_builder(self, payload: Any, active_stage: str | None) -> Any:
        req_id = payload.request_id
        _emit_event(
            request_id=req_id,
            stage=active_stage,
            event_name="scheduler_request_build_start",
        )
        req_data = self._request_builder(payload)
        _emit_event(
            request_id=req_id,
            stage=active_stage,
            event_name="scheduler_request_build_end",
        )
        return req_data

    def _stage_request_build_payloads(
        self, recv_reqs: list[Any]
    ) -> tuple[list[Any], list[Any]]:
        if self._request_build_executor is None:
            return list(recv_reqs), []

        with self._request_admission_lock:
            backlog = self._backlogged_request_build_payloads
            pending_builds = self._pending_request_builds
            rejected: list[Any] = []
            backlog_ids = {payload.request_id for payload in backlog}
            capacity = max(
                0,
                self.request_build_max_pending - len(pending_builds),
            )
            selected: list[Any] = []
            selected_ids: set[str] = set()
            while capacity > 0 and backlog:
                payload = backlog.popleft()
                req_id = payload.request_id
                backlog_ids.discard(req_id)
                if req_id in self._aborted_request_ids or req_id in pending_builds:
                    continue
                selected.append(payload)
                selected_ids.add(req_id)
                capacity -= 1

            for payload in recv_reqs:
                req_id = payload.request_id
                if (
                    req_id in self._aborted_request_ids
                    or req_id in pending_builds
                    or req_id in backlog_ids
                    or req_id in selected_ids
                ):
                    continue
                if capacity > 0:
                    selected.append(payload)
                    selected_ids.add(req_id)
                    capacity -= 1
                    continue
                if len(backlog) >= self._request_build_backlog_limit:
                    rejected.append(payload)
                    continue
                backlog.append(payload)
                backlog_ids.add(req_id)
            return selected, rejected

    def _reject_request_build_backlog_overflow(self, payload: Any) -> None:
        req_id = payload.request_id
        error = RuntimeError(
            "request-build backlog is full "
            f"(backlog_limit={self._request_build_backlog_limit})"
        )
        logger.warning("Rejecting request %s before build: %s", req_id, error)
        self._emit_request_error(req_id, error)
        self.abort(req_id)

    def _drain_request_build_results(self) -> None:
        while True:
            with self._request_admission_lock:
                if not self._pending_request_builds:
                    return
                req_id, (payload, pending_stream_done, future) = next(
                    iter(self._pending_request_builds.items())
                )
                if not future.done():
                    return
                self._pending_request_builds.pop(req_id, None)
                if req_id in self._aborted_request_ids:
                    continue
            try:
                req_data = future.result()
            except Exception as exc:
                with self._request_admission_lock:
                    if req_id in self._aborted_request_ids:
                        continue
                logger.exception(f"OmniScheduler: request builder failed for {req_id}")
                self._emit_request_error(req_id, exc)
                self.abort(req_id)
                continue
            with self._request_admission_lock:
                if req_id in self._aborted_request_ids:
                    continue
                self._enqueue_built_request(
                    payload,
                    pending_stream_done,
                    req_data,
                    request_admission_lock_held=True,
                )

    def _enqueue_built_request(
        self,
        payload: Any,
        pending_stream_done: bool,
        req_data: Any,
        *,
        request_admission_lock_held: bool = False,
    ) -> None:
        req_id = payload.request_id
        if pending_stream_done:
            self._pending_stream_done.discard(req_id)
        self._deferred_request_payloads.pop(req_id, None)
        req = req_data.req
        req._omni_data = req_data
        req_id = req.rid
        if bool(getattr(req_data, "enforce_request_limits", False)):
            error_msg = self._prepare_request_limits(req_data)
            if error_msg:
                self._emit_request_error(req_id, ValueError(error_msg))
                self.abort(req_id)
                return
        kv_error = self._request_kv_capacity_error(req)
        if kv_error is not None:
            logger.warning(f"Rejecting request {req_id} before scheduling: {kv_error}")
            self._emit_request_error(req_id, ValueError(kv_error))
            self.abort(req_id)
            return
        self._initialize_request_stream_state(req_data, payload)
        for chunk in self._pending_stream_chunks.pop(req_id, []) or []:
            self._append_stream_chunk(req_data, chunk)
        if req_id in self._pending_stream_done:
            self._pending_stream_done.discard(req_id)
            self._mark_stream_done(req_data)

        def enqueue_if_live() -> None:
            if req_id in self._aborted_request_ids:
                return
            _emit_event(
                request_id=req_id,
                stage=None,
                event_name="scheduler_queue_enter",
            )
            self.waiting_queue.append(req)

        if request_admission_lock_held:
            enqueue_if_live()
        else:
            with self._request_admission_lock:
                enqueue_if_live()

    def _prepare_request_limits(self, req_data: Any) -> str | None:
        req = req_data.req
        self.init_req_max_new_tokens(req)
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            allow_auto_truncate=False,
        )
        if error_msg:
            return error_msg
        if hasattr(req_data, "max_new_tokens"):
            req_data.max_new_tokens = int(req.sampling_params.max_new_tokens)
        return None

    def _take_deferred_request_payloads(self) -> list[Any]:
        if not self._dirty_deferred_request_ids:
            return []
        deferred: list[Any] = []
        for req_id in list(self._dirty_deferred_request_ids):
            payload = self._deferred_request_payloads.pop(req_id, None)
            if payload is not None:
                deferred.append(payload)
        self._dirty_deferred_request_ids.clear()
        return deferred

    def _should_recheck_deferred_request_on_stream_chunk(
        self, request_id: str, chunk: Any
    ) -> bool:
        del request_id, chunk
        return True

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

    def _request_kv_capacity_error(self, req: Any) -> str | None:
        input_len = len(req.origin_input_ids)
        max_new_tokens = int(req.sampling_params.max_new_tokens or 0)
        required_tokens = input_len + max_new_tokens
        kv_capacity = int(self.max_req_len)
        if required_tokens <= kv_capacity:
            return None

        mem_fraction = self.server_args.mem_fraction_static
        if mem_fraction is not None:
            mem_hint = (
                f" Current mem_fraction_static is {mem_fraction:.3f}; try setting "
                "--thinker-mem-fraction-static higher."
            )
        else:
            mem_hint = " Try setting a higher --thinker-mem-fraction-static value."

        return (
            "Request requires more tokens than the thinker KV cache can hold "
            f"(input_tokens={input_len}, max_new_tokens={max_new_tokens}, "
            f"required_tokens={required_tokens}, kv_capacity={kv_capacity})."
            f"{mem_hint}"
        )

    def _emit_request_error(self, request_id: str, error: Exception) -> None:
        if not getattr(self, "is_entry_rank", True):
            return
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="error",
                data=error,
            )
        )

    def run_batch(self, batch, pp_proxy_tensors=None):
        try:
            return self._run_batch(batch, pp_proxy_tensors)
        except Exception as exc:
            self._handle_batch_failure(batch, exc)
            return _FAILED_BATCH_RESULT

    def _run_batch(self, batch, pp_proxy_tensors=None):
        """Run a batch through the model runner.

        The custom model runner (for example ThinkerModelRunner or a
        model-specific talker runner)
        accepts a ``SchedulerOutput`` wrapper and returns a
        ``ModelRunnerOutput``.  The upstream ``process_batch_result`` expects
        a ``GenerationBatchResult``.  We bridge the two formats here.
        """
        self._emit_prefill_start_for_batch(batch)
        if self._model_runner is not None:
            # Mirror upstream run_batch's per-forward counter: OmniScheduler
            # overrides run_batch, so without this forward_ct stays 0 and
            # SGLANG_TEST_RETRACT fires every step. Only the custom-runner path
            # needs it (the fallback reaches upstream run_batch, which counts).
            self.forward_ct = getattr(self, "forward_ct", 0) + 1
            sched_output = self._build_sched_output(batch)
            mr_output = self._model_runner.execute(sched_output)
            self._emit_stream_output(sched_output, mr_output)
            return self._make_batch_result(batch, mr_output)
        # Fallback: call upstream's run_batch (uses tp_worker directly)
        return _Upstream.run_batch(self, batch, pp_proxy_tensors)

    def _build_sched_output(self, batch):
        """Wrap a ScheduleBatch into the SchedulerOutput the model runner
        expects. Shared by the sync and async (launch) paths."""
        from sglang_omni.scheduling.types import SchedulerOutput, SchedulerRequest

        sched_reqs = [
            SchedulerRequest(request_id=req.rid, data=req._omni_data)
            for req in batch.reqs
        ]
        return SchedulerOutput(requests=sched_reqs, batch_data=batch)

    def _emit_stream_output(self, sched_output, mr_output, skip_rids=()) -> None:
        """Emit per-request stream chunks from a ModelRunnerOutput. Shared by
        the sync and async (resolve) paths. ``skip_rids`` suppresses emission
        for requests already finished in an earlier step (the lookahead
        overrun) — emitting their extra chunk would corrupt the downstream
        vocoder's delayed-code stream. Aborted requests are suppressed for the
        same reason: an abort landing mid-step must not ship one more chunk."""
        if self._stream_output_builder is None:
            return
        for sched_req in sched_output.requests:
            rid = sched_req.request_id
            if rid in skip_rids or rid in self._aborted_request_ids:
                continue
            req_output = mr_output.outputs[rid]
            emitted_any = False
            for msg in self._stream_output_builder(rid, sched_req.data, req_output):
                if not emitted_any:
                    if rid not in self._first_emit_done:
                        self._first_emit_done.add(rid)
                        _emit_event(
                            request_id=rid,
                            stage=None,
                            event_name="scheduler_first_emit",
                        )
                    emitted_any = True
                self.outbox.put(msg)

    @staticmethod
    def _make_batch_result(batch, mr_output):
        # process_batch_result reads .next_token_ids / .logits_output; the
        # model runner already set batch.output_ids during execute/resolve.
        from sglang.srt.managers.scheduler import GenerationBatchResult

        next_token_ids = batch.output_ids
        if isinstance(next_token_ids, torch.Tensor):
            batch.input_ids = next_token_ids.to(torch.int64)
        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=next_token_ids,
            can_run_cuda_graph=mr_output.can_run_cuda_graph,
        )

    def _run_batch_launch(self, batch):
        """Async: build SchedulerOutput and launch the decode step on the GPU
        (forward + sample, then ``post_decode_launch`` publishes the resolve
        payload), without waiting. Returns ``(sched_output, pending_step)``; the
        caller holds the pending step (launch-first keeps two steps in flight)."""
        self._emit_prefill_start_for_batch(batch)
        # One forward per launch; mirror upstream run_batch's per-forward
        # counter (the matching resolve does no forward, so it must not count).
        self.forward_ct = getattr(self, "forward_ct", 0) + 1
        sched_output = self._build_sched_output(batch)
        pending_step = self._model_runner.execute_launch(sched_output)
        return sched_output, pending_step

    def _run_batch_resolve(self, batch, sched_output, pending_step, skip_rids=()):
        """Async: resolve the given launched step (wait event, host collect),
        emit its stream chunks (except overrun reqs in ``skip_rids``), and
        return its GenerationBatchResult.

        next_token_ids comes from the resolved step's own batch_result, not
        ``batch.output_ids`` — the running batch's output_ids was already
        consumed (reset to None) by the next step's prepare_for_decode.
        """
        from sglang.srt.managers.scheduler import GenerationBatchResult

        mr_output = self._model_runner.execute_resolve(pending_step)
        if mr_output is None:
            return _FAILED_BATCH_RESULT
        self._emit_stream_output(sched_output, mr_output, skip_rids=skip_rids)
        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=pending_step.batch_result.next_token_ids,
            can_run_cuda_graph=mr_output.can_run_cuda_graph,
        )

    def _handle_batch_failure(self, batch: Any, error: Exception) -> None:
        reqs = list(batch.reqs)
        request_ids = [req.rid for req in reqs]
        logger.exception("OmniScheduler batch failed for requests=%s", request_ids)
        for req in reqs:
            self._emit_request_error(req.rid, error)
            self.abort(req.rid, defer_running_cleanup=False)

    def _emit_prefill_start_for_batch(self, batch: ScheduleBatch) -> None:
        """Emit once when a request's first executable batch is selected."""
        metadata = {
            "is_prefill_only": bool(batch.is_prefill_only),
            "is_extend_in_batch": bool(batch.is_extend_in_batch),
        }
        for req in batch.reqs:
            rid = req.rid
            if rid in self._prefill_start_done:
                continue
            self._prefill_start_done.add(rid)
            _emit_event(
                request_id=rid,
                stage=None,
                event_name="scheduler_prefill_start",
                metadata=metadata,
            )

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
            if rid in self._aborted_request_ids:
                # note (Gaokai): an abort landing mid-step finishes here via
                # FINISH_ABORT; run the cleanup abort() deferred (callbacks are
                # idempotent) and drop the stale terminal result so it cannot
                # resurrect the request downstream.
                if self._abort_callback is not None:
                    try:
                        self._abort_callback(rid)
                    except Exception:
                        logger.exception(
                            "OmniScheduler: abort cleanup failed for %s", rid
                        )
                self._first_emit_done.discard(rid)
                self._prefill_start_done.discard(rid)
                continue

            # Build result payload from the Req
            data = req._omni_data
            data.output_ids = list(req.output_ids)
            data.weight_version = self.server_args.weight_version
            finished_reason = req.finished_reason
            data.finish_reason = (
                finished_reason.to_json().get("type")
                if finished_reason is not None
                else None
            )
            try:
                result = self._result_adapter(data)
            except Exception as exc:
                logger.exception(
                    "OmniScheduler result adapter failed for request %s", rid
                )
                self._first_emit_done.discard(rid)
                self._prefill_start_done.discard(rid)
                self._emit_request_error(rid, exc)
                continue
            finally:
                data.prefill_input_embeds = None
                data.decode_input_embeds = None

            self._first_emit_done.discard(rid)
            self._prefill_start_done.discard(rid)
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

    def _on_stream_chunk(self, request_id: str, chunk: Any) -> None:
        req_data = self._find_request_data(request_id)
        if req_data is not None:
            self._append_stream_chunk(req_data, chunk)
            return
        self._pending_stream_chunks.setdefault(request_id, []).append(chunk)
        if (
            request_id in self._deferred_request_payloads
            and self._should_recheck_deferred_request_on_stream_chunk(request_id, chunk)
        ):
            self._dirty_deferred_request_ids.add(request_id)

    def _on_stream_done(self, request_id: str) -> None:
        req_data = self._find_request_data(request_id)
        if req_data is not None:
            self._mark_stream_done(req_data)
            return
        self._pending_stream_done.add(request_id)
        if request_id in self._deferred_request_payloads:
            self._dirty_deferred_request_ids.add(request_id)

    def start(self) -> None:
        self._scheduler_thread_id = threading.get_ident()
        self._running = True
        try:
            if getattr(self, "enable_async_decode", False):
                self._event_loop_async_decode()
            elif self.enable_overlap:
                self._event_loop_overlap()
            else:
                self._event_loop_normal()
        finally:
            self._scheduler_thread_id = None
            self._shutdown_request_build_executor()

    def event_loop(self) -> None:
        self.start()

    def stop(self) -> None:
        self._running = False

    def _shutdown_request_build_executor(self) -> None:
        executor = self._request_build_executor
        if executor is None:
            return
        executor.shutdown(wait=False, cancel_futures=True)
        self._request_build_executor = None

    def abort(self, request_id: str, *, defer_running_cleanup: bool = True) -> None:
        running_abort = (
            self._mark_running_request_aborted(request_id)
            if defer_running_cleanup
            else False
        )
        with self._request_admission_lock:
            if request_id not in self._aborted_request_ids:
                if len(self._aborted_request_ids) >= _ABORTED_REQUEST_ID_LIMIT:
                    # note (Gaokai): evict oldest-first so a still-quiescing
                    # abort survives.
                    while (
                        len(self._aborted_request_ids) >= _ABORTED_REQUEST_ID_RETAINED
                    ):
                        self._aborted_request_ids.discard(
                            self._aborted_request_id_order.popleft()
                        )
                self._aborted_request_ids.add(request_id)
                self._aborted_request_id_order.append(request_id)
            pending = self._pending_request_builds.pop(request_id, None)
            if pending is not None:
                pending[2].cancel()
            if self._backlogged_request_build_payloads:
                retained = [
                    payload
                    for payload in self._backlogged_request_build_payloads
                    if payload.request_id != request_id
                ]
                self._backlogged_request_build_payloads.clear()
                self._backlogged_request_build_payloads.extend(retained)
            self.waiting_queue = [
                req for req in self.waiting_queue if req.rid != request_id
            ]
        if self._abort_callback is not None and not running_abort:
            try:
                self._abort_callback(request_id)
            except Exception:
                logger.exception(
                    "OmniScheduler: abort cleanup failed for %s", request_id
                )
        self._pending_stream_chunks.pop(request_id, None)
        self._pending_stream_done.discard(request_id)
        self._deferred_request_payloads.pop(request_id, None)
        self._dirty_deferred_request_ids.discard(request_id)
        self._first_emit_done.discard(request_id)
        self._prefill_start_done.discard(request_id)
        if not running_abort:
            self._release_immediate_request_resources(request_id)
            _remove_from_batch(self.running_batch, request_id)
            _remove_from_batch(self.cur_batch, request_id)
            _remove_from_batch(self.last_batch, request_id)
            _remove_from_batch(self._async_pending_batch(), request_id)
        self._drain_inbox_for_request(request_id)

    def admin(
        self, action: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload = dict(payload or {})
        if self._should_enqueue_admin():
            return self._enqueue_admin(action, payload)
        return self._run_admin_action(action, payload)

    def _should_enqueue_admin(self) -> bool:
        scheduler_thread_id = getattr(self, "_scheduler_thread_id", None)
        return (
            bool(getattr(self, "_running", False))
            and scheduler_thread_id is not None
            and threading.get_ident() != scheduler_thread_id
        )

    def _enqueue_admin(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        timeout_s = float(payload.get("_admin_timeout_s", 300.0))
        queued_payload = dict(payload)
        queued_payload.pop("_admin_timeout_s", None)
        response_queue = _queue_mod.Queue(maxsize=1)
        self._admin_queue.put((action, queued_payload, response_queue))
        try:
            return response_queue.get(timeout=timeout_s)
        except _queue_mod.Empty:
            return {
                "success": False,
                "message": f"admin operation timed out after {timeout_s:.1f}s",
                "error": "admin operation timed out",
            }

    def _process_admin_requests(self) -> int:
        processed = 0
        while True:
            try:
                action, payload, response_queue = self._admin_queue.get_nowait()
            except _queue_mod.Empty:
                break
            try:
                response = self._run_admin_action(action, payload)
            except Exception as exc:
                logger.exception("OmniScheduler admin operation failed: %s", action)
                response = {
                    "success": False,
                    "message": str(exc),
                    "error": str(exc),
                }
            response_queue.put(response)
            processed += 1
        return processed

    def _run_admin_action(
        self, action: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload = dict(payload or {})
        if action == ADMIN_MODEL_INFO:
            return self._admin_model_info()
        if action == ADMIN_PAUSE_GENERATION:
            return self._admin_pause_generation(payload)
        if action == ADMIN_CONTINUE_GENERATION:
            return self._admin_continue_generation(payload)
        if action == ADMIN_UPDATE_WEIGHTS_FROM_DISK:
            return self._admin_update_weights_from_disk(payload)
        if action == ADMIN_UPDATE_WEIGHTS_FROM_TENSOR:
            return self._admin_update_weights_from_tensor(payload)
        if action == ADMIN_UPDATE_WEIGHTS_FROM_DISTRIBUTED:
            return self._admin_update_weights_from_distributed(payload)
        if action == ADMIN_INIT_WEIGHTS_UPDATE_GROUP:
            return self._admin_init_weights_update_group(payload)
        if action == ADMIN_DESTROY_WEIGHTS_UPDATE_GROUP:
            return self._admin_destroy_weights_update_group(payload)
        if action == ADMIN_WEIGHTS_CHECKER:
            return self._admin_weights_checker(payload)
        return {
            "success": True,
            "message": f"unsupported admin action: {action}",
            "data": {"skipped": True, "unsupported": True},
        }

    def _admin_model_info(self) -> dict[str, Any]:
        info = {}
        if hasattr(self.model_worker, "model_info"):
            info.update(self.model_worker.model_info())
        with self._request_admission_lock:
            request_build_pending = len(self._pending_request_builds)
            request_build_backlog = len(self._backlogged_request_build_payloads)
            waiting_queue_size = len(self.waiting_queue)
        info.update(
            {
                "stage_tp_rank": self.tp_rank,
                "stage_tp_size": self.tp_size,
                "engine_paused": self._engine_paused,
                "waiting_queue_size": waiting_queue_size,
                "request_build_workers": self.request_build_max_workers,
                "request_build_pending": request_build_pending,
                "request_build_max_pending": self.request_build_max_pending,
                "request_build_backlog": request_build_backlog,
                "request_build_max_pending_observed": (
                    self._request_build_max_pending_observed
                ),
                "running_batch_size": len(
                    getattr(self.running_batch, "reqs", []) or []
                ),
                "model_path": self.server_args.model_path,
                "load_format": self.server_args.load_format,
                "weight_version": self.server_args.weight_version,
            }
        )
        return {"success": True, "message": "ok", "data": info}

    def _admin_pause_generation(self, payload: dict[str, Any]) -> dict[str, Any]:
        mode = str(payload.get("mode") or "abort")
        if mode not in {"abort", "retract", "in_place"}:
            return {
                "success": False,
                "message": f"invalid pause mode: {mode}",
                "error": f"invalid pause mode: {mode}",
            }

        with self._admin_lock:
            self._engine_paused = True
            self._last_pause_mode = mode
            self._resolve_pending_async()
            self._resolve_pending_overlap_results()
            num_paused = 0
            if mode == "abort":
                num_paused = self._abort_all_requests()
            elif mode == "retract":
                num_paused = self._retract_running_requests()
        return {
            "success": True,
            "message": "generation paused",
            "data": {
                "mode": mode,
                "num_paused_requests": num_paused,
                "engine_paused": self._engine_paused,
            },
        }

    def _admin_continue_generation(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._admin_lock:
            if bool(payload.get("torch_empty_cache", True)):
                self._empty_torch_cache()
            self._engine_paused = False
            self._last_pause_mode = None
        return {
            "success": True,
            "message": "generation continued",
            "data": {"engine_paused": self._engine_paused},
        }

    def _admin_update_weights_from_disk(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        if not hasattr(self.model_worker, "update_weights_from_disk"):
            return {
                "success": True,
                "message": "stage does not support update_weights_from_disk",
                "data": {"skipped": True, "unsupported": True},
            }
        return self._run_weight_update_with_lifecycle(
            payload,
            self.model_worker.update_weights_from_disk,
            {
                "model_path": payload.get("model_path"),
                "weight_version": payload.get("weight_version"),
                "token_step": payload.get("token_step"),
            },
        )

    def _run_weight_update_with_lifecycle(
        self,
        payload: dict[str, Any],
        update_fn,
        result_data: dict[str, Any],
        *,
        keep_pause_on_failure: bool = False,
    ) -> dict[str, Any]:
        keep_pause = bool(payload.get("keep_pause", False))
        keep_engine_paused = keep_pause
        with self._admin_lock:
            previous_pause_state = self._engine_paused
            self._engine_paused = True
            try:
                self._resolve_pending_async()
                self._resolve_pending_overlap_results()
                num_paused = 0
                abort_all_requests = bool(payload.get("abort_all_requests", False))
                if abort_all_requests:
                    num_paused = self._abort_all_requests()
                else:
                    active_request_ids = self._active_request_ids()
                    if active_request_ids and not self._can_update_active_requests(
                        previous_pause_state
                    ):
                        if not keep_pause:
                            self._engine_paused = previous_pause_state
                        return {
                            "success": False,
                            "message": (
                                "active requests are present; set "
                                "abort_all_requests=true or pause_generation with "
                                "mode=retract before updating weights"
                            ),
                            "error": "active requests present during weight update",
                            "data": {
                                "active_request_count": len(active_request_ids),
                                "active_request_ids": active_request_ids[:16],
                                "abort_all_requests": abort_all_requests,
                                "pause_mode": getattr(self, "_last_pause_mode", None),
                                "engine_paused": self._engine_paused,
                            },
                        }

                try:
                    success, message = update_fn(payload)
                except Exception:
                    if keep_pause_on_failure:
                        keep_engine_paused = True
                    raise
                flush_success: bool | None = None
                if success and bool(payload.get("flush_cache", True)):
                    flush_success = self._flush_cache_after_update()
                    success = success and bool(flush_success)
                    if not flush_success:
                        message = f"{message}; cache flush failed"

                if keep_pause_on_failure and not success:
                    keep_engine_paused = True
                if bool(payload.get("torch_empty_cache", False)):
                    self._empty_torch_cache()
            finally:
                if keep_engine_paused:
                    self._engine_paused = True
                else:
                    self._engine_paused = previous_pause_state

        data = {
            "num_paused_requests": num_paused,
            "flush_cache": payload.get("flush_cache", True),
            "flush_success": flush_success,
            "keep_pause": keep_pause,
            "engine_paused": self._engine_paused,
        }
        data.update(result_data)
        return {
            "success": bool(success),
            "message": str(message),
            "data": data,
            "error": None if success else str(message),
        }

    def _admin_update_weights_from_tensor(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        if not hasattr(self.model_worker, "update_weights_from_tensor"):
            return {
                "success": True,
                "message": "stage does not support update_weights_from_tensor",
                "data": {"skipped": True, "unsupported": True},
            }
        with self._admin_lock:
            success, message = self.model_worker.update_weights_from_tensor(payload)
        return {
            "success": bool(success),
            "message": str(message),
            "data": {
                "metadata_only": payload.get("serialized_named_tensors") is None,
            },
            "error": None if success else str(message),
        }

    def _admin_update_weights_from_distributed(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        if not hasattr(self.model_worker, "update_weights_from_distributed"):
            return {
                "success": True,
                "message": "stage does not support update_weights_from_distributed",
                "data": {"skipped": True, "unsupported": True},
            }
        return self._run_weight_update_with_lifecycle(
            payload,
            self.model_worker.update_weights_from_distributed,
            {
                "group_name": payload.get("group_name"),
                "names": payload.get("names", []),
            },
            keep_pause_on_failure=True,
        )

    def _admin_init_weights_update_group(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        if not hasattr(self.model_worker, "init_weights_update_group"):
            return {
                "success": True,
                "message": "stage does not support init_weights_update_group",
                "data": {"skipped": True, "unsupported": True},
            }
        # Note (Xuesong): init blocks on a NCCL/TCP rendezvous and runs on the
        # scheduler serving thread (admin is drained inline in the event loop), so
        # the serving loop is frozen until the trainer (rank 0) joins. sglang's
        # init_weights_update_group exposes no timeout, so a missing trainer
        # stalls inference up to NCCL's own timeout. Call this only in
        # coordination with the trainer (the router takes the worker out of
        # routing for the duration).
        with self._admin_lock:
            success, message = self.model_worker.init_weights_update_group(payload)
        return {
            "success": bool(success),
            "message": str(message),
            "data": {
                "group_name": payload.get("group_name"),
                "world_size": payload.get("world_size"),
                "rank_offset": payload.get("rank_offset"),
            },
            "error": None if success else str(message),
        }

    def _admin_destroy_weights_update_group(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        if not hasattr(self.model_worker, "destroy_weights_update_group"):
            return {
                "success": True,
                "message": "stage does not support destroy_weights_update_group",
                "data": {"skipped": True, "unsupported": True},
            }
        with self._admin_lock:
            success, message = self.model_worker.destroy_weights_update_group(payload)
        return {
            "success": bool(success),
            "message": str(message),
            "data": {"group_name": payload.get("group_name")},
            "error": None if success else str(message),
        }

    def _admin_weights_checker(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not hasattr(self.model_worker, "weights_checker"):
            return {
                "success": True,
                "message": "stage does not support weights_checker",
                "data": {"skipped": True, "unsupported": True},
            }
        action = str(payload.get("action") or "checksum")
        with self._admin_lock:
            data = self.model_worker.weights_checker(action)
        return {"success": True, "message": "ok", "data": data}

    def _abort_all_requests(self) -> int:
        request_ids = self._active_request_ids()
        for request_id in request_ids:
            self.abort(request_id, defer_running_cleanup=False)
        return len(request_ids)

    def _active_request_ids(self) -> list[str]:
        request_ids: set[str] = set()
        with self._request_admission_lock:
            if self._pending_request_builds:
                request_ids.update(self._pending_request_builds.keys())
            if self._backlogged_request_build_payloads:
                request_ids.update(
                    payload.request_id
                    for payload in self._backlogged_request_build_payloads
                    if payload.request_id not in self._aborted_request_ids
                )
            for req in self.waiting_queue:
                rid = getattr(req, "rid", None)
                if rid is not None:
                    request_ids.add(rid)
        for batch in (
            self.running_batch,
            self.cur_batch,
            self.last_batch,
            self._async_pending_batch(),
        ):
            if batch is None:
                continue
            for req in batch.reqs:
                if req.rid is not None and not req.finished():
                    request_ids.add(req.rid)
        return sorted(request_ids)

    def _can_update_active_requests(
        self, previously_paused: bool | None = None
    ) -> bool:
        engine_paused = (
            self._engine_paused if previously_paused is None else previously_paused
        )
        return bool(
            engine_paused and getattr(self, "_last_pause_mode", None) == "retract"
        )

    def _retract_running_requests(self) -> int:
        batch = self.running_batch
        if batch is None or batch.is_empty():
            return 0
        batch.filter_batch(v1_spec_info_filtered=True)
        if len(batch.reqs) == 0:
            return 0
        retracted_reqs = batch.retract_all(self.server_args)
        add_to_queue = getattr(self, "_add_request_to_queue", None)
        for req in retracted_reqs:
            if callable(add_to_queue):
                add_to_queue(req)
            else:
                self.waiting_queue.append(req)
        batch.batch_is_full = False
        self.chunked_req = None
        return len(retracted_reqs)

    def _flush_cache_after_update(self) -> bool:
        try:
            return bool(self.flush_cache())
        except Exception:
            logger.exception("flush_cache after weight update failed")
            return False

    def _resolve_pending_overlap_results(self) -> None:
        result_queue = getattr(self, "result_queue", None)
        if result_queue is None:
            return
        while result_queue:
            batch, result = result_queue.popleft()
            self.process_batch_result(batch, result)

    @staticmethod
    def _empty_torch_cache() -> None:
        if not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()

    def _mark_running_request_aborted(self, request_id: str) -> bool:
        marked = False
        seen: set[int] = set()
        for batch in (
            self.running_batch,
            self.cur_batch,
            self.last_batch,
            self._async_pending_batch(),
        ):
            if batch is None or id(batch) in seen:
                continue
            seen.add(id(batch))
            for req in batch.reqs:
                if req.rid != request_id or req.finished():
                    continue
                req.to_finish = FINISH_ABORT()
                marked = True
        return marked

    def _release_immediate_request_resources(self, request_id: str) -> None:
        seen: set[int] = set()
        for batch in (
            self.running_batch,
            self.cur_batch,
            self.last_batch,
            self._async_pending_batch(),
        ):
            if batch is None:
                continue
            for req in batch.reqs:
                if req.rid != request_id or id(req) in seen:
                    continue
                seen.add(id(req))
                self._release_request_kv_cache(req)

    def _release_request_kv_cache(self, req: Any) -> None:
        if req.req_pool_idx is None and req.mamba_pool_idx is None:
            return
        release_kv_cache(req, self.tree_cache)

    def _event_loop_normal(self) -> None:
        # Note (Chenyang): yield the GIL when idle so co-located non-AR stages
        # (encoders, preprocessor) running in sibling threads aren't starved
        # of Python execution. Without this, in single-process mode the busy
        # AR scheduler loop pins the GIL and the audio_encoder forward pass
        # (which is mostly Python-side dispatch into many small CUDA kernels)
        # slows ~600x, dropping audio QPS from >10 to <0.5.
        while self._running:
            self._process_admin_requests()
            recv_reqs = self.recv_requests()
            recv_reqs.extend(self._take_deferred_request_payloads())
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                self._process_admin_requests()
                time.sleep(0.001)
                continue

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                if result is not _FAILED_BATCH_RESULT:
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
            self._process_admin_requests()
            recv_reqs = self.recv_requests()
            recv_reqs.extend(self._take_deferred_request_payloads())
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                self._process_admin_requests()
                time.sleep(0.001)
                continue

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch
            disable_overlap_for_batch = self.is_disable_overlap_for_batch(batch)

            if disable_overlap_for_batch and self.result_queue:
                pop_and_process()

            if batch:
                batch_result = self.run_batch(batch)
                if batch_result is not _FAILED_BATCH_RESULT:
                    self.result_queue.append((batch.copy(), batch_result))
                else:
                    batch_result = None
            else:
                batch_result = None

            if self.last_batch:
                if not disable_overlap_for_batch and self.result_queue:
                    pop_and_process()
            elif batch is None:
                self.self_check_during_idle()
                time.sleep(0.001)

            if self.is_generation:
                self.launch_batch_sample_if_needed(batch_result)

            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.self_check_during_busy()

    @staticmethod
    def _batch_is_decode(batch: ScheduleBatch) -> bool:
        mode = batch.forward_mode
        if mode is None:
            return False
        if mode.is_decode():
            return True
        return not bool(mode.is_extend())

    def _async_pending_batch(self):
        """The in-flight (launched, not yet resolved) decode batch, or None.

        ``getattr`` with default so abort paths stay safe even for schedulers
        built without going through ``__init__`` (e.g. unit-test fixtures).
        ``_async_pending`` is ``(batch, sched_output, pending_step)`` or None.
        """
        pending = getattr(self, "_async_pending", None)
        return pending[0] if pending is not None else None

    def _resolve_and_process(self, batch, sched_output, pending_step) -> None:
        """Resolve a launched step and feed it to process_batch_result, after
        dropping requests that already finished in an earlier step.

        Lookahead overrun: a request that finishes at step S is still present in
        step S+1's (already-launched) batch — its S+1 output is discarded by the
        collect's ``_cg_was_done`` skip, but upstream process_batch_result would
        re-free its KV. So drop reqs that were ALREADY finished in an earlier
        step (and their next_token_ids rows) from this lagged batch.

        Crucially, snapshot finished-state BEFORE the resolve: a req that
        finishes *during* this step's collect (e.g. an EOC finish, which
        _mark_sampler_finished sets) must be KEPT so process_batch_result emits
        it — only reqs finished in a *prior* step are the overrun to drop.
        """
        # A request retracted at step S is still in step S+1's lagged batch;
        # drop it like a prior-step finish so its KV is not re-freed.
        pre_finished = [
            r.finished() or bool(getattr(r, "is_retracted", False)) for r in batch.reqs
        ]
        # rids finished/retracted in a prior step (overrun): suppress their emit
        skip_rids = {batch.reqs[i].rid for i, was in enumerate(pre_finished) if was}
        result = self._run_batch_resolve(
            batch, sched_output, pending_step, skip_rids=skip_rids
        )
        if result is _FAILED_BATCH_RESULT:
            return
        keep = [i for i, was_finished in enumerate(pre_finished) if not was_finished]
        if len(keep) < len(batch.reqs):
            if result.next_token_ids is not None and keep:
                idx = torch.tensor(keep, device=result.next_token_ids.device)
                result.next_token_ids = result.next_token_ids[idx]
            # Drop overrun reqs from the batch. NOT filter_batch(): batch is a
            # ScheduleBatch.copy() which omits seq_lens (it carries only the
            # fields process_batch_result needs). process_batch_result_decode
            # zips batch.reqs with next_token_ids and uses Req attributes (not
            # positional batch tensors), so trimming reqs in lockstep suffices.
            batch.reqs = [batch.reqs[i] for i in keep]
        if batch.reqs:
            self.process_batch_result(batch, result)

    def _resolve_pending_async(self) -> None:
        """Resolve + process the in-flight decode step, if any. Used to flush
        before prefill / pause / shutdown so a launched step is never stranded.
        """
        if self._async_pending is None:
            return
        batch, sched_output, pending_step = self._async_pending
        self._async_pending = None
        try:
            self._resolve_and_process(batch, sched_output, pending_step)
        except Exception as exc:
            self._handle_batch_failure(batch, exc)

    def _free_overrun_step_slots(self, out_cache_loc, drop_indices) -> None:
        """Free the per-step decode KV slot for rows whose request finished or
        retracted in a prior step (the lookahead overrun): ``prepare_for_decode``
        allocated it but ``cache_finished_req`` truncates below it, so it leaks.

        Only under RadixCache + page_size=1; ChunkCache/paged already free the slot
        with the request, so compensating here would double-free — hence the gate.
        """
        if not drop_indices:
            return
        if self.page_size != 1 or self.server_args.disable_radix_cache:
            return
        if out_cache_loc is None:
            logger.warning("overrun step-slot free skipped: out_cache_loc is None")
            return
        assert max(drop_indices) < out_cache_loc.numel(), (
            f"overrun drop index {max(drop_indices)} out of range "
            f"({out_cache_loc.numel()} step slots)"
        )
        idx = torch.tensor(drop_indices, dtype=torch.long, device=out_cache_loc.device)
        self.token_to_kv_pool_allocator.free(out_cache_loc[idx])

    def _drop_stale_overrun(self, batch):
        """Drop reqs finished OR retracted by the just-completed drain from the
        stale fast-path batch, so run_batch does not forward/finalize them again
        (double-free of already-freed KV). Returns the filtered batch, or None if
        it empties. Mirrors the finished/is_retracted pre-drop in
        _resolve_and_process; the fast path previously dropped only finished.
        """
        if batch is None or not batch.reqs:
            return batch
        drop = [
            r.finished() or bool(getattr(r, "is_retracted", False)) for r in batch.reqs
        ]
        if not any(drop):
            return batch
        keep = [i for i, d in enumerate(drop) if not d]
        out_cache_loc = batch.out_cache_loc
        self._free_overrun_step_slots(
            out_cache_loc, [i for i, d in enumerate(drop) if d]
        )
        batch.filter_batch(keep_indices=keep)
        if out_cache_loc is not None:
            batch.out_cache_loc = out_cache_loc[keep]
        return batch if batch.reqs else None

    def _event_loop_async_decode(self) -> None:
        """One-step-lookahead decode loop (single stream + CUDA event).

        Each iteration LAUNCHES the current decode step (GPU forward + on-GPU
        sample, then ``post_decode_launch`` publishes the resolve payload, no GPU
        wait) and THEN RESOLVES the previous step's host-side collect, so the
        resolve host work overlaps the current step's GPU forward (launch-first,
        D1 in design.md section 1.3). Prefill / empty batches flush any in-flight
        decode first and run synchronously (the in-flight step is never stranded).
        """
        while self._running:
            self._process_admin_requests()
            recv_reqs = self.recv_requests()
            recv_reqs.extend(self._take_deferred_request_payloads())
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                self._process_admin_requests()
                self._resolve_pending_async()
                time.sleep(0.001)
                continue

            if (
                self._async_pending is not None
                and self.is_mixed_chunk
                and (
                    self.chunked_req is not None
                    or (self.waiting_queue and not self.running_batch.batch_is_full)
                )
            ):
                self._resolve_pending_async()

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            # Route through sync when the runner's collect has a sync-only
            # fallback (default True for runners not overriding lookahead_eligible).
            runner = getattr(self, "_model_runner", None)
            use_lookahead = (
                batch is not None
                and len(batch.reqs) >= self.async_decode_min_batch_size
                and self._batch_is_decode(batch)
                and (runner is None or runner.lookahead_eligible(batch))
            )

            if use_lookahead:
                try:
                    sched_output, pending_step = self._run_batch_launch(batch)
                except Exception as exc:
                    self._handle_batch_failure(batch, exc)
                else:
                    prev_pending = self._async_pending
                    self._async_pending = (batch.copy(), sched_output, pending_step)
                    if prev_pending is not None:
                        pb, ps, pstep = prev_pending
                        try:
                            self._resolve_and_process(pb, ps, pstep)
                        except Exception as exc:
                            self._handle_batch_failure(pb, exc)
            else:
                # Fast path (low-concurrency decode below the threshold) +
                # prefill + empty all land here: flush any in-flight lookahead
                # step first (preserve ordering — this is also the bs>=2 -> bs=1
                # drain transition), then run this batch synchronously. Bypassing
                # the lookahead at bs=1 avoids its fixed per-step overhead, which
                # at low concurrency has no overlap payoff (the bs=1 regression).
                # Skip the drain call entirely in the common no-pending case (the
                # bs=1 steady state) — _resolve_pending_async would just no-op.
                if self._async_pending is not None:
                    self._resolve_pending_async()
                    # Stale-batch overrun: `batch` was built (get_next_batch_to_run,
                    # top of loop) BEFORE this drain, which can finish OR retract reqs
                    # still present in it. Drop them before run_batch so they are not
                    # forwarded/finalized a second time (double-free of already-freed
                    # KV). Fast-path analogue of the _resolve_and_process drop.
                    batch = self._drop_stale_overrun(batch)
                    self.cur_batch = batch
                if batch:
                    result = self.run_batch(batch)
                    if result is not _FAILED_BATCH_RESULT:
                        self.process_batch_result(batch, result)
                else:
                    self.self_check_during_idle()
                    time.sleep(0.001)

            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.self_check_during_busy()

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
        # Scan all batches a live req can sit in during prefill→decode handoff.
        for batch in (self.running_batch, self.cur_batch, self.last_batch):
            if batch is None:
                continue
            for req in batch.reqs:
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
