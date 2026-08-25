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
from array import array
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import islice
from typing import Any, Callable

import torch
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    NextBatchPlan,
    ScheduleBatch,
    retract_all,
)
from sglang.srt.managers.scheduler import Scheduler as _Upstream
from sglang.srt.managers.scheduler import validate_input_length
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.utils import broadcast_pyobj

from sglang_omni.admission import QueueFullError
from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.profiler.event_recorder import (
    emit_model_path_end as _emit_model_path_end,
)
from sglang_omni.profiler.event_recorder import (
    emit_model_path_start as _emit_model_path_start,
)
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
from sglang_omni.scheduling.types import DeferredAdmission
from sglang_omni.vendor.sglang.parallel_state import create_parallel_state
from sglang_omni.vendor.sglang.server_args import override_server_args
from sglang_omni.vendor.sglang.signature import supported_kwargs

logger = logging.getLogger(__name__)

_FAILED_BATCH_RESULT = object()

_ABORTED_REQUEST_ID_LIMIT = 10000
_ABORTED_REQUEST_ID_RETAINED = 5000
_COMPLETED_REQUEST_ID_LIMIT = 10000
_PENDING_STREAM_REQUEST_LIMIT = 10000
_PENDING_STREAM_REQUEST_RETAINED = 5000


class _PendingStreamIngress:
    """Stream input buffered for a request the scheduler has not admitted."""

    __slots__ = ("chunks", "done")

    def __init__(self) -> None:
        self.chunks: list[Any] = []
        self.done = False


def _detach_request_data(req: Any) -> None:
    """Break Req -> data; async snapshots retain the one-way data -> Req edge."""
    req._omni_data = None


class _NoOpSender:
    """Stub for send_to_detokenizer — stream_output handles emission."""

    def send_output(self, *args, **kwargs):
        pass


class _UpstreamAbortSender:
    """Translate upstream scheduler abort notifications into stage output."""

    def __init__(self, scheduler: OmniScheduler) -> None:
        self._scheduler = scheduler

    def send_output(self, msg: Any, req: Any = None) -> None:
        del req
        if not isinstance(msg, AbortReq):
            raise RuntimeError(
                f"Unexpected upstream scheduler IPC output: {type(msg).__name__}"
            )

        request_id = msg.rid
        finished_reason = msg.finished_reason
        message = (
            finished_reason.get("message")
            if isinstance(finished_reason, dict)
            else None
        )
        if message is None:
            message = msg.abort_message or "Request aborted by the scheduler"

        scheduler = self._scheduler
        scheduler._emit_request_error(request_id, RuntimeError(message))
        scheduler.abort(request_id, defer_running_cleanup=False)


class _OmniIpcChannels:
    """Subset of upstream SchedulerIpcChannels reachable from Omni."""

    def __init__(self, scheduler: OmniScheduler) -> None:
        self.send_to_tokenizer = _UpstreamAbortSender(scheduler)
        self.send_to_detokenizer = scheduler.send_to_detokenizer


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
        request_finished_callback: Callable[[str], None] | None = None,
        enable_overlap: bool = False,
        enable_async_decode: bool = False,
        async_decode_min_batch_size: int = 2,
        prefill_coalesce_requests: int = 0,
        prefill_coalesce_wait_ms: float = 60.0,
        prefill_coalesce_when_idle: bool = False,
        prefill_coalesce_requires_pending_builds: bool = False,
        prefill_coalesce_after_builds_during_decode: bool = False,
        request_build_max_workers: int = 1,
        request_build_max_pending: int | None = None,
        shutdown_callback: Callable[[], None] | None = None,
    ):
        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()
        self.requires_tp_work_fanout: bool = False

        # --- Request builder: StagePayload → SGLangARRequestData ----------
        self._request_builder = request_builder
        self._result_adapter = result_adapter
        self._model_runner = None
        self._stream_output_builder = stream_output_builder
        self._stream_chunk_handler = stream_chunk_handler
        self._stream_done_handler = stream_done_handler
        self._abort_callback = abort_callback
        self._request_finished_callback = request_finished_callback
        self._shutdown_callback = shutdown_callback
        self._shutdown_lock = threading.Lock()
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
            max_queued_requests = int(server_args.max_queued_requests or 0)
            self._request_build_backlog_limit = (
                max(self.request_build_max_pending, max_queued_requests)
                if max_queued_requests > 0
                else None
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
        self._pending_request_admissions: dict[
            str, tuple[Any, bool, DeferredAdmission]
        ] = {}
        self._backlogged_request_build_payloads: deque[Any] = deque()
        self._request_build_max_pending_observed = 0

        # --- Core scheduling state (read/written by upstream methods) -----
        self.server_args = server_args
        self.model_config = model_config
        self.gpu_id = tp_worker.gpu_id
        self.tp_rank = tp_worker.tp_rank
        self.tp_size = server_args.tp_size
        self.pp_rank = 0
        self.pp_size = server_args.pp_size
        self.dp_rank = None
        self.dp_size = server_args.dp_size
        self.moe_ep_rank = 0
        self.moe_ep_size = 1
        self.moe_dp_rank = None
        self.moe_dp_size = server_args.moe_dp_size
        self.attn_cp_rank = 0
        self.attn_cp_size = server_args.attn_cp_size
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
        if self.enable_overlap and self.enable_async_decode:
            raise ValueError(
                "enable_overlap and enable_async_decode are mutually "
                "exclusive: the async loop would run a batch-result processor "
                "built for the overlap contract and leak KV for finished "
                "requests"
            )

        # Range and type are enforced at configuration validation
        # (FactoryArgs); only the TP interaction is this scheduler's call.
        requests = int(prefill_coalesce_requests)
        wait_ms = float(prefill_coalesce_wait_ms)
        if requests > 1 and int(server_args.tp_size) > 1:
            logger.warning(
                "Prefill admission coalescing is disabled for "
                f"tp_size={server_args.tp_size}: the wait deadline reads each "
                "rank's local clock, so ranks could disagree on expiry and "
                "break lockstep scheduling"
            )
            requests = 0
        self.prefill_coalesce_requests = requests
        self.prefill_coalesce_wait_s = wait_ms / 1e3
        self.prefill_coalesce_when_idle = bool(prefill_coalesce_when_idle)
        self.prefill_coalesce_requires_pending_builds = bool(
            prefill_coalesce_requires_pending_builds
        )
        self.prefill_coalesce_after_builds_during_decode = bool(
            prefill_coalesce_after_builds_during_decode
        )

        # Token / memory info (upstream reads from tp_worker.get_worker_info)
        mr = tp_worker.model_runner
        self.max_total_num_tokens = mr.max_total_num_tokens
        self.max_prefill_tokens = server_args.max_prefill_tokens
        self.max_running_requests = mr.max_running_requests
        self.max_queued_requests = server_args.max_queued_requests
        effective_max_total_num_tokens = mr.effective_max_total_num_tokens
        self.max_req_len = min(
            server_args.context_length - 1,
            effective_max_total_num_tokens - 1,
        )
        self.max_req_input_len = self.max_req_len - 1
        self.random_seed = tp_worker.random_seed
        self.device = tp_worker.device
        # Hybrid-SWA per-layer capacities: upstream sources these from its
        # kv_cache_builder; no Omni model serves hybrid-SWA, so they stay None.
        self.full_tokens_per_layer = None
        self.swa_tokens_per_layer = None
        self.min_free_slots_delayer = None
        self.enable_fpm = False

        # Global server_args field upstream sets in its __init__
        from sglang.srt.server_args import get_global_server_args

        gsa = get_global_server_args()
        if gsa is not None and gsa.pp_max_micro_batch_size is None:
            override_server_args(
                gsa,
                "sglang_omni.scheduler.pp_max_micro_batch_size_default",
                pp_max_micro_batch_size=max(
                    self.max_running_requests // self.pp_size,
                    1,
                ),
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
        if self.chunked_prefill_size is not None and self.chunked_prefill_size <= 0:
            self.chunked_prefill_size = None
        self.chunked_req = None
        self._pending_chunked_abort_req = None
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
        from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
            NewTokenRatioTracker,
        )

        self.new_token_ratio_tracker = NewTokenRatioTracker.from_server_args(
            server_args
        )
        self.prefill_delayer = None
        self.lora_drainer = None

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
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        self.spec_algorithm = SpeculativeAlgorithm.NONE
        self.dllm_config = None
        self.draft_worker = None
        self._execution_bridge = None
        if model_runner is not None:
            self.bind_model_runner(model_runner)

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
        self.ipc_channels = _OmniIpcChannels(self)
        self.init_metrics_collector(self.tp_rank, self.pp_rank, self.dp_rank)
        self.init_metrics_reporter(self.tp_rank, self.pp_rank, self.dp_rank)
        self._init_upstream_scheduler_components()

        self._running = False
        self._aborted_request_ids: set[str] = set()
        self._aborted_request_id_order: deque[str] = deque()
        # Normal completion closes stream ingress for the request. Keep a
        # bounded tombstone window so chunks already in transport cannot turn
        # back into pre-admission state after Req ownership is released.
        self._completed_request_ids: dict[str, None] = {}
        # Keyed by first-touch arrival: dict order lets the overload eviction
        # drop oldest-first.
        self._pending_stream_ingress: dict[str, _PendingStreamIngress] = {}
        self._deferred_request_payloads: dict[str, Any] = {}
        self._dirty_deferred_request_ids: set[str] = set()
        self._first_emit_done: set[str] = set()
        self._prefill_start_done: set[str] = set()
        self._prefill_end_done: set[str] = set()

    def bind_model_runner(self, model_runner: Any) -> None:
        """Attach a custom runner and its SGLang execution-contract bridge.

        Some pipelines need the scheduler-owned outbox before they can build
        their model runner. They must use this method instead of assigning
        ``_model_runner`` so late-bound runners receive the same execution
        bridge and FutureMap contract as runners supplied to ``__init__``.
        """
        if model_runner is None:
            raise ValueError("model_runner must not be None")
        if self._model_runner is model_runner and self._execution_bridge is not None:
            return
        if self._model_runner is not None:
            raise RuntimeError("OmniScheduler model runner is already bound")

        from sglang_omni.model_runner.sglang_execution import SGLangExecutionBridge

        bridge = SGLangExecutionBridge(
            device=torch.device(self.device),
            worker=self.tp_worker,
            req_to_token_pool=self.req_to_token_pool,
            spec_algorithm=self.spec_algorithm,
        )
        model_runner._async_enabled = self.enable_async_decode
        model_runner.bind_execution_bridge(bridge)
        # Keep the upstream attribute available to delegated scheduler methods,
        # but make the custom ModelRunner the sole owner of relay.
        self._model_runner = model_runner
        self._execution_bridge = bridge
        self.future_map = bridge.future_map

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

        # Instance state introduced by SGLang 0.5.16's Scheduler.__init__. We
        # borrow upstream methods rather than inheriting, so anything they read
        # off ``self`` has to be mirrored here or __getattr__ raises.
        # init_req_max_new_tokens() clamps against this one.
        self.max_new_tokens_limit = envs.SGLANG_MAX_NEW_TOKENS_LIMIT.get()
        self.cur_batch_for_debug = None
        # get_next_batch_to_run() calls prepare_for_forward() on this
        # unconditionally, so it must be a real manager, not None. Upstream
        # takes it from the model runner; no Omni model uses ngram embedding
        # (see use_ngram_embedding above), so a disabled passthrough is correct.
        from sglang.srt.model_executor.model_runner_components.ngram_embedding_manager import (  # noqa: E501
            NgramEmbeddingManager,
        )

        self.ngram_embedding_manager = NgramEmbeddingManager(
            enabled=False, table=None, n=0, k=0
        )
        # Upstream pool_stats_observer.streaming_session_count iterates
        # self.session_controller.sessions.values() during decode stats
        # reporting. We don't host SGLang's interactive-session feature, so a
        # stub with an empty sessions dict is sufficient.
        from types import SimpleNamespace

        self.session_controller = SimpleNamespace(sessions={})
        self.dllm_manager = SimpleNamespace(any_staging_reqs=lambda: False)
        self.load_snapshot_writer = None
        self.kv_events_publisher = SimpleNamespace(
            emit_kv_metrics=lambda: None,
            publish_kv_events=lambda: None,
        )
        self.device_module = torch.get_device_module(self.device)

    def _init_upstream_scheduler_components(self) -> None:
        """Install the scheduler components required by upstream hot paths."""
        from sglang.srt.managers.scheduler_components.batch_result_processor import (
            SchedulerBatchResultProcessor,
        )
        from sglang.srt.managers.scheduler_components.dp_attn import (
            SchedulerDPAttnAdapter,
        )
        from sglang.srt.managers.scheduler_components.load_inquirer import (
            SchedulerLoadInquirer,
        )
        from sglang.srt.managers.scheduler_components.logprob_result_processor import (
            SchedulerLogprobResultProcessor,
        )
        from sglang.srt.managers.scheduler_components.pool_stats_observer import (
            SchedulerPoolStatsObserver,
        )

        dp_attn_kwargs = dict(
            tp_group=self.tp_group,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            offload_tags=self.offload_tags,
            ps=self.ps,
            server_args=self.server_args,
            model_config=self.model_config,
            enable_overlap=self.enable_overlap,
            spec_algorithm=self.spec_algorithm,
            get_require_mlp_sync=lambda: self.require_mlp_sync,
        )
        dp_attn_kwargs.update(
            supported_kwargs(
                SchedulerDPAttnAdapter,
                model_runner=self._model_runner,
            )
        )
        self.dp_attn_adapter = SchedulerDPAttnAdapter(**dp_attn_kwargs)
        self.pool_stats_observer = SchedulerPoolStatsObserver(
            tree_cache=self.tree_cache,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            req_to_token_pool=self.req_to_token_pool,
            session_controller=self.session_controller,
            hisparse_coordinator=self.hisparse_coordinator,
            is_hybrid_swa=self.is_hybrid_swa,
            is_hybrid_ssm=self.is_hybrid_ssm,
            enable_hisparse=self.enable_hisparse,
            full_tokens_per_layer=self.full_tokens_per_layer,
            swa_tokens_per_layer=self.swa_tokens_per_layer,
            max_total_num_tokens=(
                self.max_total_num_tokens * self.server_args.dcp_size
            ),
            get_last_batch=lambda: self.last_batch,
            get_running_batch=lambda: self.running_batch,
        )
        empty_queue = types.SimpleNamespace(queue=[], retracted_queue=[])
        self.total_prefill_uncached_tokens = 0
        self.total_prefill_busy_us = 0
        self.decode_moment_totals: list[float] = [0.0] * 6
        self._prev_step = None
        self._sched_idled = False
        load_inquirer_kwargs = dict(
            disaggregation_mode=self.disaggregation_mode,
            ps=self.ps,
            server_args=self.server_args,
            max_total_num_tokens=self.max_total_num_tokens,
            max_running_requests=self.max_running_requests,
            pool_stats_observer=self.pool_stats_observer,
            tp_worker=self.tp_worker,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            spec_algorithm=self.spec_algorithm,
            get_running_batch=lambda: self.running_batch,
            get_waiting_queue=lambda: self.waiting_queue,
            get_stats=lambda: self.metrics_reporter.stats,
            get_chunked_req=lambda: self.chunked_req,
            get_disagg_prefill_bootstrap_queue=lambda: empty_queue,
            get_disagg_prefill_inflight_queue=lambda: [],
            get_disagg_decode_prealloc_queue=lambda: empty_queue,
            get_disagg_decode_transfer_queue=lambda: empty_queue,
            get_spec_total_num_accept_tokens=lambda: (
                self.metrics_reporter.spec_total_num_accept_tokens
            ),
            get_spec_total_num_forward_ct=lambda: (
                self.metrics_reporter.spec_total_num_forward_ct
            ),
        )
        current_load_metrics = {
            "get_total_prefill_uncached_tokens": (
                lambda: self.total_prefill_uncached_tokens
            ),
            "get_total_prefill_busy_us": lambda: self.total_prefill_busy_us,
            "get_decode_moment_totals": lambda: self.decode_moment_totals,
        }
        load_inquirer_kwargs.update(
            supported_kwargs(SchedulerLoadInquirer, **current_load_metrics)
        )
        self.load_inquirer = SchedulerLoadInquirer(**load_inquirer_kwargs)
        self.output_streamer = types.SimpleNamespace(
            stream_output=self.stream_output,
            _stream_output_generation=lambda reqs, return_logprob, **_kwargs: self.stream_output(
                reqs, return_logprob
            ),
        )
        self.batch_result_processor = SchedulerBatchResultProcessor(
            is_generation=self.is_generation,
            disaggregation_mode=self.disaggregation_mode,
            enable_overlap=self.enable_overlap,
            enable_overlap_mlx=self.enable_overlap_mlx,
            server_args=self.server_args,
            model_config=self.model_config,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            hisparse_coordinator=self.hisparse_coordinator,
            req_to_token_pool=self.req_to_token_pool,
            decode_offload_manager=self.decode_offload_manager,
            metrics_collector=self.metrics_collector,
            metrics_reporter=self.metrics_reporter,
            draft_worker=self.draft_worker,
            model_worker=self.model_worker,
            logprob_result_processor=SchedulerLogprobResultProcessor(
                **supported_kwargs(
                    SchedulerLogprobResultProcessor,
                    server_args=self.server_args,
                    model_config=self.model_config,
                )
            ),
            output_streamer=self.output_streamer,
            abort_request=lambda request: self.abort(request.rid),
        )

    def self_check_during_idle(self) -> None:
        self.new_token_ratio_tracker.reset()
        idle_sleeper = self.idle_sleeper
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
        (
            self.attn_tp_rank,
            self.attn_tp_size,
            self.attn_dp_rank,
            self.attn_dp_size,
        ) = compute_dp_attention_world_info(
            enable_dp_attention,
            self.tp_rank,
            self.tp_size,
            self.dp_size,
            self.attn_cp_size,
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
        self._refresh_upstream_parallel_state()

    def _refresh_upstream_parallel_state(self) -> None:
        """Build the rank container expected by upstream scheduler methods."""
        from sglang.srt.distributed.parallel_state_wrapper import ParallelState

        self.ps = create_parallel_state(
            ParallelState,
            tp_rank=self.tp_rank,
            dcp_size=self.server_args.dcp_size,
            tp_size=self.tp_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            dp_rank=self.dp_rank,
            dp_size=self.dp_size,
            attn_tp_rank=self.attn_tp_rank,
            attn_tp_size=self.attn_tp_size,
            attn_cp_rank=self.attn_cp_rank,
            attn_cp_size=self.attn_cp_size,
            attn_dp_rank=self.attn_dp_rank,
            attn_dp_size=self.attn_dp_size,
            moe_ep_rank=self.moe_ep_rank,
            moe_ep_size=self.moe_ep_size,
            moe_dp_rank=self.moe_dp_rank,
            moe_dp_size=self.moe_dp_size,
            gpu_id=self.gpu_id,
        )

    def recv_requests(self):
        """Drain inbox on rank 0 and broadcast scheduler inputs to TP followers."""
        recv_msgs = self._recv_scheduler_messages()
        new_reqs: list = []
        for msg in recv_msgs:
            if msg.request_id in self._aborted_request_ids:
                continue

            if msg.type == "new_request":
                self._completed_request_ids.pop(msg.request_id, None)
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
        self._drain_request_admission_results()
        self._drain_request_build_results()
        recv_reqs, rejected = self._stage_request_build_payloads(recv_reqs)
        for payload in rejected:
            self._reject_queue_full(payload)
        for payload in recv_reqs:
            req_id = payload.request_id
            with self._request_admission_lock:
                if (
                    req_id in self._aborted_request_ids
                    or req_id in self._pending_request_builds
                    or req_id in self._pending_request_admissions
                ):
                    continue
            if self._waiting_queue_is_full():
                self._reject_queue_full(payload)
                continue
            ingress = self._pending_stream_ingress.get(req_id)
            buffered_chunks: list[Any] = []
            if ingress is not None and ingress.chunks:
                # Move chunks onto the payload; the entry (and its done flag)
                # stays until the built request consumes it, so a deferred
                # recheck re-derives prefetched_stream_done from the same spot.
                buffered_chunks = ingress.chunks
                ingress.chunks = []
            existing_chunks = list(payload.prefetched_chunks)
            if existing_chunks:
                existing_chunks.extend(buffered_chunks)
                payload.prefetched_chunks = existing_chunks
            else:
                payload.prefetched_chunks = buffered_chunks
            pending_stream_done = ingress.done if ingress is not None else False
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
                        or req_id in self._pending_request_admissions
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
            self._admit_or_defer_built_request(payload, pending_stream_done, req_data)
        self._drain_request_build_results()
        self._drain_request_admission_results()

    def request_build_queue_fits_workers(self) -> bool:
        """True when pending+backlog still fits in the request-build pool.

        Without a build executor the scheduler loop must stay free, so this
        is False and admission stays deferred.
        """
        if self._request_build_executor is None:
            return False
        with self._request_admission_lock:
            queued = len(self._pending_request_builds) + len(
                self._backlogged_request_build_payloads
            )
        return queued <= self.request_build_max_workers

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

    def _sleep_during_idle(self) -> None:
        with self._request_admission_lock:
            request_admission_pending = bool(
                self._pending_request_builds or self._pending_request_admissions
            )
        time.sleep(0.0001 if request_admission_pending else 0.001)

    def _queued_admission_count(self) -> int:
        return (
            len(self.waiting_queue)
            + len(self._pending_request_builds)
            + len(self._pending_request_admissions)
            + len(self._backlogged_request_build_payloads)
            + len(self._deferred_request_payloads)
        )

    def _waiting_queue_is_full(self) -> bool:
        if self.max_queued_requests is None:
            return False
        return self._queued_admission_count() >= int(self.max_queued_requests)

    def _reject_queue_full(self, payload: Any) -> None:
        req_id = payload.request_id
        logger.warning(
            "Rejecting request %s before build: %s", req_id, QueueFullError.MESSAGE
        )
        self._emit_request_error(req_id, QueueFullError())
        self.abort(req_id)

    def _stage_request_build_payloads(
        self, recv_reqs: list[Any]
    ) -> tuple[list[Any], list[Any]]:
        if self._request_build_executor is None:
            return list(recv_reqs), []

        with self._request_admission_lock:
            backlog = self._backlogged_request_build_payloads
            pending_builds = self._pending_request_builds
            pending_admissions = self._pending_request_admissions
            rejected: list[Any] = []
            if self._waiting_queue_is_full():
                while backlog:
                    payload = backlog.popleft()
                    if payload.request_id not in self._aborted_request_ids:
                        rejected.append(payload)
                rejected.extend(
                    payload
                    for payload in recv_reqs
                    if payload.request_id not in self._aborted_request_ids
                    and payload.request_id not in pending_builds
                )
                return [], rejected

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
                if (
                    req_id in self._aborted_request_ids
                    or req_id in pending_builds
                    or req_id in pending_admissions
                ):
                    continue
                selected.append(payload)
                selected_ids.add(req_id)
                capacity -= 1

            used = self._queued_admission_count() + len(selected)
            queued_limit = self.max_queued_requests
            for payload in recv_reqs:
                req_id = payload.request_id
                if (
                    req_id in self._aborted_request_ids
                    or req_id in pending_builds
                    or req_id in pending_admissions
                    or req_id in backlog_ids
                    or req_id in selected_ids
                ):
                    continue
                if queued_limit is not None and used >= int(queued_limit):
                    rejected.append(payload)
                    continue
                if capacity > 0:
                    selected.append(payload)
                    selected_ids.add(req_id)
                    capacity -= 1
                    used += 1
                    continue
                if (
                    self._request_build_backlog_limit is not None
                    and len(backlog) >= self._request_build_backlog_limit
                ):
                    rejected.append(payload)
                    continue
                backlog.append(payload)
                backlog_ids.add(req_id)
                used += 1
            return selected, rejected

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
                self._admit_or_defer_built_request(
                    payload,
                    pending_stream_done,
                    req_data,
                    request_admission_lock_held=True,
                )

    def _admit_or_defer_built_request(
        self,
        payload: Any,
        pending_stream_done: bool,
        result: Any,
        *,
        request_admission_lock_held: bool = False,
    ) -> None:
        if not isinstance(result, DeferredAdmission):
            self._enqueue_built_request(
                payload,
                pending_stream_done,
                result,
                request_admission_lock_held=request_admission_lock_held,
            )
            return

        req_id = payload.request_id

        def admit_or_hold() -> None:
            if req_id in self._aborted_request_ids:
                return
            if not result.ready.done():
                self._pending_request_admissions[req_id] = (
                    payload,
                    pending_stream_done,
                    result,
                )
                return
            try:
                result.ready.result()
            except Exception as exc:
                logger.exception(
                    "OmniScheduler: deferred request admission failed for %s",
                    req_id,
                )
                self._emit_request_error(req_id, exc)
                self.abort(req_id)
                return
            self._enqueue_built_request(
                payload,
                pending_stream_done,
                result.value,
                request_admission_lock_held=True,
            )

        if request_admission_lock_held:
            admit_or_hold()
        else:
            with self._request_admission_lock:
                admit_or_hold()

    def _drain_request_admission_results(self) -> None:
        with self._request_admission_lock:
            ready_request_ids = [
                req_id
                for req_id, (_, _, deferred) in self._pending_request_admissions.items()
                if deferred.ready.done()
            ]
            for req_id in ready_request_ids:
                pending = self._pending_request_admissions.pop(req_id, None)
                if pending is None or req_id in self._aborted_request_ids:
                    continue
                payload, pending_stream_done, deferred = pending
                self._admit_or_defer_built_request(
                    payload,
                    pending_stream_done,
                    deferred,
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
        self._deferred_request_payloads.pop(req_id, None)
        req = req_data.req
        self._normalize_req_token_arrays(req)
        req_id = req.rid
        if req_data.enforce_request_limits:
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
        ingress = self._pending_stream_ingress.pop(req_id, None)
        if ingress is not None:
            for chunk in ingress.chunks:
                self._append_stream_chunk(req_data, chunk)
            if ingress.done and not pending_stream_done:
                self._mark_stream_done(req_data)

        def enqueue_if_live() -> None:
            if req_id in self._aborted_request_ids:
                return
            # note (guozhihao): Priority defaulting must run before the queued-limit abort.
            if not self._set_or_validate_priority(req):
                return
            if self._abort_on_queued_limit(req):
                logger.warning(
                    "Rejecting request %s: waiting queue is full "
                    "(max_queued_requests=%s, waiting=%s)",
                    req_id,
                    self.max_queued_requests,
                    len(self.waiting_queue),
                )
                return
            _emit_event(
                request_id=req_id,
                stage=None,
                event_name="scheduler_queue_enter",
            )
            req._coalesce_enqueue_t = time.perf_counter()
            req._omni_terminal_claimed = False
            req._omni_data = req_data
            self.waiting_queue.append(req)

        if request_admission_lock_held:
            enqueue_if_live()
        else:
            with self._request_admission_lock:
                enqueue_if_live()

    @staticmethod
    def _normalize_req_token_arrays(req: Any) -> None:
        """Normalize builder-produced token containers to the 0.5.16 Req shape."""
        origin_input_ids = req.origin_input_ids
        if not isinstance(origin_input_ids, array):
            req.origin_input_ids = array("q", origin_input_ids)

        unpadded = req.origin_input_ids_unpadded
        if unpadded is origin_input_ids:
            req.origin_input_ids_unpadded = req.origin_input_ids
        elif not isinstance(unpadded, array):
            req.origin_input_ids_unpadded = array("q", unpadded)

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
        for chunk in payload.prefetched_chunks:
            self._append_stream_chunk(req_data, chunk)
        if payload.prefetched_stream_done:
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
        if not self.is_entry_rank:
            return
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="error",
                data=error,
            )
        )

    def get_next_batch_to_run(self):
        """Bridge Omni's batch-owning loops to the 0.5.16 scheduler contract.

        0.5.16 stopped reading ``running_batch``/``last_batch`` off ``self`` and
        now returns a ``NextBatchPlan`` instead of the batch. Omni's event loops
        own that state, so feed it in and write the (possibly rebuilt) running
        batch back before handing the runnable batch to the caller.
        """
        plan = _Upstream.get_next_batch_to_run(
            self, self.running_batch, self.last_batch
        )
        self.running_batch = plan.running_batch
        return plan.batch_to_run

    def get_new_batch_prefill(self, running_batch):
        # Note: (maydomine) batch prefill admissions to amortize the fixed step
        # cost; the oldest-request deadline survives partial admission and aborts.
        #
        # 0.5.16 passes ``running_batch`` in and expects a ``NextBatchPlan`` back,
        # so the coalesce hold-off returns an empty plan rather than None.
        if self.prefill_coalesce_requests <= 1 or self.chunked_req is not None:
            return _Upstream.get_new_batch_prefill(self, running_batch)
        decode_is_idle = running_batch is None or running_batch.is_empty()
        if not self.prefill_coalesce_when_idle and decode_is_idle:
            return _Upstream.get_new_batch_prefill(self, running_batch)
        if self.prefill_coalesce_requires_pending_builds:
            with self._request_admission_lock:
                build_work_pending = bool(
                    self._pending_request_builds
                    or self._pending_request_admissions
                    or self._backlogged_request_build_payloads
                )
            if not build_work_pending and not (
                self.prefill_coalesce_after_builds_during_decode and not decode_is_idle
            ):
                return _Upstream.get_new_batch_prefill(self, running_batch)
        waiting = self.waiting_queue
        if not waiting or len(waiting) >= self.prefill_coalesce_requests:
            return _Upstream.get_new_batch_prefill(self, running_batch)
        now = time.perf_counter()
        oldest = now
        for req in waiting:
            t = getattr(req, "_coalesce_enqueue_t", None)
            if t is None:
                t = req._coalesce_enqueue_t = now
            oldest = min(oldest, t)
        if now - oldest >= self.prefill_coalesce_wait_s:
            return _Upstream.get_new_batch_prefill(self, running_batch)
        return NextBatchPlan(batch_to_run=None, running_batch=running_batch)

    def run_batch(self, batch, pp_proxy_tensors=None):
        try:
            return self._run_batch(batch, pp_proxy_tensors)
        except Exception as exc:
            self._handle_batch_failure(batch, exc)
            return _FAILED_BATCH_RESULT

    def _stamp_batch_launch(self, batch) -> None:
        """Mirror upstream per-forward bookkeeping for custom runner paths."""
        self.forward_ct += 1
        batch.forward_iter = self.forward_ct
        batch.launch_ts = time.monotonic()
        batch.after_idle_gap = self._sched_idled
        self._sched_idled = False

    def _run_batch(self, batch, pp_proxy_tensors=None):
        """Run a batch through the model runner.

        The custom model runner (for example ThinkerModelRunner or a
        model-specific talker runner)
        accepts a ``SchedulerOutput`` wrapper and returns a
        ``ModelRunnerOutput``.  The upstream ``process_batch_result`` expects
        a ``GenerationBatchResult``.  We bridge the two formats here.
        """
        del pp_proxy_tensors
        self._emit_prefill_start_for_batch(batch)
        self._stamp_batch_launch(batch)
        sched_output = self._build_sched_output(batch)
        mr_output = self._model_runner.execute(sched_output)
        self._emit_prefill_end_for_batch(batch)
        self._emit_stream_output(sched_output, mr_output)
        return self._make_batch_result(mr_output)

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
            self._put_stream_messages(
                rid,
                self._stream_output_builder(rid, sched_req.data, req_output),
            )

    def _put_stream_messages(self, request_id: str, messages: Any) -> None:
        emitted_any = False
        for msg in messages:
            if not emitted_any:
                if request_id not in self._first_emit_done:
                    self._first_emit_done.add(request_id)
                    _emit_event(
                        request_id=request_id,
                        stage=None,
                        event_name="scheduler_first_emit",
                    )
                emitted_any = True
            self.outbox.put(msg)

    def _flush_stream_output(self, request_id: str, req_data: Any) -> None:
        stream_output_builder = self._stream_output_builder
        if stream_output_builder is None:
            return
        flush = getattr(stream_output_builder, "flush", None)
        if flush is None:
            return
        self._put_stream_messages(request_id, flush(request_id, req_data))

    @staticmethod
    def _make_batch_result(mr_output):
        # process_batch_result reads reporting tokens. The next-forward GPU
        # token rail is independently published through FutureMap.
        from sglang.srt.managers.scheduler import GenerationBatchResult

        # Note (wenyao): reuse the runner-staged pinned host copy so the mixin's
        # .tolist() is host-only. The GPU FutureMap relay independently drives
        # the next-forward input chain under the 0.5.16 execution contract.
        next_token_ids = mr_output.next_token_ids
        if mr_output.host_token_ids is not None:
            next_token_ids = mr_output.host_token_ids
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
        self._stamp_batch_launch(batch)
        sched_output = self._build_sched_output(batch)
        pending_step = self._model_runner.execute_launch(sched_output)
        return sched_output, pending_step

    def _run_batch_resolve(self, batch, sched_output, pending_step, skip_rids=()):
        """Async: resolve the given launched step (wait event, host collect),
        emit its stream chunks (except overrun reqs in ``skip_rids``), and
        return its GenerationBatchResult.

        next_token_ids comes from the resolved step's own batch_result; the
        live batch carries no token side channel under the 0.5.16 FutureMap
        contract.
        """
        from sglang.srt.managers.scheduler import GenerationBatchResult

        mr_output = self._model_runner.execute_resolve(pending_step)
        if mr_output is None:
            return _FAILED_BATCH_RESULT
        self._emit_stream_output(sched_output, mr_output, skip_rids=skip_rids)
        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=mr_output.next_token_ids,
            can_run_cuda_graph=mr_output.can_run_cuda_graph,
        )

    def _handle_batch_failure(self, batch: Any, error: Exception) -> None:
        reqs = list(batch.reqs)
        request_ids = [req.rid for req in reqs]
        logger.exception("OmniScheduler batch failed for requests=%s", request_ids)
        for req in reqs:
            self._emit_request_error(req.rid, error)
            self._emit_model_path_end_once(req.rid, status="error")
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
            _emit_model_path_start(rid)
            _emit_event(
                request_id=rid,
                stage=None,
                event_name="scheduler_prefill_start",
                metadata=metadata,
            )

    def _emit_prefill_end_for_batch(self, batch: ScheduleBatch) -> None:
        """Emit once after a request's first executed batch returns.

        Paired with ``scheduler_prefill_start`` this frames the request's
        first model forward — for multimodal models that is encoder plus
        prefill — for streaming and non-streaming requests alike. The
        metadata carries the realized batch size (issue #1324 Q-PR2).
        """
        # note (luojiaxuan): steady-state decode reaches here after every
        # step. _prefill_end_done only ever holds rids present in
        # _prefill_start_done and both are discarded together, so equal sizes
        # mean every started request already emitted -- skip before building
        # metadata or scanning the batch.
        if len(self._prefill_end_done) == len(self._prefill_start_done):
            return
        metadata = {
            "batch_size": len(batch.reqs),
            "is_extend_in_batch": bool(batch.is_extend_in_batch),
        }
        for req in batch.reqs:
            rid = req.rid
            if rid in self._prefill_end_done or rid not in self._prefill_start_done:
                continue
            self._prefill_end_done.add(rid)
            _emit_event(
                request_id=rid,
                stage=None,
                event_name="scheduler_prefill_end",
                metadata=metadata,
            )

    def _emit_model_path_end_once(self, request_id: str, *, status: str) -> None:
        if request_id not in self._prefill_start_done:
            return
        self._prefill_start_done.discard(request_id)
        _emit_model_path_end(request_id, status=status)

    def _emit_remaining_model_path_ends(self, *, status: str) -> None:
        for request_id in tuple(self._prefill_start_done):
            self._emit_model_path_end_once(request_id, status=status)

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
            data = None
            with self._request_admission_lock:
                is_aborted = isinstance(req.finished_reason, FINISH_ABORT) or (
                    rid in self._aborted_request_ids
                )
                if not is_aborted:
                    if req._omni_terminal_claimed:
                        continue
                    data = req._omni_data
                    if data is None:
                        logger.error(
                            f"OmniScheduler: terminal request {rid!r} has no "
                            "request data; dropping a stale terminal alias"
                        )
                        self._close_completed_request(req)
                        continue
                    # Abort may run from the stage listener thread. Claiming the
                    # terminal request under the shared lock makes normal
                    # terminalization its sole cleanup owner without hiding
                    # request data from stream ingress before cleanup finishes.
                    req._omni_terminal_claimed = True

            if is_aborted:
                # note (Gaokai): an abort landing mid-step finishes here via
                # FINISH_ABORT; run the cleanup abort() deferred (callbacks are
                # idempotent) and drop the stale terminal result so it cannot
                # resurrect the request downstream.
                self._run_abort_callback(rid)
                self._first_emit_done.discard(rid)
                self._emit_model_path_end_once(rid, status="aborted")
                _detach_request_data(req)
                continue

            result = None
            terminal_error = None
            try:
                # Drain runner stream buffers before the terminal payload; both
                # use this outbox, so remaining chunks stay ahead of stream done.
                model_runner = self._model_runner
                if model_runner is not None:
                    model_runner.on_request_finished(rid, data)
                data.output_ids = list(req.output_ids)
                data.weight_version = self.server_args.weight_version
                finished_reason = req.finished_reason
                data.finish_reason = (
                    finished_reason.to_json().get("type")
                    if finished_reason is not None
                    else None
                )
                self._flush_stream_output(rid, data)
                result = self._result_adapter(data)
            except Exception as exc:
                terminal_error = exc
                logger.exception(
                    "OmniScheduler terminal output handling failed for request %s",
                    rid,
                )
            finally:
                callback = self._request_finished_callback
                if callback is not None:
                    try:
                        callback(rid)
                    except Exception as exc:
                        logger.exception(
                            f"OmniScheduler: terminal cleanup failed for {rid}"
                        )
                        if terminal_error is None:
                            terminal_error = exc
                data.prefill_input_embeds = None
                data.decode_input_embeds = None
                # Note: (Jiaxin Deng) close the model-path interval before
                # _close_completed_request, which discards the same rid that
                # _emit_model_path_end_once dedups on. Emitting afterwards
                # silently drops every terminal event on the success path.
                self._emit_model_path_end_once(
                    rid,
                    status="error" if terminal_error is not None else "success",
                )
                abort_cleanup_needed = self._close_completed_request(req)

            if abort_cleanup_needed:
                self._run_abort_callback(rid)

            if terminal_error is not None:
                self._first_emit_done.discard(rid)
                self._emit_request_error(rid, terminal_error)
                continue

            self._first_emit_done.discard(rid)
            self.outbox.put(
                OutgoingMessage(
                    request_id=rid,
                    type="result",
                    data=result,
                )
            )

    def _on_stream_chunk(self, request_id: str, chunk: Any) -> None:
        if request_id in self._completed_request_ids:
            return
        req_data = self._find_request_data(request_id)
        if req_data is not None:
            self._append_stream_chunk(req_data, chunk)
            return
        self._reserve_pending_stream_request(request_id)
        self._pending_stream_ingress.setdefault(
            request_id, _PendingStreamIngress()
        ).chunks.append(chunk)
        if (
            request_id in self._deferred_request_payloads
            and self._should_recheck_deferred_request_on_stream_chunk(request_id, chunk)
        ):
            self._dirty_deferred_request_ids.add(request_id)

    def _on_stream_done(self, request_id: str) -> None:
        if request_id in self._completed_request_ids:
            return
        req_data = self._find_request_data(request_id)
        if req_data is not None:
            self._mark_stream_done(req_data)
            return
        self._reserve_pending_stream_request(request_id)
        self._pending_stream_ingress.setdefault(
            request_id, _PendingStreamIngress()
        ).done = True
        if request_id in self._deferred_request_payloads:
            self._dirty_deferred_request_ids.add(request_id)

    def start(self) -> None:
        self._scheduler_thread_id = threading.get_ident()
        self._running = True
        model_path_status = "error"
        try:
            if self.enable_async_decode:
                self._event_loop_async_decode()
            elif self.enable_overlap:
                self._event_loop_overlap()
            else:
                self._event_loop_normal()
            model_path_status = "aborted"
        finally:
            self._emit_remaining_model_path_ends(status=model_path_status)
            self._scheduler_thread_id = None
            try:
                self._shutdown_request_build_executor()
            finally:
                self._discard_pending_request_admissions()
                self._shutdown_resources()

    def event_loop(self) -> None:
        self.start()

    def stop(self) -> None:
        self._running = False
        self._discard_pending_request_admissions()
        self._shutdown_resources()

    def _discard_pending_request_admissions(self) -> None:
        with self._request_admission_lock:
            self._pending_request_admissions.clear()

    def _shutdown_resources(self) -> None:
        with self._shutdown_lock:
            callback = self._shutdown_callback
            self._shutdown_callback = None
        if callback is not None:
            callback()

    def _shutdown_request_build_executor(self) -> None:
        executor = self._request_build_executor
        if executor is None:
            return
        executor.shutdown(wait=False, cancel_futures=True)
        self._request_build_executor = None

    def abort(self, request_id: str, *, defer_running_cleanup: bool = True) -> None:
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
            running_abort = (
                self._mark_running_request_aborted(request_id)
                if defer_running_cleanup
                else False
            )
            pending = self._pending_request_builds.pop(request_id, None)
            if pending is not None:
                pending[2].cancel()
            self._pending_request_admissions.pop(request_id, None)
            if self._backlogged_request_build_payloads:
                retained = [
                    payload
                    for payload in self._backlogged_request_build_payloads
                    if payload.request_id != request_id
                ]
                self._backlogged_request_build_payloads.clear()
                self._backlogged_request_build_payloads.extend(retained)
            waiting_queue = []
            for req in self.waiting_queue:
                if req.rid == request_id:
                    _detach_request_data(req)
                else:
                    waiting_queue.append(req)
            self.waiting_queue = waiting_queue
        if not running_abort:
            self._run_abort_callback(request_id)
        self._pending_stream_ingress.pop(request_id, None)
        self._deferred_request_payloads.pop(request_id, None)
        self._dirty_deferred_request_ids.discard(request_id)
        self._first_emit_done.discard(request_id)
        # Note: (Jiaxin Deng) emit before discarding, and discard whether or
        # not the request is still in a running batch. A running abort that
        # never reaches stream_output used to leave its rid here forever,
        # which grew unbounded on a long-lived server and then swallowed a
        # later prefill_start for the same id.
        self._emit_model_path_end_once(request_id, status="aborted")
        self._prefill_start_done.discard(request_id)
        self._prefill_end_done.discard(request_id)
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
        scheduler_thread_id = self._scheduler_thread_id
        return (
            self._running
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
            request_admission_pending = len(self._pending_request_admissions)
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
                "request_admission_pending": request_admission_pending,
                "request_build_max_pending": self.request_build_max_pending,
                "request_build_backlog": request_build_backlog,
                "request_build_max_pending_observed": (
                    self._request_build_max_pending_observed
                ),
                "running_batch_size": len(self.running_batch.reqs),
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
                                "pause_mode": self._last_pause_mode,
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
            if self._pending_request_admissions:
                request_ids.update(self._pending_request_admissions.keys())
            if self._backlogged_request_build_payloads:
                request_ids.update(
                    payload.request_id
                    for payload in self._backlogged_request_build_payloads
                    if payload.request_id not in self._aborted_request_ids
                )
            for req in self.waiting_queue:
                rid = req.rid
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
        return bool(engine_paused and self._last_pause_mode == "retract")

    def _retract_running_requests(self) -> int:
        batch = self.running_batch
        if batch is None or batch.is_empty():
            return 0
        batch.filter_batch()
        if len(batch.reqs) == 0:
            return 0
        # sglang 0.5.16 dropped ScheduleBatch.retract_all; the module-level
        # function returns None and leaves batch.reqs in place, so snapshot the
        # requests and clear the batch here (what the old method did for us).
        retracted_reqs = list(batch.reqs)
        retract_all(
            reqs=batch.reqs,
            server_args=self.server_args,
            req_to_token_pool=batch.req_to_token_pool,
            token_to_kv_pool_allocator=batch.token_to_kv_pool_allocator,
            tree_cache=batch.tree_cache,
            hisparse_coordinator=batch.hisparse_coordinator,
        )
        batch.reqs = []
        for req in retracted_reqs:
            self._add_request_to_queue(req)
        batch.batch_is_full = False
        self.chunked_req = None
        return len(retracted_reqs)

    def _flush_cache_after_update(self) -> bool:
        try:
            return bool(self.flush_cache())
        except Exception:
            logger.exception("flush_cache after weight update failed")
            return False

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
                if req.rid != request_id:
                    continue
                if req._omni_terminal_claimed:
                    # stream_output already owns final cleanup for this request.
                    if req._omni_data is not None:
                        marked = True
                    continue
                if req.finished() or req.is_retracted:
                    continue
                req.to_finish = FINISH_ABORT()
                marked = True
        return marked

    def _run_abort_callback(self, request_id: str) -> None:
        callback = self._abort_callback
        if callback is None:
            return
        try:
            callback(request_id)
        except Exception:
            logger.exception("OmniScheduler: abort cleanup failed for %s", request_id)

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
                self._sched_idled = True
                self.self_check_during_idle()
                self._sleep_during_idle()

            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.self_check_during_busy()

    def _event_loop_overlap(self) -> None:
        # Model runners read Req.inflight_middle_chunks at forward time under
        # a same-iteration process_batch_result contract. On this loop the
        # decrement lags one iteration, so a final prefill chunk still reads
        # as a middle chunk at forward time and the TTS model runners emit
        # wrong chunk boundaries — silently. No construction site enables
        # overlap today; refuse to run rather than corrupt chunked prefill if
        # one ever does.
        # The pre-guard loop body lives in git history ("Refuse the
        # OmniScheduler overlap event loop"); reviving it needs that drain,
        # not just deleting this raise.
        raise NotImplementedError(
            "OmniScheduler's overlap event loop is unsupported: "
            "Req.inflight_middle_chunks lags one iteration on this loop. "
            "Drain the result queue before the forward, then remove this "
            "guard."
        )

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

        ``_async_pending`` is ``(batch, sched_output, pending_step)`` or None.
        """
        pending = self._async_pending
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
        pre_finished = [r.finished() or r.is_retracted for r in batch.reqs]
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

    def _drop_stale_overrun(self, batch):
        """Drop reqs finished OR retracted by the just-completed drain from the
        stale fast-path batch, so run_batch does not forward/finalize them again
        (double-free of already-freed KV). Returns the filtered batch, or None if
        it empties. Mirrors the finished/is_retracted pre-drop in
        _resolve_and_process; the fast path previously dropped only finished.

        The dropped rows' step slots need no compensating free: the batch's
        prepare already advanced req.kv_committed_len over them, and the
        drain's release_kv_cache frees or caches every committed slot, so
        a second free here would put a slot on the free list that the radix
        tree (or another request) still owns.
        """
        if batch is None or not batch.reqs:
            return batch
        drop = [r.finished() or r.is_retracted for r in batch.reqs]
        if not any(drop):
            return batch
        keep = [i for i, d in enumerate(drop) if not d]
        out_cache_loc = batch.out_cache_loc
        forward_mode = batch.forward_mode
        if forward_mode is not None and forward_mode.is_extend():
            if batch.mix_running_indices is not None:
                raise RuntimeError(
                    "Omni does not support stale-row filtering for SGLang mixed "
                    "chunked-prefill batches"
                )
            # Note:(Wenyao Gao) extend/mixed batches carry per-token fields
            # (req i owns extend_lens[i] slots) that filter_batch leaves
            # stale; reslice them here. The asserted fields are never
            # populated on omni extend batches; trip instead of misslicing.
            assert (
                batch.input_embeds is None and batch.replace_embeds is None
            ), "unhandled per-token field on drop-stale extend batch"
            lens = batch.extend_lens
            starts = [0] * len(lens)
            for i in range(1, len(lens)):
                starts[i] = starts[i - 1] + lens[i - 1]
            keep_tokens = [
                t for i in keep for t in range(starts[i], starts[i] + lens[i])
            ]
            input_ids = batch.input_ids
            prefill_input_ids_cpu = batch.prefill_input_ids_cpu
            if input_ids is None and prefill_input_ids_cpu is None:
                raise RuntimeError(
                    "extend batch carries neither input_ids nor "
                    "prefill_input_ids_cpu"
                )
            prefix_lens = batch.prefix_lens
            extend_logprob_start_lens = batch.extend_logprob_start_lens
            lp_token_ids = batch.extend_input_logprob_token_ids
            batch.filter_batch(keep_indices=keep)
            if input_ids is not None:
                batch.input_ids = input_ids[keep_tokens]
            else:
                batch.prefill_input_ids_cpu = prefill_input_ids_cpu[keep_tokens]
            if out_cache_loc is not None:
                batch.out_cache_loc = out_cache_loc[keep_tokens]
            batch.extend_lens = [lens[i] for i in keep]
            batch.extend_num_tokens = sum(batch.extend_lens)
            batch.prefix_lens = [prefix_lens[i] for i in keep]
            batch.extend_logprob_start_lens = [
                extend_logprob_start_lens[i] for i in keep
            ]
            if lp_token_ids is not None:
                # Note:(Wenyao Gao) every req contributes a segment of
                # lens[i] - start_lens[i] ids; not aligned with the token
                # slices above.
                if not batch.return_logprob:
                    batch.extend_input_logprob_token_ids = None
                else:
                    lp_lens = [
                        lens[i] - extend_logprob_start_lens[i] for i in range(len(lens))
                    ]
                    lp_starts = [0] * len(lp_lens)
                    for i in range(1, len(lp_lens)):
                        lp_starts[i] = lp_starts[i - 1] + lp_lens[i - 1]
                    keep_lp_tokens = [
                        t
                        for i in keep
                        for t in range(lp_starts[i], lp_starts[i] + lp_lens[i])
                    ]
                    batch.extend_input_logprob_token_ids = lp_token_ids[keep_lp_tokens]
        else:
            batch.filter_batch(keep_indices=keep)
            if out_cache_loc is not None:
                batch.out_cache_loc = out_cache_loc[keep]
        if batch.decoding_reqs:
            kept_ids = {id(r) for r in batch.reqs}
            batch.decoding_reqs = [r for r in batch.decoding_reqs if id(r) in kept_ids]
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
            runner = self._model_runner
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
                    self._sched_idled = True
                    self.self_check_during_idle()
                    self._sleep_during_idle()

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

    def _remember_completed_request(self, request_id: str) -> None:
        if request_id in self._completed_request_ids:
            return
        if len(self._completed_request_ids) >= _COMPLETED_REQUEST_ID_LIMIT:
            del self._completed_request_ids[next(iter(self._completed_request_ids))]
        self._completed_request_ids[request_id] = None
        self._pending_stream_ingress.pop(request_id, None)

    def _reserve_pending_stream_request(self, request_id: str) -> None:
        pending = self._pending_stream_ingress
        if request_id in pending:
            return
        if len(pending) < _PENDING_STREAM_REQUEST_LIMIT:
            return

        # Oldest-first: stale entries go before a request that is about to be
        # admitted.
        evict_count = len(pending) - _PENDING_STREAM_REQUEST_RETAINED + 1
        for stale_request_id in list(islice(pending, evict_count)):
            del pending[stale_request_id]
        logger.warning(
            "OmniScheduler evicted %d pending stream request(s) after reaching "
            "the %d-request limit",
            evict_count,
            _PENDING_STREAM_REQUEST_LIMIT,
        )

    def _close_completed_request(self, req: Any) -> bool:
        request_id = req.rid
        with self._request_admission_lock:
            _detach_request_data(req)
            self._remember_completed_request(request_id)
            abort_cleanup_needed = request_id in self._aborted_request_ids
        self._first_emit_done.discard(request_id)
        self._prefill_start_done.discard(request_id)
        self._prefill_end_done.discard(request_id)
        return abort_cleanup_needed

    def _find_request_data(self, request_id: str) -> Any | None:
        # Scan all batches a live req can sit in during prefill→decode handoff.
        for batch in (
            self.running_batch,
            self.cur_batch,
            self.last_batch,
            self._async_pending_batch(),
        ):
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
    remaining_reqs = []
    for req in batch.reqs:
        if req.rid == request_id:
            _detach_request_data(req)
        else:
            remaining_reqs.append(req)
    batch.reqs = remaining_reqs
    if not batch.reqs:
        batch.batch_is_full = False
