# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS-Realtime SGLang engine builder."""

from __future__ import annotations

import importlib
import logging
from typing import Any

from sglang_omni.models.moss_tts_realtime import request_builders
from sglang_omni.models.moss_tts_realtime.config import (
    DEFAULT_MOSS_TTS_REALTIME_CODEC_MODEL,
    MossTTSRealtimeResourceLimits,
)
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder

logger = logging.getLogger(__name__)

_DEFAULT_AR_MEM_FRACTION_STATIC = 0.85


class MossTTSRealtimeEngineBuilder(TtsEngineBuilder):
    """Build the session-aware eager AR stage for MOSS-TTS-Realtime."""

    model_name = "MOSS-TTS-Realtime"
    model_arch_override = "MossTTSRealtimeSGLangModel"

    def __init__(
        self,
        *,
        max_seq_len: int | None = None,
        total_gpu_memory_fraction: float | None = None,
        codec_model_path: str = DEFAULT_MOSS_TTS_REALTIME_CODEC_MODEL,
        max_sessions: int,
        max_held_sessions: int,
        max_active_turns: int,
        max_pending_text_tokens: int,
        max_pending_text_bytes: int,
        max_input_updates: int,
        terminal_tombstone_limit: int,
        input_idle_timeout_s: float,
        turn_timeout_s: float,
        session_idle_ttl_s: float,
    ) -> None:
        self.limits = MossTTSRealtimeResourceLimits(
            max_sessions=max_sessions,
            max_held_sessions=max_held_sessions,
            max_active_turns=max_active_turns,
            max_pending_text_tokens=max_pending_text_tokens,
            max_pending_text_bytes=max_pending_text_bytes,
            max_input_updates=max_input_updates,
            terminal_tombstone_limit=terminal_tombstone_limit,
            input_idle_timeout_s=input_idle_timeout_s,
            turn_timeout_s=turn_timeout_s,
            session_idle_ttl_s=session_idle_ttl_s,
        )
        # Idle streaming sessions retain one SGLang request-pool slot. Reserve
        # room for those slots plus first turns that do not have a warm slot to
        # reuse, while respecting the total logical-session bound.
        self.request_pool_capacity = min(
            self.limits.max_sessions,
            self.limits.max_held_sessions + self.limits.max_active_turns,
        )
        self.requested_context_length = (
            None if max_seq_len is None else int(max_seq_len)
        )
        self.context_length = self.requested_context_length or 1
        if total_gpu_memory_fraction is not None and not (
            0.0 < float(total_gpu_memory_fraction) <= 1.0
        ):
            raise ValueError("total_gpu_memory_fraction must be in (0, 1]")
        if not codec_model_path.strip():
            raise ValueError("codec_model_path must not be empty")

        self.total_gpu_memory_fraction = (
            None
            if total_gpu_memory_fraction is None
            else float(total_gpu_memory_fraction)
        )
        self.codec_model_path = codec_model_path
        self.gpu_memory_bytes: int | None = None
        self.codec_decoder_bytes: int | None = None
        self.codec_streaming_state_bytes: int | None = None
        self.codec_runtime_margin_bytes: int | None = None
        self.minimum_codec_mem_reserve: float | None = None
        self.profile_total_gpu_memory_fraction: float | None = None
        self.processor: Any | None = None

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        from sglang_omni.models.moss_tts_realtime.stages import (
            load_moss_tts_realtime_processor,
        )

        # Validate the checkpoint context and processor metadata before
        # ServerArgs constructs SGLang's ModelConfig.
        self.processor = load_moss_tts_realtime_processor(checkpoint_dir)
        config = self.processor.model_config
        max_context_rows = int(config.language_config.max_position_embeddings)
        self.context_length = self.requested_context_length or max_context_rows
        if not 1 <= self.context_length <= max_context_rows:
            raise ValueError(
                f"MOSS-TTS-Realtime max_seq_len must be in [1, {max_context_rows}]"
            )
        if self.limits.max_pending_text_tokens < int(config.delay_tokens_len):
            raise ValueError(
                "max_pending_text_tokens must admit the checkpoint prefill delay"
            )
        if self.total_gpu_memory_fraction is not None:
            self._derive_colocated_codec_memory_budget()

    def _derive_colocated_codec_memory_budget(self) -> None:
        from sglang_omni.models.moss_tts_realtime.stages import (
            estimate_moss_tts_realtime_codec_memory,
        )
        from sglang_omni.utils.gpu_memory import get_gpu_device_info

        device_info = get_gpu_device_info(self.gpu_id)
        total_memory_bytes = device_info.total_memory_bytes
        if total_memory_bytes is None or total_memory_bytes <= 0:
            raise RuntimeError(
                "MOSS-TTS-Realtime colocated startup requires the total HBM "
                f"capacity for gpu_id={self.gpu_id}"
            )

        decoder_bytes, streaming_state_bytes = estimate_moss_tts_realtime_codec_memory(
            self.codec_model_path,
            # Streaming codec slots are session-scoped (held across a session's
            # turns), so the pool tops out at held sessions plus active turns.
            stream_slots=(self.limits.max_held_sessions + self.limits.max_active_turns),
        )
        runtime_margin_bytes = max(
            2 * 1024**3,
            (total_memory_bytes + 49) // 50,
        )
        required_bytes = decoder_bytes + streaming_state_bytes + runtime_margin_bytes
        reserve_millis = (
            required_bytes * 1000 + total_memory_bytes - 1
        ) // total_memory_bytes

        self.gpu_memory_bytes = total_memory_bytes
        self.codec_decoder_bytes = decoder_bytes
        self.codec_streaming_state_bytes = streaming_state_bytes
        self.codec_runtime_margin_bytes = runtime_margin_bytes
        self.minimum_codec_mem_reserve = reserve_millis / 1000

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "max_running_requests": self.request_pool_capacity,
            "dtype": dtype,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "enable_torch_compile": False,
            "max_prefill_tokens": self.context_length,
            "sampling_backend": "pytorch",
            "trust_remote_code": True,
        }
        if self.total_gpu_memory_fraction is None:
            defaults["mem_fraction_static"] = _DEFAULT_AR_MEM_FRACTION_STATIC
        return defaults

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        # Model requirements are authoritative over custom stage arguments.
        overrides["max_running_requests"] = self.request_pool_capacity
        overrides["enable_streaming_session"] = True
        overrides["disable_overlap_schedule"] = True
        overrides["enable_torch_compile"] = False

        total = self.total_gpu_memory_fraction
        if total is None:
            self.profile_total_gpu_memory_fraction = None
            return
        reserve = self.minimum_codec_mem_reserve
        if reserve is None:
            raise RuntimeError(
                "MOSS-TTS-Realtime codec memory budget was not derived before "
                "SGLang argument resolution"
            )
        max_ar_fraction = total - reserve
        if max_ar_fraction < 0.1:
            raise ValueError(
                "MOSS-TTS-Realtime colocated memory budget leaves less than "
                "the safe AR floor 0.1 after reserving decoder and streaming "
                f"state memory: total={total:.3f}, reserve={reserve:.3f}"
            )

        explicit = overrides.get("mem_fraction_static")
        if explicit is None:
            effective = max_ar_fraction
            overrides["mem_fraction_static"] = effective
        else:
            effective = float(explicit)
            if effective - max_ar_fraction > 1e-9:
                raise ValueError(
                    "MOSS-TTS-Realtime mem_fraction_static leaves less than "
                    "the required decoder and streaming-state reserve: "
                    f"{effective:.3f} > {max_ar_fraction:.3f}"
                )
        if not 0.0 < effective < 1.0:
            raise ValueError(
                "MOSS-TTS-Realtime effective mem_fraction_static must be in (0, 1)"
            )
        if effective < 0.1:
            raise ValueError(
                "MOSS-TTS-Realtime effective mem_fraction_static is below the "
                "safe floor 0.1"
            )

        self.profile_total_gpu_memory_fraction = effective
        from sglang_omni.utils.gpu_memory import get_process_gpu_memory_bytes

        if get_process_gpu_memory_bytes(self.gpu_id) is None:
            logger.warning(
                "MOSS-TTS-Realtime colocated process memory accounting is "
                "unavailable; falling back to upstream SGLang profiling"
            )
            self.profile_total_gpu_memory_fraction = None

    def customize_server_args(self, server_args: Any) -> None:
        from sglang_omni.utils.gpu_memory import format_bytes_gib
        from sglang_omni.vendor.sglang.server_args import override_server_args

        override_server_args(
            server_args,
            "sglang_omni.moss_tts_realtime.disable_overlap_schedule",
            disable_overlap_schedule=True,
        )
        logger.info(
            "MOSS-TTS-Realtime SGLang startup: gpu_id=%s "
            "total_gpu_memory_fraction=%s minimum_codec_mem_reserve=%s "
            "mem_fraction_static=%s profile_total_gpu_memory_fraction=%s",
            self.gpu_id,
            self.total_gpu_memory_fraction,
            self.minimum_codec_mem_reserve,
            server_args.mem_fraction_static,
            self.profile_total_gpu_memory_fraction,
        )
        if self.minimum_codec_mem_reserve is not None:
            logger.info(
                "MOSS-TTS-Realtime codec memory budget: hbm=%s decoder=%s "
                "streaming_state=%s max_active_turns=%d runtime_margin=%s",
                format_bytes_gib(self.gpu_memory_bytes),
                format_bytes_gib(self.codec_decoder_bytes),
                format_bytes_gib(self.codec_streaming_state_bytes),
                self.limits.max_active_turns,
                format_bytes_gib(self.codec_runtime_margin_bytes),
            )

    def infra_kwargs(self) -> dict[str, Any]:
        return {
            "total_gpu_memory_fraction": self.profile_total_gpu_memory_fraction,
        }

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        del checkpoint_dir, device, gpu_id, server_args
        from sglang_omni.models.moss_tts_realtime.stages import (
            bind_moss_tts_realtime_processor_config,
        )

        model = model_worker.model_runner.model
        if self.processor is None:
            raise RuntimeError("MOSS-TTS-Realtime processor was not initialized")
        bind_moss_tts_realtime_processor_config(model.config, self.processor)
        if (
            int(model.config.language_config.max_position_embeddings)
            < self.context_length
        ):
            raise ValueError("loaded model context is smaller than max_seq_len")
        # The scheduler already admits each turn against this same context
        # length. Using it for the fixed sampling-history pool avoids a second,
        # arbitrary per-turn generation cap while keeping the pool bounded.
        model_worker.moss_tts_realtime_max_history_frames = self.context_length
        model_worker.moss_tts_realtime_max_active_turns = self.limits.max_active_turns

    def get_model_buffer_bs(self, model: Any) -> int | None:
        return int(model._decode_input_embedding.num_embeddings)

    def post_cuda_graph_setup(self, model: Any, server_args: Any) -> None:
        from sglang_omni.scheduling.generation_batch_policy import (
            get_decode_cuda_graph_bs,
        )

        # Match MOSS-TTS Local: the generic SGLang CUDA-graph policy owns the
        # enable/disable switch and capture buckets. Realtime's physical request
        # pool also contains idle streaming sessions, so its frame decoder only
        # needs buckets up to the active-turn limit.
        batch_sizes = [
            int(batch_size)
            for batch_size in get_decode_cuda_graph_bs(server_args)
            if int(batch_size) <= self.limits.max_active_turns
        ]
        if self.limits.max_active_turns not in batch_sizes:
            batch_sizes.append(self.limits.max_active_turns)
        model.init_frame_decode_graphs(sorted(set(batch_sizes)))

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        model_runner_mod = importlib.import_module(
            "sglang_omni.models.moss_tts_realtime.model_runner"
        )
        return model_runner_mod.MossTTSRealtimeModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        return request_builders.make_moss_tts_realtime_scheduler_adapters(model=model)

    def make_abort_callback(self) -> Any | None:
        return request_builders.cleanup_prepared_moss_tts_realtime_request

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {
            **self.limits.model_dump(),
            "enable_async_decode": False,
        }

    def make_scheduler(
        self,
        *,
        model_worker: Any,
        tree_cache: Any,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        server_args: Any,
        model_config: Any,
        model_runner: Any,
        request_builder: Any,
        result_adapter: Any,
    ) -> Any:
        from sglang_omni.models.moss_tts_realtime.scheduler import (
            MossTTSRealtimeScheduler,
        )

        return MossTTSRealtimeScheduler(
            tp_worker=model_worker,
            tree_cache=tree_cache,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            server_args=server_args,
            model_config=model_config,
            model_runner=model_runner,
            request_builder=request_builder,
            result_adapter=result_adapter,
            abort_callback=self.make_abort_callback(),
            request_finished_callback=self.make_request_finished_callback(),
            **self.extra_scheduler_kwargs(),
        )

    def post_scheduler_setup(self, scheduler: Any, model_runner: Any) -> None:
        model_runner.set_stream_outbox(scheduler.outbox)
