# SPDX-License-Identifier: Apache-2.0
"""Whisper ASR SGLang engine builder."""

from __future__ import annotations

import logging
from typing import Any

from sglang_omni.models.whisper_asr.encoder_service import (
    WhisperPreLMEncoderService,
    build_cache_namespace,
)
from sglang_omni.models.whisper_asr.request_builders import MAX_PREV_CONTEXT_TOKENS
from sglang_omni.platforms import current_platform
from sglang_omni.scheduling.engine_factory import AsrEngineBuilder
from sglang_omni.scheduling.generation_batch_policy import (
    CudaGraphBackend,
    build_default_prefill_cuda_graph_bs,
)

logger = logging.getLogger(__name__)

_DEFAULT_ENCODER_GRAPH_BATCH_BUCKETS = (1, 2, 4, 8, 12, 16)


_TASK_PREFIX_SLACK_TOKENS = 8
_DECODER_PREFILL_TOKENS_PER_REQUEST = (
    MAX_PREV_CONTEXT_TOKENS + _TASK_PREFIX_SLACK_TOKENS
)


def _max_reachable_decoder_prefill_tokens(
    *,
    budget: int,
    encoder_token_count: int,
    request_limit: int,
) -> int:
    """Return the largest decoder-token batch allowed by atomic admission."""
    max_request_tokens = encoder_token_count + _DECODER_PREFILL_TOKENS_PER_REQUEST
    lower_request_count = min(
        request_limit,
        max(1, budget // max_request_tokens),
    )
    upper_request_count = min(
        request_limit,
        max(1, (budget + max_request_tokens - 1) // max_request_tokens),
    )

    # note (xuanyili): This is the O(1) equivalent of enumerating request_count
    # and maximizing min(request_count * decoder_tokens,
    # budget - request_count * encoder_tokens). The lower envelope of those
    # increasing and decreasing lines peaks next to their intersection.
    lower_cap = min(
        lower_request_count * _DECODER_PREFILL_TOKENS_PER_REQUEST,
        max(0, budget - lower_request_count * encoder_token_count),
    )
    upper_cap = min(
        upper_request_count * _DECODER_PREFILL_TOKENS_PER_REQUEST,
        max(0, budget - upper_request_count * encoder_token_count),
    )
    # The scheduler always admits the first whole request.
    return max(_DECODER_PREFILL_TOKENS_PER_REQUEST, lower_cap, upper_cap)


def _reachable_prefill_cuda_graph_max_bs(
    overrides: dict[str, Any],
    *,
    encoder_token_count: int,
    max_running_requests: int | None,
) -> int | None:
    """Cap the prefill graph ladder at the largest decoder batch admission can form.

    Admission charges each request its encoder placeholders as well, while the
    graph sees only the admitted requests' decoder tokens.
    """
    max_prefill_tokens = overrides.get("max_prefill_tokens")
    if encoder_token_count < 1 or not max_prefill_tokens or int(max_prefill_tokens) < 1:
        return None
    budget = int(max_prefill_tokens)
    request_limit = max(1, budget // encoder_token_count)
    if max_running_requests is not None and int(max_running_requests) >= 1:
        request_limit = min(request_limit, int(max_running_requests))

    caps = [
        _max_reachable_decoder_prefill_tokens(
            budget=budget,
            encoder_token_count=encoder_token_count,
            request_limit=request_limit,
        )
    ]
    for key in ("cuda_graph_max_bs_prefill", "max_prefill_tokens", "max_total_tokens"):
        value = overrides.get(key)
        if value is not None and int(value) > 0:
            caps.append(int(value))
    cap = min(caps)
    overrides["cuda_graph_max_bs_prefill"] = cap
    return cap


def _normalize_encoder_graph_buckets(buckets: list[int] | None) -> tuple[int, ...]:
    values = _DEFAULT_ENCODER_GRAPH_BATCH_BUCKETS if buckets is None else buckets
    normalized = {int(value) for value in values}
    return tuple(sorted(value for value in normalized if value >= 1))


def _resolve_encoder_graph_buckets(
    buckets: tuple[int, ...],
    *,
    enable_pre_lm_encoder: bool,
    pre_lm_max_batch_size: int,
    max_prefill_tokens: int,
    encoder_token_count: int,
    max_running_requests: int | None,
) -> tuple[int, ...]:
    # note (guozhihao-224): pre-LM encoding is no longer limited by atomic prefill
    # admission, so capture against pre_lm_max_batch_size. Keep
    # max_prefill_tokens // encoder_token_count only for encoder-in-prefill.
    if enable_pre_lm_encoder:
        capture_limit = pre_lm_max_batch_size
    else:
        if max_prefill_tokens < 1:
            raise ValueError(
                f"max_prefill_tokens must be >= 1, got {max_prefill_tokens}"
            )
        if encoder_token_count < 1:
            raise ValueError(
                f"encoder_token_count must be >= 1, got {encoder_token_count}"
            )
        capture_limit = max_prefill_tokens // encoder_token_count
    if max_running_requests is not None:
        if max_running_requests < 1:
            raise ValueError(
                "max_running_requests must be >= 1, " f"got {max_running_requests}"
            )
        capture_limit = min(capture_limit, max_running_requests)
    resolved = {bucket for bucket in buckets if bucket <= capture_limit}
    if max_running_requests is not None and capture_limit >= 1:
        resolved.add(capture_limit)
    return tuple(sorted(resolved))


class WhisperASREngineBuilder(AsrEngineBuilder):
    model_name = "Whisper ASR"
    model_arch_override = "WhisperForConditionalGeneration"
    supports_breakable_prefill_cuda_graph = True

    def __init__(
        self,
        *,
        max_running_requests: int,
        max_new_tokens: int,
        mem_fraction_static: float,
        enable_encoder_cuda_graph: bool = False,
        encoder_graph_batch_buckets: list[int] | None = None,
        request_build_max_workers: int = 8,
        enable_async_decode: bool = True,
        async_decode_min_batch_size: int = 2,
        request_build_max_pending: int | None = 16,
        prefill_coalesce_requests: int = 2,
        prefill_coalesce_wait_ms: float = 6.0,
        prefill_coalesce_when_idle: bool = True,
        prefill_coalesce_requires_pending_builds: bool = True,
        prefill_coalesce_after_builds_during_decode: bool = False,
        enable_pre_lm_encoder: bool = True,
        pre_lm_cache_max_entries: int = 1024,
        pre_lm_cache_size_bytes: int | None = None,
        pre_lm_max_batch_size: int = 8,
        pre_lm_max_batch_wait_ms: int = 0,
        pre_lm_cache_pin_host_memory: bool = True,
    ) -> None:
        if pre_lm_max_batch_size < 1:
            raise ValueError(
                f"pre_lm_max_batch_size must be >= 1, got {pre_lm_max_batch_size}"
            )
        if pre_lm_max_batch_wait_ms < 0:
            raise ValueError(
                f"pre_lm_max_batch_wait_ms must be >= 0, got {pre_lm_max_batch_wait_ms}"
            )
        self.max_running_requests = max_running_requests
        self.max_new_tokens = max_new_tokens
        self.mem_fraction_static = mem_fraction_static
        self.enable_encoder_cuda_graph = bool(enable_encoder_cuda_graph)
        self._using_default_encoder_graph_buckets = encoder_graph_batch_buckets is None
        self.encoder_graph_batch_buckets = _normalize_encoder_graph_buckets(
            encoder_graph_batch_buckets
        )
        self.enable_async_decode = enable_async_decode
        self.async_decode_min_batch_size = async_decode_min_batch_size
        self.request_build_max_workers = request_build_max_workers
        self.request_build_max_pending = request_build_max_pending
        self.prefill_coalesce_requests = prefill_coalesce_requests
        self.prefill_coalesce_wait_ms = prefill_coalesce_wait_ms
        self.prefill_coalesce_when_idle = prefill_coalesce_when_idle
        self.prefill_coalesce_requires_pending_builds = (
            prefill_coalesce_requires_pending_builds
        )
        self.prefill_coalesce_after_builds_during_decode = (
            prefill_coalesce_after_builds_during_decode
        )
        self.enable_pre_lm_encoder = bool(enable_pre_lm_encoder)
        self.pre_lm_cache_max_entries = int(pre_lm_cache_max_entries)
        # None: derive the byte budget from the entry count (see encoder_service).
        self.pre_lm_cache_size_bytes = (
            None if pre_lm_cache_size_bytes is None else int(pre_lm_cache_size_bytes)
        )
        self.pre_lm_max_batch_size = int(pre_lm_max_batch_size)
        self.pre_lm_max_batch_wait_ms = int(pre_lm_max_batch_wait_ms)
        self.pre_lm_cache_pin_host_memory = bool(pre_lm_cache_pin_host_memory)
        self.processor: Any = None
        self.tokenizer: Any = None
        self.generation_config: Any = None
        self.encoder_token_count = 0
        self.context_length = 0
        self.decoder_context_len = 0
        self.audio_encoder_service: Any | None = None
        # Assigned by AsrEngineBuilder.build before generation_defaults runs.
        self.device: str | None = None

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        from transformers import AutoConfig, AutoProcessor, GenerationConfig

        self.processor = AutoProcessor.from_pretrained(checkpoint_dir)
        self.tokenizer = self.processor.tokenizer
        self.generation_config = GenerationConfig.from_pretrained(checkpoint_dir)
        self.encoder_token_count = int(
            self.processor.feature_extractor.nb_max_frames // 2
        )
        self.context_length = (
            self.encoder_token_count
            + MAX_PREV_CONTEXT_TOKENS
            + self.max_new_tokens
            + _TASK_PREFIX_SLACK_TOKENS
        )
        self.decoder_context_len = int(
            AutoConfig.from_pretrained(checkpoint_dir).max_target_positions or 448
        )

    def setup_model_resources(
        self,
        model: Any,
        server_args: Any,
        *,
        generation_cuda_graph_enabled: bool,
    ) -> None:
        if not self.enable_encoder_cuda_graph or not generation_cuda_graph_enabled:
            return
        from sglang.srt.runtime_context import get_schedule

        max_prefill_tokens = int(get_schedule().max_prefill_tokens)
        max_running_requests = int(get_schedule().max_running_requests)
        resolved_buckets = _resolve_encoder_graph_buckets(
            self.encoder_graph_batch_buckets,
            enable_pre_lm_encoder=self.enable_pre_lm_encoder,
            pre_lm_max_batch_size=self.pre_lm_max_batch_size,
            max_prefill_tokens=max_prefill_tokens,
            encoder_token_count=self.encoder_token_count,
            max_running_requests=(
                max_running_requests
                if self._using_default_encoder_graph_buckets
                else None
            ),
        )
        logger.info(
            "Resolved Whisper encoder CUDA graph buckets configured=%s "
            "reachable=%s pre_lm=%s max_prefill_tokens=%d "
            "encoder_token_count=%d max_running_requests=%d",
            self.encoder_graph_batch_buckets,
            resolved_buckets,
            self.enable_pre_lm_encoder,
            max_prefill_tokens,
            self.encoder_token_count,
            max_running_requests,
        )
        model.init_encoder_graphs(
            resolved_buckets,
            int(self.processor.feature_extractor.nb_max_frames),
        )

    def setup_runtime_resources(self, model: Any, server_args: Any) -> None:
        del server_args
        if self._uses_mlx():
            # The pre-LM service caches Torch encoder states and drives encoder
            # CUDA graphs. On MLX the runner owns encoding, and its output goes
            # straight into the per-request cross-attention cache.
            self.audio_encoder_service = None
            return
        if not self.enable_pre_lm_encoder:
            return

        self.audio_encoder_service = WhisperPreLMEncoderService(
            model,
            cache_namespace=build_cache_namespace(
                model,
                model_path=self.checkpoint_dir,
                feature_extractor=self.processor.feature_extractor,
            ),
            encoder_token_count=self.encoder_token_count,
            cache_max_entries=self.pre_lm_cache_max_entries,
            cache_max_bytes=self.pre_lm_cache_size_bytes,
            max_batch_size=self.pre_lm_max_batch_size,
            max_batch_wait_ms=self.pre_lm_max_batch_wait_ms,
            pin_host_memory=self.pre_lm_cache_pin_host_memory,
        )
        service = self.audio_encoder_service
        logger.info(
            "Whisper pre-LM encoder enabled (max_batch=%d, cache capacity=%d "
            "entries x %.2f MB = %.2f GB, configured entries=%d, byte cap=%s, "
            "pinned host memory=%s)",
            self.pre_lm_max_batch_size,
            service.cache_capacity_entries,
            service.entry_bytes / 1e6,
            service.cache_max_bytes / 1e9,
            self.pre_lm_cache_max_entries,
            self.pre_lm_cache_size_bytes,
            service.pin_host_memory,
        )

    @staticmethod
    def _uses_mlx() -> bool:
        from sglang.srt.hardware_backend.mlx.runtime import use_mlx

        return bool(use_mlx())

    def _uses_torch_mps(self) -> bool:
        """True when this stage runs Torch on Metal, without the MLX runner.

        Keyed off the resolved device rather than ``current_platform``, which is
        a process-wide singleton: on macOS arm64 it reports MPS even for a stage
        explicitly placed on CPU, which would then inherit the Metal-only
        profile. Matches the Qwen3-ASR builder.
        """
        import torch

        return (
            not self._uses_mlx()
            and self.device is not None
            and torch.device(self.device).type == "mps"
        )

    def validate_before_infrastructure(self, server_args: Any) -> None:
        """Reject Apple settings the runners cannot honor, before startup.

        Concurrency is clamped in adjust_overrides rather than rejected here,
        since the stage's own EngineArgs carry the CUDA value and a hard failure
        would make the default launch unusable.
        """
        # The Apple check runs first so the error names the setting to change,
        # where the shared batch policy would report a generic mismatch.
        if self._uses_mlx() and getattr(server_args, "mlx_enable_sampling", False):
            raise ValueError("Whisper MLX currently requires mlx_enable_sampling=False")
        super().validate_before_infrastructure(server_args)

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        if int(overrides.get("chunked_prefill_size") or 0) > 0:
            raise ValueError(
                "Whisper ASR requires chunked_prefill_size=0 because its encoder "
                "prefix must be admitted atomically"
            )
        overrides["chunked_prefill_size"] = 0
        # Note (Akazaakane): Timestamped Whisper requests install an internal
        # per-request processor; this flag permits SGLang to execute it.
        # The MLX path decodes greedily and rejects logit editing in
        # prefill_start, so leave the flag off rather than advertising support.
        overrides["enable_custom_logit_processor"] = not self._uses_mlx()
        if self._uses_mlx() or self._uses_torch_mps():
            # Both Apple paths decode one request at a time. This has to
            # happen here rather than in generation_defaults, because the
            # stage's own EngineArgs take precedence over those defaults and
            # would restore the CUDA value.
            requested = overrides.get("max_running_requests")
            if requested is not None and int(requested) != 1:
                logger.warning(
                    "Whisper %s decodes one request at a time; overriding "
                    "max_running_requests=%s with 1",
                    "MLX" if self._uses_mlx() else "Torch MPS",
                    requested,
                )
            overrides["max_running_requests"] = 1
            # Metal has no CUDA graphs, so the prefill ladder below would
            # be dead configuration. build_generation_batch_overrides only
            # forces cuda_graph_backend_prefill=DISABLED when the deployment
            # passes disable_cuda_graph itself, so the Apple default set in
            # generation_defaults does not reach that check.
            return

        if (
            overrides.get("cuda_graph_backend_prefill") == CudaGraphBackend.DISABLED
            or "cuda_graph_bs_prefill" in overrides
        ):
            return

        cap = _reachable_prefill_cuda_graph_max_bs(
            overrides,
            encoder_token_count=self.encoder_token_count,
            max_running_requests=overrides.get("max_running_requests"),
        )
        if cap is None:
            return
        overrides["cuda_graph_bs_prefill"] = build_default_prefill_cuda_graph_bs(cap)

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        if self._uses_mlx():
            if not current_platform.is_mps():
                raise RuntimeError("SGLANG_USE_MLX=1 requires the Apple Metal platform")
            # The encoder output lives in the MLX prefill's cross-attention
            # cache rather than in the KV pool, so token-only radix reuse and
            # split prefill would drop it.
            return {
                # Clamped again in adjust_overrides, which runs after the
                # stage's own EngineArgs and is what actually decides.
                "max_running_requests": 1,
                "disable_cuda_graph": True,
                "disable_overlap_schedule": True,
                "disable_radix_cache": True,
                "enable_torch_compile": False,
                "mem_fraction_static": self.mem_fraction_static,
                "max_prefill_tokens": self.context_length,
                "chunked_prefill_size": 0,
                # Without this the backend defaults to flashinfer, and the
                # scheduler's KV-index writer then takes its Triton path:
                # write_req_to_token_pool_triton[grid](...) raises
                # "'function' object is not subscriptable" against the Triton
                # stub on Apple. torch_native selects the Python fallback.
                "attention_backend": "torch_native",
                "mm_attention_backend": "sdpa",
                "dtype": dtype,
            }
        if self._uses_torch_mps():
            # Metal has no CUDA graph or Triton lifecycle, and the flashinfer
            # default is absent too: it fails at import with "name
            # 'BatchPrefillWithRaggedKVCacheWrapper' is not defined".
            #
            # max_total_tokens matters as much as the backend. Unified memory
            # reports far more free memory than the machine can back, and
            # PyTorch MPS lets allocations run past physical RAM, so the pool
            # sizer walks up until Metal refuses. Bounding the KV budget to this
            # model's own context keeps it honest.
            return {
                "max_running_requests": 1,
                "disable_cuda_graph": True,
                "disable_overlap_schedule": True,
                "disable_radix_cache": True,
                "enable_torch_compile": False,
                "mem_fraction_static": self.mem_fraction_static,
                "max_total_tokens": self.context_length,
                "max_prefill_tokens": self.context_length,
                "chunked_prefill_size": 0,
                "attention_backend": "torch_native",
                "mm_attention_backend": "sdpa",
                "dtype": dtype,
            }
        return {
            "max_running_requests": self.max_running_requests,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "enable_torch_compile": True,
            "mem_fraction_static": self.mem_fraction_static,
            "max_prefill_tokens": 6144,
            "chunked_prefill_size": 0,
            "sampling_backend": "pytorch",
            "dtype": dtype,
            "cuda_graph_backend_prefill": CudaGraphBackend.BREAKABLE,
        }

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        if self._uses_mlx():
            from sglang_omni.model_runner.mlx_model_worker import (
                MlxSchedulerModelRunner,
            )

            return MlxSchedulerModelRunner(model_worker, output_proc)
        return super().make_model_runner(model_worker, output_proc)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        del model
        from sglang_omni.models.whisper_asr.request_builders import (
            make_whisper_scheduler_adapters,
        )

        return make_whisper_scheduler_adapters(
            processor=self.processor,
            tokenizer=self.tokenizer,
            generation_config=self.generation_config,
            encoder_token_count=self.encoder_token_count,
            max_new_tokens=self.max_new_tokens,
            decoder_context_len=self.decoder_context_len,
            audio_encoder_service=self.audio_encoder_service,
        )

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {
            "enable_async_decode": self.enable_async_decode,
            "async_decode_min_batch_size": self.async_decode_min_batch_size,
            "request_build_max_workers": self.request_build_max_workers,
            "request_build_max_pending": self.request_build_max_pending,
            "prefill_coalesce_requests": self.prefill_coalesce_requests,
            "prefill_coalesce_wait_ms": self.prefill_coalesce_wait_ms,
            "prefill_coalesce_when_idle": self.prefill_coalesce_when_idle,
            "prefill_coalesce_requires_pending_builds": (
                self.prefill_coalesce_requires_pending_builds
            ),
            "prefill_coalesce_after_builds_during_decode": (
                self.prefill_coalesce_after_builds_during_decode
            ),
        }

    def extra_scheduler_callbacks(self) -> dict[str, Any]:
        if self.audio_encoder_service is None:
            return {}
        return {"shutdown_callback": self.audio_encoder_service.close}

    def cleanup_build_failure(self) -> None:
        if self.audio_encoder_service is not None:
            self.audio_encoder_service.close()
            self.audio_encoder_service = None
