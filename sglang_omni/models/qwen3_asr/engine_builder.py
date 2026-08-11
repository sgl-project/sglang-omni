# SPDX-License-Identifier: Apache-2.0
"""Qwen3-ASR SGLang engine builder."""

from __future__ import annotations

import logging
from typing import Any

from sglang.srt.managers.mm_utils import init_mm_embedding_cache
from transformers import AutoFeatureExtractor, AutoTokenizer

from sglang_omni.models.qwen3_asr import request_builders
from sglang_omni.models.qwen3_asr.audio_lengths import (
    qwen3_asr_max_audio_tokens,
    qwen3_asr_max_output_tokens,
)
from sglang_omni.models.qwen3_asr.encoder_service import (
    Qwen3ASRPreLMEncoderService,
    build_cache_namespace,
)
from sglang_omni.scheduling.engine_factory import AsrEngineBuilder
from sglang_omni.scheduling.generation_batch_policy import get_decode_cuda_graph_bs
from sglang_omni.utils.gpu_compat import get_visible_gpu_sm_version
from sglang_omni.utils.gpu_memory import format_bytes_gib, get_process_gpu_memory_bytes

logger = logging.getLogger(__name__)


class Qwen3ASREngineBuilder(AsrEngineBuilder):
    model_name = "Qwen3-ASR"
    model_arch_override = "Qwen3ASRForConditionalGeneration"

    def __init__(
        self,
        *,
        max_running_requests: int,
        max_new_tokens: int,
        enable_async_decode: bool,
        async_decode_min_batch_size: int,
        mem_fraction_static: float | None,
        mm_embedding_cache_size_bytes: int,
        enable_torch_compile: bool,
        torch_compile_max_bs: int,
        mm_attention_backend: str | None,
        request_build_max_workers: int,
        request_build_max_pending: int | None,
        prefill_coalesce_requests: int,
        prefill_coalesce_wait_ms: float,
        prefill_coalesce_when_idle: bool,
        prefill_coalesce_requires_pending_builds: bool,
        prefill_coalesce_after_builds_during_decode: bool,
        enable_pre_lm_encoder: bool = True,
        pre_lm_cache_max_entries: int = 4096,
        pre_lm_cache_size_bytes: int = 2 * 1024**3,
        pre_lm_max_batch_size: int = 8,
        pre_lm_max_batch_wait_ms: int = 0,
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
        self.enable_async_decode = enable_async_decode
        self.async_decode_min_batch_size = async_decode_min_batch_size
        self.mem_fraction_static = mem_fraction_static
        self.mm_embedding_cache_size_bytes = mm_embedding_cache_size_bytes
        self.enable_torch_compile = enable_torch_compile
        self.torch_compile_max_bs = torch_compile_max_bs
        self.mm_attention_backend = mm_attention_backend
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
        self.enable_pre_lm_encoder = enable_pre_lm_encoder
        self.pre_lm_cache_max_entries = pre_lm_cache_max_entries
        self.pre_lm_cache_size_bytes = pre_lm_cache_size_bytes
        self.pre_lm_max_batch_size = pre_lm_max_batch_size
        self.pre_lm_max_batch_wait_ms = pre_lm_max_batch_wait_ms
        self.tokenizer: Any = None
        self.feature_extractor: Any = None
        self.context_length = 0
        self.model_path: str | None = None
        self.audio_encoder_service: Any = None

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        self.model_path = checkpoint_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_dir, trust_remote_code=True
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            checkpoint_dir, trust_remote_code=True
        )
        # Note(Jeffro): Size context_length for the model's native max input + output budget.
        # model natively accepts: 1,200s (MAX_ASR_INPUT_SECONDS, see
        # https://github.com/QwenLM/Qwen3-ASR/blob/956766769/qwen_asr/inference/utils.py#L34)
        # = 15,600 audio tokens, plus 64 slack for the ~15-token chat prompt,
        # +  the max output budget for that input (12,000).
        max_prompt_tokens = qwen3_asr_max_audio_tokens() + 64
        max_output_budget = max(self.max_new_tokens, qwen3_asr_max_output_tokens())
        self.context_length = max_prompt_tokens + max_output_budget + 8

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "max_running_requests": self.max_running_requests,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "enable_torch_compile": self.enable_torch_compile,
            "torch_compile_max_bs": self.torch_compile_max_bs,
            "mem_fraction_static": self.mem_fraction_static,
            "max_prefill_tokens": 4096,
            "chunked_prefill_size": 4096,
            "sampling_backend": "pytorch",
            "dtype": dtype,
        }
        if self.mm_attention_backend is not None:
            defaults["mm_attention_backend"] = self.mm_attention_backend
        else:
            sm_version = get_visible_gpu_sm_version(self.gpu_id)
            if sm_version is not None and sm_version >= 100:
                defaults["mm_attention_backend"] = "triton_attn"
        return defaults

    def _log_memory_checkpoint(self, checkpoint: str) -> None:
        logger.info(
            "Qwen3-ASR memory checkpoint=%s gpu=%d process_gpu_memory=%s",
            checkpoint,
            self.gpu_id,
            format_bytes_gib(get_process_gpu_memory_bytes(self.gpu_id)),
        )

    def validate_before_infrastructure(self, server_args: Any) -> None:
        super().validate_before_infrastructure(server_args)
        logger.info(
            "Qwen3-ASR runtime profile: dtype=%s attention_backend=%s "
            "mm_attention_backend=%s cuda_graph=%s cuda_graph_bs=%s "
            "torch_compile=%s max_running_requests=%s mem_fraction_static=%s",
            getattr(server_args, "dtype", None),
            getattr(server_args, "attention_backend", None),
            getattr(server_args, "mm_attention_backend", None),
            not getattr(server_args, "disable_cuda_graph", False),
            get_decode_cuda_graph_bs(server_args),
            getattr(server_args, "enable_torch_compile", False),
            getattr(server_args, "max_running_requests", None),
            getattr(server_args, "mem_fraction_static", None),
        )
        self._log_memory_checkpoint("pre_model_load")

    def validate_after_model_setup(self, model: Any, server_args: Any) -> None:
        del model, server_args
        self._log_memory_checkpoint("post_static_allocation")

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        if "context_length" in overrides:
            self.context_length = int(overrides.pop("context_length"))

    def customize_server_args(self, server_args: Any) -> None:
        self.context_length = int(server_args.context_length)

    def setup_model_resources(
        self,
        model: Any,
        server_args: Any,
        *,
        generation_cuda_graph_enabled: bool,
    ) -> None:
        del generation_cuda_graph_enabled
        self._log_memory_checkpoint("post_cuda_graph_capture")
        init_mm_embedding_cache(self.mm_embedding_cache_size_bytes)
        if self.enable_pre_lm_encoder:
            # note (luojiaxuan): constructed after SGLang's generation CUDA
            # graphs so the encoder's dedicated stream never interleaves with
            # graph capture.
            self.audio_encoder_service = Qwen3ASRPreLMEncoderService(
                model,
                cache_namespace=build_cache_namespace(
                    model,
                    model_path=self.model_path or "",
                    feature_extractor=self.feature_extractor,
                    mm_attention_backend=getattr(
                        server_args, "mm_attention_backend", None
                    ),
                ),
                cache_max_entries=self.pre_lm_cache_max_entries,
                cache_max_bytes=self.pre_lm_cache_size_bytes,
                max_batch_size=self.pre_lm_max_batch_size,
                max_batch_wait_ms=self.pre_lm_max_batch_wait_ms,
            )

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        del model
        return request_builders.make_qwen3_asr_scheduler_adapters(
            tokenizer=self.tokenizer,
            feature_extractor=self.feature_extractor,
            max_new_tokens=self.max_new_tokens,
            context_length=self.context_length,
            audio_encoder_service=self.audio_encoder_service,
        )

    def extra_scheduler_callbacks(self) -> dict[str, Any]:
        if self.audio_encoder_service is None:
            return {}
        return {"shutdown_callback": self.audio_encoder_service.close}

    def cleanup_build_failure(self) -> None:
        if self.audio_encoder_service is not None:
            self.audio_encoder_service.close()
            self.audio_encoder_service = None

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
