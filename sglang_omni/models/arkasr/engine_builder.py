# SPDX-License-Identifier: Apache-2.0
"""ARK-ASR SGLang engine builder."""

from __future__ import annotations

import logging
from typing import Any

from sglang.srt.managers.mm_utils import init_mm_embedding_cache
from transformers import AutoConfig, AutoTokenizer, WhisperFeatureExtractor

from sglang_omni.models.arkasr import request_builders
from sglang_omni.models.arkasr.encoder_service import (
    ArkasrPreLMEncoderService,
    build_cache_namespace,
)
from sglang_omni.scheduling.engine_factory import AsrEngineBuilder
from sglang_omni.scheduling.generation_batch_policy import CudaGraphBackend
from sglang_omni.utils.gpu_compat import get_visible_gpu_sm_version

logger = logging.getLogger(__name__)


class ArkasrEngineBuilder(AsrEngineBuilder):
    model_name = "ARK-ASR"
    model_arch_override = "ArkasrForConditionalGeneration"
    supports_breakable_prefill_cuda_graph = True

    def __init__(
        self,
        *,
        max_running_requests: int,
        encoder_max_batch_size: int,
        max_new_tokens: int,
        enable_async_decode: bool,
        async_decode_min_batch_size: int,
        mem_fraction_static: float | None,
        mm_embedding_cache_size_bytes: int,
        enable_torch_compile: bool,
        mm_attention_backend: str | None,
        request_build_max_workers: int,
        request_build_max_pending: int | None,
        prefill_coalesce_requests: int,
        prefill_coalesce_wait_ms: float,
        prefill_coalesce_when_idle: bool,
        prefill_coalesce_requires_pending_builds: bool,
        enable_pre_lm_encoder: bool = True,
        pre_lm_cache_max_entries: int = 4096,
        pre_lm_cache_size_bytes: int = 2 * 1024**3,
        pre_lm_max_batch_size: int = 8,
        pre_lm_max_batch_wait_ms: int = 0,
        pre_lm_max_pending: int = 32,
        enable_encoder_cuda_graph: bool = False,
        stream_emit_interval_s: float = 0.05,
    ) -> None:
        if pre_lm_max_batch_size < 1:
            raise ValueError(
                f"pre_lm_max_batch_size must be >= 1, got {pre_lm_max_batch_size}"
            )
        if pre_lm_max_batch_wait_ms < 0:
            raise ValueError(
                f"pre_lm_max_batch_wait_ms must be >= 0, got {pre_lm_max_batch_wait_ms}"
            )
        if pre_lm_max_pending < 1:
            raise ValueError(
                f"pre_lm_max_pending must be >= 1, got {pre_lm_max_pending}"
            )
        self.max_running_requests = max_running_requests
        self.encoder_max_batch_size = encoder_max_batch_size
        self.max_new_tokens = max_new_tokens
        self.enable_async_decode = enable_async_decode
        self.async_decode_min_batch_size = async_decode_min_batch_size
        self.mem_fraction_static = mem_fraction_static
        self.mm_embedding_cache_size_bytes = mm_embedding_cache_size_bytes
        self.enable_torch_compile = enable_torch_compile
        self.mm_attention_backend = mm_attention_backend
        self.request_build_max_workers = request_build_max_workers
        self.request_build_max_pending = request_build_max_pending
        self.prefill_coalesce_requests = prefill_coalesce_requests
        self.prefill_coalesce_wait_ms = prefill_coalesce_wait_ms
        self.prefill_coalesce_when_idle = prefill_coalesce_when_idle
        self.prefill_coalesce_requires_pending_builds = (
            prefill_coalesce_requires_pending_builds
        )
        self.enable_pre_lm_encoder = enable_pre_lm_encoder
        self.pre_lm_cache_max_entries = pre_lm_cache_max_entries
        self.pre_lm_cache_size_bytes = pre_lm_cache_size_bytes
        self.pre_lm_max_batch_size = pre_lm_max_batch_size
        self.pre_lm_max_batch_wait_ms = pre_lm_max_batch_wait_ms
        self.pre_lm_max_pending = pre_lm_max_pending
        self.enable_encoder_cuda_graph = enable_encoder_cuda_graph
        self.stream_emit_interval_s = stream_emit_interval_s
        self.tokenizer: Any = None
        self.feature_extractor: Any = None
        self.merge_factor = 4
        self.audio_token_id = 151663
        self.context_length = 0
        self.model_path: str | None = None
        self.audio_encoder_service: Any = None

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        self.model_path = checkpoint_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_dir, trust_remote_code=True
        )
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(checkpoint_dir)
        hf_config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
        self.merge_factor = int(getattr(hf_config, "merge_factor", 4))
        self.audio_token_id = int(getattr(hf_config, "audio_token_id", 151663))
        encoder_token_count = self.feature_extractor.nb_max_frames // 2
        self.context_length = encoder_token_count + self.max_new_tokens + 8

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "max_running_requests": self.max_running_requests,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "enable_torch_compile": self.enable_torch_compile,
            "mem_fraction_static": self.mem_fraction_static,
            "max_prefill_tokens": 4096,
            "chunked_prefill_size": 4096,
            "sampling_backend": "pytorch",
            "dtype": dtype,
            "cuda_graph_backend_prefill": CudaGraphBackend.BREAKABLE,
        }
        if self.mm_attention_backend is not None:
            defaults["mm_attention_backend"] = self.mm_attention_backend
        else:
            sm_version = get_visible_gpu_sm_version(self.gpu_id)
            if sm_version is not None and sm_version >= 100:
                defaults["mm_attention_backend"] = "triton_attn"
        return defaults

    def setup_model_resources(
        self,
        model: Any,
        server_args: Any,
        *,
        generation_cuda_graph_enabled: bool,
    ) -> None:
        del generation_cuda_graph_enabled
        model.set_encoder_max_batch_size(self.encoder_max_batch_size)
        if self.enable_encoder_cuda_graph:
            from sglang_omni.models.arkasr.encoder_cuda_graph import (
                ArkasrEncoderCudaGraphRunner,
            )

            runner = ArkasrEncoderCudaGraphRunner(
                model.audio_encoder,
                max_batch_size=self.encoder_max_batch_size,
                max_mel_frames=self.feature_extractor.nb_max_frames,
            )
            runner.capture_working_set(self.feature_extractor.feature_size)
            model.encoder_cuda_graph_runner = runner
            logger.info(
                "ARK-ASR encoder CUDA graphs enabled (working-set precapture, max_batch=%d)",
                self.encoder_max_batch_size,
            )
        init_mm_embedding_cache(self.mm_embedding_cache_size_bytes)
        if self.enable_pre_lm_encoder:
            # note (guozhihao-224): constructed after generation CUDA graphs so the
            # encoder's dedicated stream never interleaves with graph capture.
            self.audio_encoder_service = ArkasrPreLMEncoderService(
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
                max_queue_size=self.pre_lm_max_pending,
            )

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        del model
        return request_builders.make_arkasr_scheduler_adapters(
            tokenizer=self.tokenizer,
            feature_extractor=self.feature_extractor,
            max_new_tokens=self.max_new_tokens,
            merge_factor=self.merge_factor,
            audio_token_id=self.audio_token_id,
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
            "stream_output_builder": request_builders.make_arkasr_stream_output_builder(
                tokenizer=self.tokenizer,
                min_emit_interval_s=self.stream_emit_interval_s,
            ),
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
        }
