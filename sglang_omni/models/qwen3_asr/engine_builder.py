# SPDX-License-Identifier: Apache-2.0
"""Qwen3-ASR SGLang engine builder."""

from __future__ import annotations

from typing import Any

from sglang.srt.managers.mm_utils import init_mm_embedding_cache
from transformers import AutoFeatureExtractor, AutoTokenizer

from sglang_omni.models.qwen3_asr import request_builders
from sglang_omni.models.qwen3_asr.encoder_service import (
    Qwen3ASRPreLMEncoderService,
    build_cache_namespace,
)
from sglang_omni.scheduling.engine_factory import AsrEngineBuilder
from sglang_omni.utils.gpu_compat import get_visible_gpu_sm_version


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
        mm_attention_backend: str | None,
        request_build_max_workers: int,
        request_build_max_pending: int | None,
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
        self.mm_attention_backend = mm_attention_backend
        self.request_build_max_workers = request_build_max_workers
        self.request_build_max_pending = request_build_max_pending
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
        encoder_token_count = int(self.feature_extractor.nb_max_frames // 2)
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
        }
