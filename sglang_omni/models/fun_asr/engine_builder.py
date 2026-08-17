# SPDX-License-Identifier: Apache-2.0
"""Fun-ASR SGLang engine builder."""

from __future__ import annotations

import logging
from typing import Any

from sglang.srt.managers.mm_utils import init_mm_embedding_cache
from transformers import AutoFeatureExtractor, AutoTokenizer

from sglang_omni.models.fun_asr import request_builders
from sglang_omni.models.fun_asr.encoder_service import (
    FunASRPreLMEncoderService,
    build_cache_namespace,
)
from sglang_omni.models.fun_asr.tool_funcs.audio_lengths import (
    fun_asr_low_frame_rate_length,
)
from sglang_omni.scheduling.engine_factory import AsrEngineBuilder
from sglang_omni.scheduling.generation_batch_policy import (
    CudaGraphBackend,
    build_default_prefill_cuda_graph_bs,
)
from sglang_omni.utils.gpu_compat import get_visible_gpu_sm_version

logger = logging.getLogger(__name__)


class FunASREngineBuilder(AsrEngineBuilder):
    model_name = "Fun-ASR"
    model_arch_override = "FunAsrNanoForConditionalGeneration"
    supports_breakable_prefill_cuda_graph = True

    def __init__(
        self,
        *,
        max_running_requests: int,
        max_new_tokens: int,
        mem_fraction_static: float | None,
        mm_embedding_cache_size_bytes: int,
        enable_torch_compile: bool,
        enable_encoder_torch_compile: bool,
        enable_encoder_cuda_graph: bool,
        enable_async_decode: bool,
        async_decode_min_batch_size: int,
        prefill_coalesce_requests: int,
        prefill_coalesce_wait_ms: float,
        mm_attention_backend: str | None,
        enable_pre_lm_encoder: bool,
        pre_lm_cache_max_entries: int,
        pre_lm_cache_size_bytes: int,
        pre_lm_max_batch_size: int,
        pre_lm_max_batch_wait_ms: int,
        request_build_max_workers: int,
        request_build_max_pending: int | None,
        stream_emit_interval_s: float,
    ) -> None:
        self.max_running_requests = max_running_requests
        self.max_new_tokens = max_new_tokens
        self.mem_fraction_static = mem_fraction_static
        self.mm_embedding_cache_size_bytes = mm_embedding_cache_size_bytes
        self.enable_torch_compile = enable_torch_compile
        self.enable_encoder_torch_compile = enable_encoder_torch_compile
        self.enable_encoder_cuda_graph = enable_encoder_cuda_graph
        self.enable_async_decode = enable_async_decode
        self.async_decode_min_batch_size = async_decode_min_batch_size
        self.prefill_coalesce_requests = prefill_coalesce_requests
        self.prefill_coalesce_wait_ms = prefill_coalesce_wait_ms
        self.mm_attention_backend = mm_attention_backend
        self.enable_pre_lm_encoder = enable_pre_lm_encoder
        self.pre_lm_cache_max_entries = pre_lm_cache_max_entries
        self.pre_lm_cache_size_bytes = pre_lm_cache_size_bytes
        self.pre_lm_max_batch_size = pre_lm_max_batch_size
        self.pre_lm_max_batch_wait_ms = pre_lm_max_batch_wait_ms
        self.request_build_max_workers = request_build_max_workers
        self.request_build_max_pending = request_build_max_pending
        self.stream_emit_interval_s = stream_emit_interval_s
        self.tokenizer: Any = None
        self.feature_extractor: Any = None
        self.audio_encoder_service: FunASRPreLMEncoderService | None = None
        self.context_length = 0

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_dir, trust_remote_code=True
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            checkpoint_dir, trust_remote_code=True
        )
        encoder_token_count = int(
            fun_asr_low_frame_rate_length(self.feature_extractor.nb_max_frames)
        )
        prompt_overhead = request_builders.fun_asr_prompt_overhead_tokens(
            self.tokenizer
        )
        self.context_length = (
            encoder_token_count + self.max_new_tokens + prompt_overhead
        )

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "max_running_requests": self.max_running_requests,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "enable_torch_compile": self.enable_torch_compile,
            "mem_fraction_static": self.mem_fraction_static,
            "max_prefill_tokens": 4096,
            "chunked_prefill_size": 4096,
            # Qualified capture budget; longer prefills run eager.
            "cuda_graph_backend_prefill": CudaGraphBackend.BREAKABLE,
            "cuda_graph_bs_prefill": build_default_prefill_cuda_graph_bs(256),
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
        if self.enable_encoder_cuda_graph:
            # Capture needs the eager forwards; a dynamo-compiled callable
            # cannot be captured, so the graph takes precedence over encoder
            # compile.
            if self.enable_encoder_torch_compile:
                logger.warning(
                    "enable_encoder_cuda_graph supersedes "
                    "enable_encoder_torch_compile; the encoder runs from "
                    "captured CUDA graphs (eager capture), not dynamo"
                )
            from sglang_omni.models.fun_asr.encoder_cuda_graph import (
                FunASREncoderCudaGraphRunner,
            )

            model.encoder_cuda_graph_runner = FunASREncoderCudaGraphRunner(
                model.audio_tower,
                model.multi_modal_projector,
                max_batch_size=self.pre_lm_max_batch_size,
            )
            logger.info(
                "Fun-ASR encoder CUDA graphs enabled "
                "(lazy capture per batch/length bucket, max_batch=%d)",
                self.pre_lm_max_batch_size,
            )
        elif self.enable_encoder_torch_compile:
            from sglang_omni.models.fun_asr.stages import _compile_fun_asr_audio_encoder

            _compile_fun_asr_audio_encoder(
                model,
                warmup_inference_mode=self.enable_pre_lm_encoder,
            )
        init_mm_embedding_cache(self.mm_embedding_cache_size_bytes)

    def setup_runtime_resources(self, model: Any, server_args: Any) -> None:
        if not self.enable_pre_lm_encoder:
            return
        self.audio_encoder_service = FunASRPreLMEncoderService(
            model,
            cache_namespace=build_cache_namespace(
                model,
                model_path=self.checkpoint_dir,
                feature_extractor=self.feature_extractor,
                mm_attention_backend=server_args.mm_attention_backend,
            ),
            cache_max_entries=self.pre_lm_cache_max_entries,
            cache_max_bytes=self.pre_lm_cache_size_bytes,
            max_batch_size=self.pre_lm_max_batch_size,
            max_batch_wait_ms=self.pre_lm_max_batch_wait_ms,
        )

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        del model
        return request_builders.make_fun_asr_scheduler_adapters(
            tokenizer=self.tokenizer,
            feature_extractor=self.feature_extractor,
            max_new_tokens=self.max_new_tokens,
            context_length=self.context_length,
            audio_encoder_service=self.audio_encoder_service,
        )

    def extra_scheduler_callbacks(self) -> dict[str, Any]:
        return {
            "shutdown_callback": (
                self.audio_encoder_service.close
                if self.audio_encoder_service is not None
                else None
            )
        }

    def cleanup_build_failure(self) -> None:
        if self.audio_encoder_service is not None:
            self.audio_encoder_service.close()

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {
            "stream_output_builder": request_builders.make_fun_asr_stream_output_builder(
                tokenizer=self.tokenizer,
                min_emit_interval_s=self.stream_emit_interval_s,
            ),
            "enable_async_decode": self.enable_async_decode,
            "async_decode_min_batch_size": self.async_decode_min_batch_size,
            "prefill_coalesce_requests": self.prefill_coalesce_requests,
            "prefill_coalesce_wait_ms": self.prefill_coalesce_wait_ms,
            "request_build_max_workers": self.request_build_max_workers,
            "request_build_max_pending": self.request_build_max_pending,
        }
