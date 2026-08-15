# SPDX-License-Identifier: Apache-2.0
"""MOSS-Transcribe-Diarize SGLang engine builder."""

from __future__ import annotations

from typing import Any

from sglang.srt.managers.mm_utils import init_mm_embedding_cache

from sglang_omni.models.moss_transcribe_diarize import CAPABILITIES, request_builders
from sglang_omni.models.moss_transcribe_diarize.encoder_service import (
    BatchedAudioEncoderService,
)
from sglang_omni.scheduling.engine_factory import AsrEngineBuilder
from sglang_omni.scheduling.generation_batch_policy import (
    CudaGraphBackend,
    build_default_prefill_cuda_graph_bs,
)


class MossTranscribeDiarizeEngineBuilder(AsrEngineBuilder):
    model_name = "MOSS-Transcribe-Diarize"
    model_arch_override = "MossTranscribeDiarizeForConditionalGeneration"
    supports_breakable_prefill_cuda_graph = (
        CAPABILITIES.supports_breakable_prefill_cuda_graph
    )

    def __init__(
        self,
        *,
        max_running_requests: int,
        max_new_tokens: int | None,
        context_length: int | None,
        mem_fraction_static: float | None,
        mm_embedding_cache_size_bytes: int,
        encoder_cache_size_bytes: int,
        enable_torch_compile: bool,
        enable_async_decode: bool,
        async_decode_min_batch_size: int,
        prefill_coalesce_requests: int,
        prefill_coalesce_wait_ms: float,
        prefill_coalesce_when_idle: bool,
        prefill_coalesce_requires_pending_builds: bool,
        prefill_coalesce_after_builds_during_decode: bool,
        encoder_chunk_buckets: list[int],
        encoder_torch_compile: bool,
        encoder_max_batch_size: int,
        request_build_max_workers: int,
        request_build_max_pending: int | None,
        stream_emit_interval_s: float,
    ) -> None:
        self.max_running_requests = max_running_requests
        self.requested_max_new_tokens = max_new_tokens
        self.requested_context_length = context_length
        self.mem_fraction_static = mem_fraction_static
        self.mm_embedding_cache_size_bytes = mm_embedding_cache_size_bytes
        self.encoder_cache_size_bytes = encoder_cache_size_bytes
        self.enable_torch_compile = enable_torch_compile
        self.enable_async_decode = enable_async_decode
        self.async_decode_min_batch_size = async_decode_min_batch_size
        self.prefill_coalesce_requests = prefill_coalesce_requests
        self.prefill_coalesce_wait_ms = prefill_coalesce_wait_ms
        self.prefill_coalesce_when_idle = prefill_coalesce_when_idle
        self.prefill_coalesce_requires_pending_builds = (
            prefill_coalesce_requires_pending_builds
        )
        self.prefill_coalesce_after_builds_during_decode = (
            prefill_coalesce_after_builds_during_decode
        )
        self.encoder_chunk_buckets = encoder_chunk_buckets
        self.encoder_torch_compile = encoder_torch_compile
        self.encoder_max_batch_size = encoder_max_batch_size
        self.request_build_max_workers = request_build_max_workers
        self.request_build_max_pending = request_build_max_pending
        self.stream_emit_interval_s = stream_emit_interval_s
        self.processor: Any = None
        self.tokenizer: Any = None
        self.audio_encoder_service: BatchedAudioEncoderService | None = None
        self.max_new_tokens = 0
        self.context_length = 0

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        from transformers import AutoProcessor

        from sglang_omni.models.moss_transcribe_diarize import stages

        with stages._missing_additional_chat_templates_compat():
            self.processor = AutoProcessor.from_pretrained(
                checkpoint_dir, trust_remote_code=True
            )
        self.tokenizer = self.processor.tokenizer
        self.max_new_tokens = (
            int(self.requested_max_new_tokens)
            if self.requested_max_new_tokens is not None
            else stages._default_max_new_tokens(checkpoint_dir)
        )
        self.context_length = (
            int(self.requested_context_length)
            if self.requested_context_length is not None
            else stages._default_context_length(checkpoint_dir)
        )

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        return {
            "max_running_requests": self.max_running_requests,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "enable_torch_compile": self.enable_torch_compile,
            "mem_fraction_static": self.mem_fraction_static,
            "max_prefill_tokens": 4096,
            "chunked_prefill_size": 4096,
            "sampling_backend": "pytorch",
            "cuda_graph_backend_prefill": CudaGraphBackend.BREAKABLE,
            "cuda_graph_bs_prefill": build_default_prefill_cuda_graph_bs(4096),
            "dtype": dtype,
        }

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        # note (Dayuxiaoshui): context_length is an explicit server-args
        # parameter, so consume the operator override before the shared builder
        # expands overrides.
        if "context_length" in overrides:
            self.context_length = int(overrides.pop("context_length"))

    def customize_server_args(self, server_args: Any) -> None:
        # note (Dayuxiaoshui): adapters must use the context length finalized by
        # ServerArgs, matching the pre-refactor factory behavior.
        self.context_length = int(server_args.context_length)

    def setup_model_resources(
        self,
        model: Any,
        server_args: Any,
        *,
        generation_cuda_graph_enabled: bool,
    ) -> None:
        del server_args
        input_feature_len = int(self.processor.feature_extractor.nb_max_frames)
        if self.encoder_torch_compile:
            model.compile_encoder(self.encoder_chunk_buckets, input_feature_len)
        elif generation_cuda_graph_enabled:
            model.init_encoder_graphs(self.encoder_chunk_buckets, input_feature_len)
        init_mm_embedding_cache(self.mm_embedding_cache_size_bytes)
        model.init_encoder_cache(self.encoder_cache_size_bytes)

    def setup_runtime_resources(self, model: Any, server_args: Any) -> None:
        del server_args
        self.audio_encoder_service = BatchedAudioEncoderService(
            model,
            max_batch_size=self.encoder_max_batch_size,
        )

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        del model
        return request_builders.make_moss_transcribe_diarize_scheduler_adapters(
            processor=self.processor,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
            context_length=self.context_length,
            duration_scaled_default=self.requested_max_new_tokens is None,
            audio_encoder_service=self.audio_encoder_service,
        )

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {
            "stream_output_builder": (
                request_builders.make_moss_transcribe_diarize_stream_output_builder(
                    tokenizer=self.tokenizer,
                    min_emit_interval_s=self.stream_emit_interval_s,
                )
            ),
            "enable_async_decode": self.enable_async_decode,
            "async_decode_min_batch_size": self.async_decode_min_batch_size,
            "prefill_coalesce_requests": self.prefill_coalesce_requests,
            "prefill_coalesce_wait_ms": self.prefill_coalesce_wait_ms,
            "prefill_coalesce_when_idle": self.prefill_coalesce_when_idle,
            "prefill_coalesce_requires_pending_builds": (
                self.prefill_coalesce_requires_pending_builds
            ),
            "prefill_coalesce_after_builds_during_decode": (
                self.prefill_coalesce_after_builds_during_decode
            ),
            "request_build_max_workers": self.request_build_max_workers,
            "request_build_max_pending": self.request_build_max_pending,
        }
