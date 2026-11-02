# SPDX-License-Identifier: Apache-2.0
"""MOSS-Transcribe-Diarize SGLang engine builder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from sglang.srt.managers.mm_utils import init_mm_embedding_cache
from sglang.srt.server_args import ServerArgs
from transformers import PreTrainedTokenizerBase, ProcessorMixin

from sglang_omni.models.moss_transcribe_diarize import CAPABILITIES, request_builders
from sglang_omni.models.moss_transcribe_diarize.encoder_service import (
    BatchedAudioEncoderService,
)
from sglang_omni.models.moss_transcribe_diarize.request_builders import (
    MossTranscribeDiarizeRequestData,
)
from sglang_omni.platforms import current_platform
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.engine_factory import (
    AsrEngineBuilder,
    GenerationDefaults,
    SchedulerExtras,
)
from sglang_omni.scheduling.generation_batch_policy import (
    CudaGraphBackend,
    CudaGraphConfig,
    build_default_prefill_cuda_graph_bs,
    get_prefill_cuda_graph_backend,
)

if TYPE_CHECKING:

    from sglang_omni.models.moss_transcribe_diarize.sglang_model import (
        MossTranscribeDiarizeForConditionalGeneration,
    )
else:
    pass

# Request slots baked into each full prefill graph; larger prefill batches
# replay eagerly, so this covers the coalesced admission bursts of a C32 load.
FULL_PREFILL_MAX_REQUESTS = 16


class MossTranscribeDiarizeEngineBuilder(
    AsrEngineBuilder[MossTranscribeDiarizeRequestData]
):
    model_name = "MOSS-Transcribe-Diarize"
    model_arch_override = "MossTranscribeDiarizeForConditionalGeneration"
    supports_breakable_prefill_cuda_graph = (
        CAPABILITIES.supports_breakable_prefill_cuda_graph
    )
    supports_full_prefill_cuda_graph = CAPABILITIES.supports_full_prefill_cuda_graph

    def __init__(
        self,
        *,
        max_running_requests: int,
        max_new_tokens: int | None,
        context_length: int | None,
        mem_fraction_static: float | None,
        mm_embedding_cache_size_bytes: int,
        encoder_cache_size_bytes: int,
        enable_torch_compile: bool | None,
        torch_compile_max_bs: int,
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
        enable_trimmed_fbank: bool = True,
    ) -> None:
        self.max_running_requests = max_running_requests
        self.requested_max_new_tokens = max_new_tokens
        self.requested_context_length = context_length
        self.mem_fraction_static = mem_fraction_static
        self.mm_embedding_cache_size_bytes = mm_embedding_cache_size_bytes
        self.encoder_cache_size_bytes = encoder_cache_size_bytes
        self.enable_torch_compile = enable_torch_compile
        self.torch_compile_max_bs = torch_compile_max_bs
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
        self.enable_trimmed_fbank = enable_trimmed_fbank
        self.request_build_max_workers = request_build_max_workers
        self.request_build_max_pending = request_build_max_pending
        self.stream_emit_interval_s = stream_emit_interval_s
        self.processor: ProcessorMixin | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.audio_encoder_service: BatchedAudioEncoderService | None = None
        self.max_new_tokens = 0
        self.context_length = 0
        self.full_prefill_enabled = False

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        from transformers import AutoProcessor

        from sglang_omni.models.moss_transcribe_diarize import stages

        with stages.missing_additional_chat_templates_compat():
            self.processor = AutoProcessor.from_pretrained(
                checkpoint_dir, trust_remote_code=True
            )
        self.tokenizer = self.processor.tokenizer
        self.max_new_tokens = (
            int(self.requested_max_new_tokens)
            if self.requested_max_new_tokens is not None
            else stages.default_max_new_tokens(checkpoint_dir)
        )
        self.context_length = (
            int(self.requested_context_length)
            if self.requested_context_length is not None
            else stages.default_context_length(checkpoint_dir)
        )

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        # note (Xinyu): cached-prefix extends commonly contain one or two new
        # tokens, so keep exact graph buckets below the shared ladder's 4-token
        # floor instead of failing the prefill padding-factor replay guard.
        prefill_cuda_graph_bs = [
            1,
            2,
            *build_default_prefill_cuda_graph_bs(4096),
        ]
        return {
            "max_running_requests": self.max_running_requests,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "enable_torch_compile": self.enable_torch_compile,
            "torch_compile_max_bs": self.torch_compile_max_bs,
            "mem_fraction_static": self.mem_fraction_static,
            "max_prefill_tokens": 4096,
            "chunked_prefill_size": 4096,
            "sampling_backend": "pytorch",
            # Prompts are ~100-700 tokens, so the breakable graph's 28 eager
            # attention breaks dominate prefill host time on the scheduler
            # thread; one full body graph per bucket removes them.
            "cuda_graph_backend_prefill": (
                CudaGraphBackend.FULL
                if current_platform.is_cuda()
                else CudaGraphBackend.BREAKABLE
            ),
            "cuda_graph_bs_prefill": prefill_cuda_graph_bs,
            "cuda_graph_config": {
                "prefill": {"full_prefill_max_req": FULL_PREFILL_MAX_REQUESTS}
            },
            "dtype": dtype,
        }

    def adjust_overrides(self, overrides: dict[str, object]) -> None:
        # note (Dayuxiaoshui): context_length is an explicit server-args
        # parameter, so consume the operator override before the shared builder
        # expands overrides.
        if "context_length" in overrides:
            self.context_length = int(overrides.pop("context_length"))
        else:
            pass

        if current_platform.is_cuda():
            graph_config = overrides.get("cuda_graph_config") or {}
            if isinstance(graph_config, CudaGraphConfig):
                graph_config = graph_config.to_dict()
            else:
                pass
            prefill_config = graph_config.get("prefill") or {}
            overrides["cuda_graph_config"] = {
                **graph_config,
                "prefill": {
                    "full_prefill_max_req": FULL_PREFILL_MAX_REQUESTS,
                    **prefill_config,
                },
            }
        else:
            pass

    def customize_server_args(self, server_args: ServerArgs) -> None:
        # note (Dayuxiaoshui): adapters must use the context length finalized by
        # ServerArgs, matching the pre-refactor factory behavior.
        self.context_length = int(server_args.context_length)

    def setup_model_resources(
        self,
        model: MossTranscribeDiarizeForConditionalGeneration,
        server_args: ServerArgs,
        *,
        generation_cuda_graph_enabled: bool,
    ) -> None:
        self.full_prefill_enabled = (
            generation_cuda_graph_enabled
            and current_platform.is_cuda()
            and get_prefill_cuda_graph_backend(server_args) == CudaGraphBackend.FULL
        )
        input_feature_len = int(self.processor.feature_extractor.nb_max_frames)
        if self.encoder_torch_compile:
            model.compile_encoder(self.encoder_chunk_buckets, input_feature_len)
        elif generation_cuda_graph_enabled:
            model.init_encoder_graphs(self.encoder_chunk_buckets, input_feature_len)
        else:
            pass
        init_mm_embedding_cache(self.mm_embedding_cache_size_bytes)
        model.init_encoder_cache(self.encoder_cache_size_bytes)

    def setup_runtime_resources(
        self,
        model: MossTranscribeDiarizeForConditionalGeneration,
        server_args: ServerArgs,
    ) -> None:
        del server_args
        self.audio_encoder_service = BatchedAudioEncoderService(
            model,
            max_batch_size=self.encoder_max_batch_size,
        )

    def make_adapters(self, model: object) -> tuple[
        Callable[[StagePayload], MossTranscribeDiarizeRequestData],
        Callable[[MossTranscribeDiarizeRequestData], StagePayload],
    ]:
        del model
        return request_builders.make_moss_transcribe_diarize_scheduler_adapters(
            processor=self.processor,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
            context_length=self.context_length,
            duration_scaled_default=self.requested_max_new_tokens is None,
            audio_encoder_service=self.audio_encoder_service,
            enable_trimmed_fbank=self.enable_trimmed_fbank,
        )

    def extra_scheduler_kwargs(
        self,
    ) -> SchedulerExtras[MossTranscribeDiarizeRequestData]:
        scheduler_kwargs: SchedulerExtras[MossTranscribeDiarizeRequestData] = {
            "stream_output_builder": (
                request_builders.make_moss_transcribe_diarize_stream_output_builder(
                    tokenizer=self.tokenizer,
                    min_emit_interval_s=self.stream_emit_interval_s,
                )
            ),
            "enable_async_decode": self.enable_async_decode,
            "async_decode_min_batch_size": self.async_decode_min_batch_size,
            # The scheduler otherwise idles on every prefill's sampled-token
            # copy while the next decode could already launch.
            "async_prefill": self.enable_async_decode and self.full_prefill_enabled,
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
        if scheduler_kwargs["async_prefill"]:
            # The full-graph scheduler need not stall for encoder builds. Keep
            # the existing admission behavior on fallback attention backends.
            scheduler_kwargs["request_build_admission_wait_s"] = 0.0
        else:
            pass
        return scheduler_kwargs
