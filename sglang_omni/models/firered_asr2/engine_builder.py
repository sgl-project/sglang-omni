# SPDX-License-Identifier: Apache-2.0
"""Build a FireRedASR2 engine using the shared native ASR runtime."""

import logging

import torch
from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.managers.mm_utils import init_mm_embedding_cache
from sglang.srt.server_args import ServerArgs
from transformers import AutoConfig, AutoTokenizer

from sglang_omni.models.firered_asr2.audio import FireRedFeatureExtractor
from sglang_omni.models.firered_asr2.encoder_graph import FireRedEncoderGraph
from sglang_omni.models.firered_asr2.npu_prefill_graph import FireRedNpuPrefillGraph
from sglang_omni.models.firered_asr2.request_builders import (
    FireRedRequestBuilder,
    FireRedRequestData,
    FireRedResultAdapter,
    make_firered_asr2_scheduler_adapters,
    make_firered_asr2_stream_output_builder,
)
from sglang_omni.platforms import current_platform
from sglang_omni.scheduling.engine_factory import (
    AsrEngineBuilder,
    GenerationDefaults,
    SchedulerExtras,
)
from sglang_omni.scheduling.generation_batch_policy import CudaGraphBackend
from sglang_omni.utils.checkpoint import resolve_checkpoint

logger = logging.getLogger(__name__)


class FireRedASR2EngineBuilder(AsrEngineBuilder[FireRedRequestData]):
    model_name = "FireRedASR2-LLM"
    model_arch_override = "FireRedASR2ForConditionalGeneration"
    context_length = 4096
    supports_breakable_prefill_cuda_graph = True

    def __init__(
        self,
        *,
        max_new_tokens: int,
        max_running_requests: int,
        mm_embedding_cache_size_bytes: int,
        enable_encoder_cuda_graph: bool,
    ) -> None:
        if mm_embedding_cache_size_bytes < 0:
            raise ValueError("mm_embedding_cache_size_bytes must be nonnegative")
        else:
            pass
        self.max_new_tokens = max_new_tokens
        self.max_running_requests = max_running_requests
        self.mm_embedding_cache_size_bytes = mm_embedding_cache_size_bytes
        self.enable_encoder_cuda_graph = enable_encoder_cuda_graph

    def resolve_checkpoint(self, model_path: str) -> str:
        return resolve_checkpoint(model_path)

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.feature_extractor = FireRedFeatureExtractor.from_pretrained(checkpoint_dir)
        self.config = AutoConfig.from_pretrained(checkpoint_dir)

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        return {
            "max_running_requests": self.max_running_requests,
            "disable_cuda_graph": False,
            "cuda_graph_backend_prefill": (
                CudaGraphBackend.BREAKABLE
                if current_platform.is_cuda()
                else CudaGraphBackend.DISABLED
            ),
            "disable_overlap_schedule": True,
            "enable_torch_compile": False,
            "max_prefill_tokens": 4096,
            "chunked_prefill_size": 4096,
            "sampling_backend": "pytorch",
            "dtype": dtype,
        }

    def make_adapters(
        self, model: torch.nn.Module
    ) -> tuple[FireRedRequestBuilder, FireRedResultAdapter]:
        return make_firered_asr2_scheduler_adapters(
            tokenizer=self.tokenizer,
            feature_extractor=self.feature_extractor,
            max_new_tokens=self.max_new_tokens,
            downsample_rate=int(self.config.encoder_downsample_rate),
        )

    def validate_before_infrastructure(self, server_args: ServerArgs) -> None:
        super().validate_before_infrastructure(server_args)
        config = resolved_view(server_args)
        if 0 < config.chunked_prefill_size < config.page_size:
            raise ValueError(
                "chunked_prefill_size must cover at least one KV cache page"
            )
        else:
            pass

    def setup_model_resources(
        self,
        model: torch.nn.Module,
        server_args: ServerArgs,
        *,
        generation_cuda_graph_enabled: bool,
    ) -> None:
        # note (cocoa00): Local model registration bypasses SGLang multimodal cache setup.
        init_mm_embedding_cache(self.mm_embedding_cache_size_bytes)
        backend = current_platform.get_device_graph_backend(
            next(model.encoder.parameters()).device
        )
        if (
            generation_cuda_graph_enabled
            and self.enable_encoder_cuda_graph
            and backend is not None
        ):
            if current_platform.is_npu():
                # note (cocoa00): Conv2D graph capture requires the ACLNN path.
                torch.npu.config.allow_internal_format = False
            else:
                pass
            model.encoder_graph = FireRedEncoderGraph(
                model.encoder, model.encoder_projector, backend
            )
        else:
            pass

        if (
            generation_cuda_graph_enabled
            and current_platform.is_npu()
            and not resolved_view(server_args).disable_prefill_cuda_graph
        ):
            model.prefill_graph = FireRedNpuPrefillGraph(model.language_model)
            logger.info(
                "FireRed NPU prefill graphs enabled inside the model; captures are lazy"
            )
        else:
            pass

    def extra_scheduler_kwargs(self) -> SchedulerExtras[FireRedRequestData]:
        return {
            "stream_output_builder": make_firered_asr2_stream_output_builder(
                self.tokenizer
            ),
            "request_build_max_workers": 2,
            "request_build_max_pending": 16,
        }
