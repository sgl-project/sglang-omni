# SPDX-License-Identifier: Apache-2.0
"""Build a FireRedASR2 engine using the shared native ASR runtime."""

import torch
from sglang.srt.managers.mm_utils import init_mm_embedding_cache
from sglang.srt.server_args import ServerArgs
from transformers import AutoConfig, AutoTokenizer

from sglang_omni.models.firered_asr2.audio import FireRedFeatureExtractor
from sglang_omni.models.firered_asr2.request_builders import (
    FireRedRequestBuilder,
    FireRedRequestData,
    FireRedResultAdapter,
    make_firered_asr2_scheduler_adapters,
    make_firered_asr2_stream_output_builder,
)
from sglang_omni.scheduling.engine_factory import (
    AsrEngineBuilder,
    GenerationDefaults,
    SchedulerExtras,
)
from sglang_omni.scheduling.generation_batch_policy import CudaGraphBackend
from sglang_omni.utils.checkpoint import resolve_checkpoint


class FireRedASR2EngineBuilder(AsrEngineBuilder[FireRedRequestData]):
    model_name = "FireRedASR2-LLM"
    model_arch_override = "FireRedASR2ForConditionalGeneration"
    context_length = 4096

    def __init__(
        self,
        *,
        max_new_tokens: int,
        max_running_requests: int,
        mm_embedding_cache_size_bytes: int,
    ) -> None:
        if mm_embedding_cache_size_bytes < 0:
            raise ValueError("mm_embedding_cache_size_bytes must be nonnegative")
        else:
            pass
        self.max_new_tokens = max_new_tokens
        self.max_running_requests = max_running_requests
        self.mm_embedding_cache_size_bytes = mm_embedding_cache_size_bytes

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
            "cuda_graph_backend_prefill": CudaGraphBackend.DISABLED,
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

    def setup_model_resources(
        self,
        model: torch.nn.Module,
        server_args: ServerArgs,
        *,
        generation_cuda_graph_enabled: bool,
    ) -> None:
        # note (cocoa00): Local model registration bypasses SGLang multimodal cache setup.
        init_mm_embedding_cache(self.mm_embedding_cache_size_bytes)

    def extra_scheduler_kwargs(self) -> SchedulerExtras[FireRedRequestData]:
        return {
            "stream_output_builder": make_firered_asr2_stream_output_builder(
                self.tokenizer
            ),
            "request_build_max_workers": 2,
            "request_build_max_pending": 16,
        }
