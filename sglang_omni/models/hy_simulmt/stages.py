# SPDX-License-Identifier: Apache-2.0
"""Build Hy-SimulMT with the shared ASR scheduler."""

from collections.abc import Mapping

from sglang.srt.managers.mm_utils import init_mm_embedding_cache
from sglang.srt.server_args import ServerArgs
from transformers import AutoConfig, AutoTokenizer, WhisperFeatureExtractor

from sglang_omni.models.hy_simulmt.request_builders import (
    HySimulMTRequestBuilder,
    HySimulMTRequestData,
    HySimulMTResultAdapter,
)
from sglang_omni.models.hy_simulmt.sglang_model import HYV3SpeechForCausalLM
from sglang_omni.scheduling.engine_factory import (
    AsrEngineBuilder,
    GenerationDefaults,
    SchedulerExtras,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


class HySimulMTEngineBuilder(AsrEngineBuilder[HySimulMTRequestData]):
    model_name = "Hy-SimulMT"
    model_arch_override = "HYV3SpeechForCausalLM"
    supports_context_length_override = True

    def __init__(
        self,
        max_running_requests: int,
        max_new_tokens: int,
        mem_fraction_static: float | None,
        mm_embedding_cache_size_bytes: int,
    ) -> None:
        self.max_running_requests = max_running_requests
        self.max_new_tokens = max_new_tokens
        self.mem_fraction_static = mem_fraction_static
        self.mm_embedding_cache_size_bytes = mm_embedding_cache_size_bytes
        self.context_length = 8192

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        config = AutoConfig.from_pretrained(checkpoint_dir)
        self.request_builder = HySimulMTRequestBuilder(
            tokenizer=tokenizer,
            feature_extractor=WhisperFeatureExtractor.from_pretrained(checkpoint_dir),
            max_new_tokens=self.max_new_tokens,
            context_length=self.context_length,
            audio_token_id=config.audio_token_id,
            stop_token_ids=[config.eos_token_id, config.turn_end_token_id],
        )
        self.result_adapter = HySimulMTResultAdapter(tokenizer=tokenizer)

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        return {
            "max_running_requests": self.max_running_requests,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "enable_torch_compile": False,
            "mem_fraction_static": self.mem_fraction_static,
            "max_prefill_tokens": 4096,
            "chunked_prefill_size": 4096,
            "sampling_backend": "pytorch",
            "dtype": dtype,
        }

    def customize_server_args(self, server_args: ServerArgs) -> None:
        self.request_builder.context_length = int(server_args.context_length)

    def make_adapters(
        self, model: HYV3SpeechForCausalLM
    ) -> tuple[HySimulMTRequestBuilder, HySimulMTResultAdapter]:
        assert model.config.audio_token_id == self.request_builder.audio_token_id
        init_mm_embedding_cache(self.mm_embedding_cache_size_bytes)
        return self.request_builder, self.result_adapter

    def extra_scheduler_kwargs(self) -> SchedulerExtras[HySimulMTRequestData]:
        return {"stream_output_builder": self.result_adapter.stream_builder()}


def create_hy_simulmt_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    max_running_requests: int = 16,
    max_new_tokens: int = 256,
    mem_fraction_static: float | None = None,
    mm_embedding_cache_size_bytes: int = 0,
    server_args_overrides: Mapping[str, object] | None = None,
) -> OmniScheduler[HySimulMTRequestData]:
    return HySimulMTEngineBuilder(
        max_running_requests=max_running_requests,
        max_new_tokens=max_new_tokens,
        mem_fraction_static=mem_fraction_static,
        mm_embedding_cache_size_bytes=mm_embedding_cache_size_bytes,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )
