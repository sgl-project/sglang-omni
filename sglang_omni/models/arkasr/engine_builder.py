# SPDX-License-Identifier: Apache-2.0
"""ARK-ASR SGLang engine builder."""

from __future__ import annotations

from typing import Any

from sglang.srt.managers.mm_utils import init_mm_embedding_cache
from transformers import AutoConfig, AutoTokenizer, WhisperFeatureExtractor

from sglang_omni.models.arkasr import request_builders
from sglang_omni.scheduling.engine_factory import AsrEngineBuilder
from sglang_omni.utils.gpu_compat import get_visible_gpu_sm_version


class ArkasrEngineBuilder(AsrEngineBuilder):
    model_name = "ARK-ASR"
    model_arch_override = "ArkasrForConditionalGeneration"

    def __init__(
        self,
        *,
        max_running_requests: int,
        max_new_tokens: int,
        mem_fraction_static: float | None,
        mm_embedding_cache_size_bytes: int,
        enable_torch_compile: bool,
        mm_attention_backend: str | None,
        request_build_max_workers: int,
        request_build_max_pending: int | None,
    ) -> None:
        self.max_running_requests = max_running_requests
        self.max_new_tokens = int(max_new_tokens)
        self.mem_fraction_static = mem_fraction_static
        self.mm_embedding_cache_size_bytes = mm_embedding_cache_size_bytes
        self.enable_torch_compile = enable_torch_compile
        self.mm_attention_backend = mm_attention_backend
        self.request_build_max_workers = request_build_max_workers
        self.request_build_max_pending = request_build_max_pending
        self.tokenizer: Any = None
        self.feature_extractor: Any = None
        self.merge_factor = 4
        self.audio_token_id = 151663
        self.context_length = 0

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_dir, trust_remote_code=True
        )
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(checkpoint_dir)
        hf_config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
        self.merge_factor = int(getattr(hf_config, "merge_factor", 4))
        self.audio_token_id = int(getattr(hf_config, "audio_token_id", 151663))
        encoder_token_count = int(
            getattr(self.feature_extractor, "nb_max_frames", 3000) // 2
        )
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
        del model, server_args, generation_cuda_graph_enabled
        init_mm_embedding_cache(self.mm_embedding_cache_size_bytes)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        del model
        return request_builders.make_arkasr_scheduler_adapters(
            tokenizer=self.tokenizer,
            feature_extractor=self.feature_extractor,
            max_new_tokens=self.max_new_tokens,
            merge_factor=self.merge_factor,
            audio_token_id=self.audio_token_id,
        )

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {
            "request_build_max_workers": self.request_build_max_workers,
            "request_build_max_pending": self.request_build_max_pending,
        }
