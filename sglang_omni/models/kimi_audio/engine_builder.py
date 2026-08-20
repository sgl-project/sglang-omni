# SPDX-License-Identifier: Apache-2.0
"""SGLang engine builder for Kimi-Audio text output."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.models.kimi_audio import request_builders
from sglang_omni.models.kimi_audio.checkpoint import resolve_kimi_audio_text_checkpoint
from sglang_omni.models.kimi_audio.hf_config import KimiAudioConfig
from sglang_omni.models.kimi_audio.processor import KimiAudioProcessor
from sglang_omni.scheduling.engine_factory import AsrEngineBuilder


class KimiAudioEngineBuilder(AsrEngineBuilder):
    model_name = "Kimi-Audio"
    model_arch_override = "KimiAudioForTextGeneration"

    def __init__(
        self,
        *,
        max_running_requests: int,
        max_new_tokens: int,
        mem_fraction_static: float | None,
        enable_torch_compile: bool,
        request_build_max_workers: int,
        request_build_max_pending: int | None,
        audio_tokenizer_path: str,
    ) -> None:
        self.max_running_requests = max_running_requests
        self.max_new_tokens = max_new_tokens
        self.mem_fraction_static = mem_fraction_static
        self.enable_torch_compile = enable_torch_compile
        self.request_build_max_workers = request_build_max_workers
        self.request_build_max_pending = request_build_max_pending
        self.audio_tokenizer_path = audio_tokenizer_path
        self.context_length = 8192
        self.processor: KimiAudioProcessor | None = None

    def resolve_checkpoint(self, model_path: str) -> str:
        return resolve_kimi_audio_text_checkpoint(model_path)

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        config = KimiAudioConfig.from_pretrained(checkpoint_dir)
        self.context_length = int(config.max_position_embeddings)

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        return {
            "max_running_requests": self.max_running_requests,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_torch_compile": self.enable_torch_compile,
            "mem_fraction_static": self.mem_fraction_static,
            "max_prefill_tokens": self.context_length,
            "chunked_prefill_size": -1,
            "sampling_backend": "pytorch",
            "dtype": dtype,
        }

    def setup_model_resources(
        self,
        model: Any,
        server_args: Any,
        *,
        generation_cuda_graph_enabled: bool,
    ) -> None:
        del server_args, generation_cuda_graph_enabled
        parameter = next(model.parameters())
        processor_dtype = parameter.dtype
        if processor_dtype not in (torch.float16, torch.bfloat16):
            processor_dtype = torch.bfloat16
        self.processor = KimiAudioProcessor(
            self.checkpoint_dir,
            device=self.device,
            dtype=processor_dtype,
            audio_tokenizer_path=self.audio_tokenizer_path,
        )

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        del model
        if self.processor is None:
            raise RuntimeError("Kimi-Audio processor was not initialized")
        return request_builders.make_kimi_audio_scheduler_adapters(
            processor=self.processor,
            max_new_tokens=self.max_new_tokens,
            context_length=self.context_length,
        )

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {
            "enable_async_decode": True,
            "async_decode_min_batch_size": 2,
            "request_build_max_workers": self.request_build_max_workers,
            "request_build_max_pending": self.request_build_max_pending,
        }
