# SPDX-License-Identifier: Apache-2.0
"""Whisper ASR SGLang engine builder."""

from __future__ import annotations

import logging
from typing import Any

from sglang_omni.scheduling.engine_factory import AsrEngineBuilder
from sglang_omni.scheduling.generation_batch_policy import get_decode_cuda_graph_bs
from sglang_omni.utils.gpu_compat import get_visible_gpu_sm_version
from sglang_omni.utils.gpu_memory import format_bytes_gib, get_process_gpu_memory_bytes

logger = logging.getLogger(__name__)


class WhisperASREngineBuilder(AsrEngineBuilder):
    model_name = "Whisper ASR"
    model_arch_override = "WhisperForConditionalGeneration"

    def __init__(
        self,
        *,
        max_running_requests: int,
        max_new_tokens: int,
        mem_fraction_static: float,
        enable_torch_compile: bool,
    ) -> None:
        self.max_running_requests = max_running_requests
        self.max_new_tokens = max_new_tokens
        self.mem_fraction_static = mem_fraction_static
        self.enable_torch_compile = enable_torch_compile
        self.processor: Any = None
        self.tokenizer: Any = None
        self.generation_config: Any = None
        self.encoder_token_count = 0
        self.context_length = 0

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        from transformers import AutoProcessor, GenerationConfig

        self.processor = AutoProcessor.from_pretrained(checkpoint_dir)
        self.tokenizer = self.processor.tokenizer
        self.generation_config = GenerationConfig.from_pretrained(checkpoint_dir)
        self.encoder_token_count = int(
            self.processor.feature_extractor.nb_max_frames // 2
        )
        self.context_length = self.encoder_token_count + self.max_new_tokens + 8

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
            "dtype": dtype,
        }

    def validate_before_infrastructure(self, server_args: Any) -> None:
        super().validate_before_infrastructure(server_args)
        logger.info(
            "Whisper ASR runtime profile: sm=%s dtype=%s "
            "attention_backend=%s encoder_attention_backend=torch_sdpa "
            "cuda_graph=%s cuda_graph_bs=%s torch_compile=%s "
            "max_running_requests=%s mem_fraction_static=%s",
            get_visible_gpu_sm_version(self.gpu_id),
            server_args.dtype,
            server_args.attention_backend,
            not server_args.disable_cuda_graph,
            get_decode_cuda_graph_bs(server_args),
            server_args.enable_torch_compile,
            server_args.max_running_requests,
            server_args.mem_fraction_static,
        )
        self._log_memory_checkpoint("pre_model_load")

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        del model_worker, checkpoint_dir, device, gpu_id, server_args
        self._log_memory_checkpoint("post_static_allocation")

    def post_cuda_graph_setup(self, model: Any, server_args: Any) -> None:
        del model, server_args
        self._log_memory_checkpoint("post_cuda_graph_capture")

    def _log_memory_checkpoint(self, checkpoint: str) -> None:
        logger.info(
            "Whisper ASR memory checkpoint=%s gpu=%d process_gpu_memory=%s",
            checkpoint,
            self.gpu_id,
            format_bytes_gib(get_process_gpu_memory_bytes(self.gpu_id)),
        )

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        del model
        from sglang_omni.models.whisper_asr.request_builders import (
            make_whisper_scheduler_adapters,
        )

        return make_whisper_scheduler_adapters(
            processor=self.processor,
            tokenizer=self.tokenizer,
            generation_config=self.generation_config,
            encoder_token_count=self.encoder_token_count,
            max_new_tokens=self.max_new_tokens,
        )
