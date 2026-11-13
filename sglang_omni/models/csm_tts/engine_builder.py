# SPDX-License-Identifier: Apache-2.0
"""CSM TTS SGLang engine builder."""

from __future__ import annotations

import importlib
import logging
from typing import Any

from sglang_omni.models.csm_tts import request_builders
from sglang_omni.models.csm_tts.utils import BACKBONE_CTX, DEFAULT_MAX_FRAMES
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder
from sglang_omni.vendor.sglang.server_args import override_server_args

logger = logging.getLogger(__name__)


class CsmTtsEngineBuilder(TtsEngineBuilder):
    model_name = "CSM TTS"
    context_length = BACKBONE_CTX
    supports_breakable_prefill_cuda_graph = False

    def __init__(
        self,
        *,
        max_new_tokens: int | None,
        max_running_requests: int,
        cuda_graph_max_bs: int,
        enable_async_decode: bool,
        async_decode_min_batch_size: int,
        prefill_coalesce_requests: int = 0,
        prefill_coalesce_wait_ms: float = 60.0,
        total_gpu_memory_fraction: float | None = None,
    ) -> None:
        if total_gpu_memory_fraction is not None and not (
            0.0 < total_gpu_memory_fraction < 1.0
        ):
            raise ValueError(
                "CSM tts_engine total_gpu_memory_fraction must be in (0, 1): "
                "it drives sglang mem_fraction_static, which requires < 1"
            )
        self.max_new_tokens = (
            DEFAULT_MAX_FRAMES if max_new_tokens is None else max_new_tokens
        )
        self.max_running_requests = max_running_requests
        self.cuda_graph_max_bs = cuda_graph_max_bs
        self.enable_async_decode = enable_async_decode
        self.async_decode_min_batch_size = async_decode_min_batch_size
        self.prefill_coalesce_requests = prefill_coalesce_requests
        self.prefill_coalesce_wait_ms = prefill_coalesce_wait_ms
        self.total_gpu_memory_fraction = total_gpu_memory_fraction
        self.model: Any | None = None

    def generation_defaults(
        self,
        *,
        dtype: str,
    ) -> dict[str, Any]:
        del dtype
        return {
            "max_running_requests": self.max_running_requests,
            "cuda_graph_max_bs": self.cuda_graph_max_bs,
            "disable_cuda_graph": True,
            "disable_radix_cache": True,
            "mem_fraction_static": (
                self.total_gpu_memory_fraction
                if self.total_gpu_memory_fraction is not None
                else 0.5
            ),
            "chunked_prefill_size": 2048,
            "dtype": "bfloat16",
        }

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        expected = self.total_gpu_memory_fraction
        if expected is None:
            return
        actual = overrides.get("mem_fraction_static")
        if actual is not None and abs(actual - expected) <= 1e-9:
            return
        logger.warning(
            "CSM tts_engine mem_fraction_static=%s overrides the "
            "placement-validated total_gpu_memory_fraction=%s",
            actual,
            expected,
        )

    def resolve_checkpoint(self, model_path: str) -> str:
        from sglang_omni.models.csm_tts.utils import resolve_checkpoint

        return resolve_checkpoint(model_path)

    def customize_server_args(self, server_args: Any) -> None:
        override_server_args(
            server_args,
            "sglang_omni.csm_tts.disable_overlap_schedule",
            disable_overlap_schedule=True,
        )

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        del checkpoint_dir, device, gpu_id, server_args
        self.model = model_worker.model_runner.model

    def get_model_buffer_bs(self, model: Any) -> int | None:
        return model.sampler_pool_max_running_requests

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        model_runner_mod = importlib.import_module(
            "sglang_omni.models.csm_tts.model_runner"
        )
        return model_runner_mod.CsmTTSModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        del model
        return request_builders.make_csm_scheduler_adapters(
            max_new_tokens_cap=self.max_new_tokens,
        )

    def make_abort_callback(self) -> Any | None:
        assert self.model is not None
        return self.model.reset_request

    def make_request_finished_callback(self) -> Any | None:
        assert self.model is not None
        return self.model.reset_request

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {
            "enable_async_decode": self.enable_async_decode,
            "async_decode_min_batch_size": self.async_decode_min_batch_size,
            "prefill_coalesce_requests": self.prefill_coalesce_requests,
            "prefill_coalesce_wait_ms": self.prefill_coalesce_wait_ms,
        }

    def post_scheduler_setup(self, scheduler: Any, model_runner: Any) -> None:
        model_runner.set_stream_outbox(scheduler.outbox)
