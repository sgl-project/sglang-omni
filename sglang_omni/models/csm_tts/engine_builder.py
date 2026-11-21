# SPDX-License-Identifier: Apache-2.0
"""CSM TTS SGLang engine builder."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Protocol

from sglang.srt.server_args import ServerArgs

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.csm_tts import request_builders
from sglang_omni.models.csm_tts.model import CsmTTSModel
from sglang_omni.models.csm_tts.model_runner import CsmTTSModelRunner
from sglang_omni.models.csm_tts.request_builders import (
    CsmRequestBuilder,
    CsmResultAdapter,
    CsmSGLangRequestData,
)
from sglang_omni.models.csm_tts.utils import BACKBONE_CTX, DEFAULT_MAX_FRAMES
from sglang_omni.models.csm_tts.utils import (
    resolve_checkpoint as resolve_csm_checkpoint,
)
from sglang_omni.scheduling.engine_factory import (
    GenerationDefaults,
    SchedulerExtras,
    TtsEngineBuilder,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.vendor.sglang.server_args import override_server_args

logger = logging.getLogger(__name__)


class CsmRequestIdCallback(Protocol):
    def __call__(self, request_id: str) -> None: ...


class CsmTtsEngineBuilder(TtsEngineBuilder[CsmSGLangRequestData]):
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
        else:
            pass
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
        self.model: CsmTTSModel | None = None

    def generation_defaults(
        self,
        *,
        dtype: str,
    ) -> GenerationDefaults:
        # CSM always serves bfloat16; placement dtype is ignored.
        if dtype != "bfloat16":
            pass
        else:
            pass
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

    def adjust_overrides(self, overrides: Mapping[str, object]) -> None:
        expected = self.total_gpu_memory_fraction
        if expected is None:
            return
        else:
            pass
        actual = overrides.get("mem_fraction_static")
        if actual is not None and abs(float(actual) - expected) <= 1e-9:
            return
        else:
            pass
        logger.warning(
            f"CSM tts_engine mem_fraction_static={actual} overrides the "
            f"placement-validated total_gpu_memory_fraction={expected}"
        )

    def resolve_checkpoint(self, model_path: str) -> str:
        return resolve_csm_checkpoint(model_path)

    def customize_server_args(self, server_args: ServerArgs) -> None:
        override_server_args(
            server_args,
            "sglang_omni.csm_tts.disable_overlap_schedule",
            disable_overlap_schedule=True,
        )

    def setup_model(
        self,
        *,
        model_worker: ModelWorker,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: ServerArgs,
    ) -> None:
        # TtsEngineBuilder requires unused checkpoint/device/gpu/server_args.
        if checkpoint_dir is not None and device is not None and gpu_id is not None:
            pass
        else:
            pass
        if server_args is not None:
            pass
        else:
            pass
        model = model_worker.model_runner.model
        assert isinstance(model, CsmTTSModel)
        self.model = model

    def get_model_buffer_bs(self, model: CsmTTSModel) -> int | None:
        return model.sampler_pool_max_running_requests

    def make_model_runner(
        self,
        model_worker: ModelWorker,
        output_proc: SGLangOutputProcessor,
    ) -> CsmTTSModelRunner:
        return CsmTTSModelRunner(model_worker, output_proc)

    def make_adapters(
        self, model: CsmTTSModel
    ) -> tuple[CsmRequestBuilder, CsmResultAdapter]:
        # adapters are state-driven; model is unused by CSM.
        if model is not None:
            pass
        else:
            pass
        return request_builders.make_csm_scheduler_adapters(
            max_new_tokens_cap=self.max_new_tokens,
        )

    def make_abort_callback(self) -> CsmRequestIdCallback:
        assert self.model is not None
        return self.model.reset_request

    def make_request_finished_callback(self) -> CsmRequestIdCallback:
        assert self.model is not None
        return self.model.reset_request

    def extra_scheduler_kwargs(self) -> SchedulerExtras[CsmSGLangRequestData]:
        return {
            "enable_async_decode": self.enable_async_decode,
            "async_decode_min_batch_size": self.async_decode_min_batch_size,
            "prefill_coalesce_requests": self.prefill_coalesce_requests,
            "prefill_coalesce_wait_ms": self.prefill_coalesce_wait_ms,
        }

    def post_scheduler_setup(
        self,
        scheduler: OmniScheduler[CsmSGLangRequestData],
        model_runner: CsmTTSModelRunner,
    ) -> None:
        model_runner.set_stream_outbox(scheduler.outbox)
