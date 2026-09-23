# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS Delay SGLang engine builder."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from sglang_omni.models.moss_tts import request_builders
from sglang_omni.models.moss_tts.hf_loading import (
    MOSS_TTS_DEFAULT_CONTEXT_LENGTH,
    resolve_moss_tts_context_length,
)
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

    from sglang_omni.model_runner.model_worker import ModelWorker
    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
    from sglang_omni.models.moss_tts.request_builders import MossTTSSGLangRequestData
    from sglang_omni.models.moss_tts.sglang_model import MossTTSDelaySGLangModel
    from sglang_omni.proto import StagePayload
    from sglang_omni.scheduling.messages import OutgoingMessage
    from sglang_omni.scheduling.sglang_backend.output_processor import (
        SGLangOutputProcessor,
    )
    from sglang_omni.scheduling.types import RequestOutput


class MossTtsEngineBuilder(TtsEngineBuilder["MossTTSSGLangRequestData"]):
    model_name = "MOSS-TTS"
    context_length = MOSS_TTS_DEFAULT_CONTEXT_LENGTH
    model_arch_override = "MossTTSDelaySGLangModel"
    supports_context_length_override = True
    supports_breakable_prefill_cuda_graph = True

    def __init__(self, *, total_gpu_memory_fraction: float | None = None) -> None:
        super().__init__()
        self.total_gpu_memory_fraction = total_gpu_memory_fraction

    def infra_kwargs(self) -> dict[str, float]:
        # Note (Jiaxin Deng): without this the declared stage budget stops at the
        # placement validator and KV sizing profiles against whatever the card happens
        # to have free, so capacity would depend on which process loaded first. Emitted
        # only when a budget is declared, so the single-process path is untouched.
        if self.total_gpu_memory_fraction is None:
            return {}
        return {"total_gpu_memory_fraction": self.total_gpu_memory_fraction}

    def resolve_context_length(
        self,
        checkpoint_dir: str,
        *,
        server_args_overrides: Mapping[str, object] | None = None,
    ) -> int:
        return resolve_moss_tts_context_length(
            checkpoint_dir,
            server_args_overrides=server_args_overrides,
        )

    def generation_defaults(
        self,
        *,
        dtype: str,
    ) -> dict[str, str | int]:
        return {
            "max_running_requests": 16,
            "dtype": dtype,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "enable_torch_compile": False,
            "max_prefill_tokens": min(self.context_length, 8192),
            "sampling_backend": "pytorch",
            "trust_remote_code": True,
        }

    def setup_model(
        self,
        *,
        model_worker: ModelWorker | MlxTpModelWorker,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: object,
    ) -> None:
        del checkpoint_dir, device, gpu_id, server_args
        self._model_runner = model_worker.model_runner

    def post_cuda_graph_setup(
        self, model: MossTTSDelaySGLangModel, server_args: object
    ) -> None:
        del server_args
        graph_runner = self._model_runner.decode_cuda_graph_runner
        model.init_sampling_graphs(
            list(graph_runner.capture_bs),
            disable_padding=graph_runner.disable_padding,
        )

    def make_model_runner(
        self,
        model_worker: ModelWorker | MlxTpModelWorker,
        output_proc: SGLangOutputProcessor,
    ) -> MossTTSModelRunner:
        model_runner_mod = importlib.import_module(
            "sglang_omni.models.moss_tts.model_runner"
        )

        return model_runner_mod.MossTTSModelRunner(model_worker, output_proc)

    def make_adapters(self, model: MossTTSDelaySGLangModel | None) -> tuple[
        Callable[[StagePayload], MossTTSSGLangRequestData],
        Callable[[MossTTSSGLangRequestData], StagePayload],
    ]:
        self._stream_output_builder = (
            request_builders.make_moss_tts_stream_output_builder()
        )
        return request_builders.make_moss_tts_scheduler_adapters(model=model)

    def extra_scheduler_kwargs(
        self,
    ) -> dict[
        str,
        Callable[
            [str, MossTTSSGLangRequestData, RequestOutput | None],
            list[OutgoingMessage],
        ],
    ]:
        return {"stream_output_builder": self._stream_output_builder}

    def make_abort_callback(self) -> Callable[[str], None]:
        return request_builders.cleanup_prepared_moss_tts_request
