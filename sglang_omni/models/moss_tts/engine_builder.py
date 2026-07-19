# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS Delay SGLang engine builder."""

from __future__ import annotations

import importlib
from typing import Any

from sglang_omni.models.moss_tts import request_builders
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder


class MossTtsEngineBuilder(TtsEngineBuilder):
    model_name = "MOSS-TTS"
    context_length = 8192
    model_arch_override = "MossTTSDelaySGLangModel"

    def generation_defaults(
        self,
        *,
        dtype: str,
    ) -> dict[str, Any]:
        return {
            "max_running_requests": 16,
            "dtype": dtype,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "enable_torch_compile": False,
            "max_prefill_tokens": 8192,
            "sampling_backend": "pytorch",
            "trust_remote_code": True,
        }

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

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        model_runner_mod = importlib.import_module(
            "sglang_omni.models.moss_tts.model_runner"
        )

        return model_runner_mod.MossTTSModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        return request_builders.make_moss_tts_scheduler_adapters(model=model)

    def make_abort_callback(self) -> Any | None:
        return request_builders.cleanup_prepared_moss_tts_request
