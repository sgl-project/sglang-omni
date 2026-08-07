# SPDX-License-Identifier: Apache-2.0
"""SGLang engine builder for MOSS-TTS-Realtime."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.moss_tts_local.engine_builder import MossTtsLocalEngineBuilder
from sglang_omni.models.moss_tts_realtime import request_builders
from sglang_omni.models.moss_tts_realtime.hf_config import (
    register_moss_tts_realtime_hf_config,
)
from sglang_omni.models.moss_tts_realtime.model_runner import MossTTSRealtimeModelRunner


class MossTTSRealtimeEngineBuilder(MossTtsLocalEngineBuilder):
    model_name = "MOSS-TTS-Realtime"
    context_length = 40960
    model_arch_override = "MossTTSRealtimeSGLangModel"

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        del checkpoint_dir
        register_moss_tts_realtime_hf_config()

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        return {
            "max_running_requests": 1,
            "dtype": dtype,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "enable_torch_compile": False,
            "max_prefill_tokens": 40960,
            "sampling_backend": "pytorch",
            "trust_remote_code": True,
        }

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        return MossTTSRealtimeModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        return request_builders.make_scheduler_adapters(model=model)

    def make_abort_callback(self) -> Any | None:
        assert self.model is not None
        model = self.model

        def abort_request(request_id: str) -> None:
            request_builders.cleanup_prepared_request(request_id)
            model.reset_request(request_id)

        return abort_request


__all__ = ["MossTTSRealtimeEngineBuilder"]
