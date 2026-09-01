# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS-Nano SGLang engine builder."""

from __future__ import annotations

import importlib
from typing import Any

from sglang_omni.models.moss_tts_local.engine_builder import MossTtsLocalEngineBuilder
from sglang_omni.models.moss_tts_nano import request_builders
from sglang_omni.models.moss_tts_nano.hf_config import (
    select_moss_tts_nano_model_config_parser,
)


class MossTtsNanoEngineBuilder(MossTtsLocalEngineBuilder):
    model_name = "MOSS-TTS-Nano"
    context_length = 32768
    model_arch_override = "MossTTSNanoSGLangModel"

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        super().adjust_overrides(overrides)
        tp_size = int(overrides.get("tp_size", 1))
        pp_size = int(overrides.get("pp_size", 1))
        if tp_size != 1 or pp_size != 1:
            raise ValueError(
                "MOSS-TTS-Nano currently requires tp_size=1 and pp_size=1; "
                f"got tp_size={tp_size}, pp_size={pp_size}"
            )
        select_moss_tts_nano_model_config_parser(overrides)

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        module = importlib.import_module(
            "sglang_omni.models.moss_tts_nano.model_runner"
        )
        return module.MossTTSNanoModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        return request_builders.make_moss_tts_nano_scheduler_adapters(model=model)

    def make_abort_callback(self) -> Any | None:
        assert self.model is not None
        model = self.model

        def abort_request(request_id: str) -> None:
            request_builders.cleanup_prepared_moss_tts_nano_request(request_id)
            model.reset_request(request_id)

        return abort_request


EntryClass = MossTtsNanoEngineBuilder
