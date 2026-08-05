# SPDX-License-Identifier: Apache-2.0
"""Fun-CosyVoice3 SGLang engine builder."""

from __future__ import annotations

import importlib
import os
from typing import Any

from sglang_omni.models.fun_cosyvoice3 import request_builders
from sglang_omni.models.fun_cosyvoice3.utils import (
    CosyVoice3Tokenizer,
    SpeakerEncoder,
    SpeechTokenizerV3,
)
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder
from sglang_omni.utils.checkpoint import resolve_checkpoint as _resolve_checkpoint


class FunCosyVoice3EngineBuilder(TtsEngineBuilder):
    model_name = "Fun-CosyVoice3"
    context_length = 32768
    model_arch_override = "FunCosyVoice3SGLangModel"

    def generation_defaults(
        self,
        *,
        dtype: str,
    ) -> dict[str, Any]:
        return {
            "max_running_requests": 32,
            "cuda_graph_max_bs": 32,
            "torch_compile_max_bs": 32,
            "dtype": dtype,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "enable_torch_compile": False,
            "mem_fraction_static": 0.85,
            "max_prefill_tokens": 4096,
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
        del gpu_id, server_args
        model = model_worker.model_runner.model

        tokenizer_path = os.path.join(checkpoint_dir, "CosyVoice-BlankEN")
        speech_tokenizer_path = os.path.join(
            checkpoint_dir, "speech_tokenizer_v3.onnx"
        )
        campplus_path = os.path.join(checkpoint_dir, "campplus.onnx")

        tokenizer = CosyVoice3Tokenizer(tokenizer_path)
        speech_tokenizer = SpeechTokenizerV3(speech_tokenizer_path, device=device)
        speaker_encoder = SpeakerEncoder(campplus_path, device=device)

        request_builders.set_cosyvoice3_preprocessing_context(
            model=model,
            tokenizer=tokenizer,
            speech_tokenizer=speech_tokenizer,
            speaker_encoder=speaker_encoder,
        )

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        model_runner_mod = importlib.import_module(
            "sglang_omni.models.fun_cosyvoice3.model_runner"
        )
        return model_runner_mod.FunCosyVoice3ModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        return request_builders.make_cosyvoice3_scheduler_adapters(model=model)

    def make_abort_callback(self) -> Any | None:
        return request_builders.cleanup_prepared_cosyvoice3_request
