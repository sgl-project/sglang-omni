# SPDX-License-Identifier: Apache-2.0
"""Voxtral TTS SGLang generation engine builder."""

from __future__ import annotations

import importlib
from typing import Any

from sglang_omni.models.voxtral_tts import request_builders
from sglang_omni.models.voxtral_tts.pipeline import stages as voxtral_stages
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder


class VoxtralTtsEngineBuilder(TtsEngineBuilder):
    model_name = "Voxtral TTS"
    context_length = 8192

    def __init__(self) -> None:
        self.decrypted_config_file: str | None = None
        self.voice_embeddings: dict[str, Any] = {}

    def resolve_checkpoint(self, model_path: str) -> str:
        return voxtral_stages._resolve_checkpoint(model_path)

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        self.decrypted_config_file = voxtral_stages._write_voxtral_sglang_config(
            checkpoint_dir
        )

    def generation_defaults(
        self,
        *,
        dtype: str,
    ) -> dict[str, Any]:
        del dtype
        return {
            "max_running_requests": 16,
            "dtype": "bfloat16",
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "decrypted_config_file": self.decrypted_config_file,
            "enable_torch_compile": True,
            "mem_fraction_static": 0.85,
            "max_prefill_tokens": 8192,
            "sampling_backend": "pytorch",
        }

    def customize_server_args(self, server_args: Any) -> None:
        if getattr(server_args, "enable_torch_compile", False):
            voxtral_stages._enable_inductor_gemm_autotune()

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        del model_worker, gpu_id, server_args
        self.voice_embeddings = voxtral_stages._load_voxtral_voice_embeddings(
            checkpoint_dir,
            device,
        )

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        model_runner_mod = importlib.import_module(
            "sglang_omni.models.voxtral_tts.model_runner"
        )

        return model_runner_mod.VoxtralTTSModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        return request_builders.make_voxtral_scheduler_adapters(
            model=model,
            voice_embeddings=self.voice_embeddings,
        )
