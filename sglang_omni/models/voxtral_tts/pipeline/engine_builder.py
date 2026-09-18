# SPDX-License-Identifier: Apache-2.0
"""Voxtral TTS SGLang generation engine builder."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from sglang_omni.models.voxtral_tts import request_builders
from sglang_omni.models.voxtral_tts.pipeline import stages as voxtral_stages
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker
    from sglang.srt.server_args import ServerArgs
    from torch import Tensor

    from sglang_omni.model_runner.model_worker import ModelWorker
    from sglang_omni.models.voxtral_tts.model_runner import VoxtralTTSModelRunner
    from sglang_omni.models.voxtral_tts.request_builders import VoxtralSGLangRequestData
    from sglang_omni.models.voxtral_tts.sglang_model import VoxtralSGLangTTSModel
    from sglang_omni.proto import StagePayload
    from sglang_omni.scheduling.sglang_backend.output_processor import (
        SGLangOutputProcessor,
    )


class VoxtralTtsEngineBuilder(TtsEngineBuilder):
    model_name = "Voxtral TTS"
    context_length = 8192

    def __init__(self) -> None:
        self.decrypted_config_file: str | None = None
        self.voice_embeddings: dict[str, Tensor] = {}

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        self.decrypted_config_file = voxtral_stages._write_voxtral_sglang_config(
            checkpoint_dir
        )

    def generation_defaults(
        self,
        *,
        dtype: str,
    ) -> dict[str, str | int | float | None]:
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

    def customize_server_args(self, server_args: ServerArgs) -> None:
        if server_args.enable_torch_compile:
            voxtral_stages._enable_inductor_gemm_autotune()

    def setup_model(
        self,
        *,
        model_worker: object,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: object,
    ) -> None:
        del model_worker, gpu_id, server_args
        self.voice_embeddings = voxtral_stages._load_voxtral_voice_embeddings(
            checkpoint_dir,
            device,
        )

    def make_model_runner(
        self,
        model_worker: ModelWorker | MlxTpModelWorker,
        output_proc: SGLangOutputProcessor,
    ) -> VoxtralTTSModelRunner:
        model_runner_mod = importlib.import_module(
            "sglang_omni.models.voxtral_tts.model_runner"
        )

        return model_runner_mod.VoxtralTTSModelRunner(model_worker, output_proc)

    def make_adapters(self, model: VoxtralSGLangTTSModel | None) -> tuple[
        Callable[[StagePayload], VoxtralSGLangRequestData],
        Callable[[VoxtralSGLangRequestData], StagePayload],
    ]:
        return request_builders.make_voxtral_scheduler_adapters(
            model=model,
            voice_embeddings=self.voice_embeddings,
        )
