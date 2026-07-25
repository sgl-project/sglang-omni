# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS SGLang engine builder."""

from __future__ import annotations

import importlib
from typing import Any

from sglang_omni.models.qwen3_tts import request_builders
from sglang_omni.models.qwen3_tts import stages as qwen3_stages
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder


class Qwen3TtsEngineBuilder(TtsEngineBuilder):
    model_name = "Qwen3-TTS"
    context_length = 8192
    model_arch_override = "Qwen3TTSTalker"

    def __init__(self, *, attn_implementation: str | None = None) -> None:
        self.attn_implementation = attn_implementation
        self.wrapper: Any | None = None

    def resolve_checkpoint(self, model_path: str) -> str:
        qwen3_stages.apply_qwen_tts_transformers_compatibility_patches()
        qwen_tts = importlib.import_module("qwen_tts")
        if not hasattr(qwen_tts, "Qwen3TTSModel"):
            raise ImportError("qwen_tts does not expose Qwen3TTSModel")

        return super().resolve_checkpoint(model_path)

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        del checkpoint_dir
        qwen3_stages.apply_qwen_tts_transformers_compatibility_patches()
        qwen3_stages._register_qwen3_tts_hf_config()

    def generation_defaults(
        self,
        *,
        dtype: str,
    ) -> dict[str, Any]:
        return {
            "max_running_requests": 16,
            "cuda_graph_max_bs": 32,
            "torch_compile_max_bs": 32,
            "dtype": dtype,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "enable_torch_compile": True,
            "mem_fraction_static": 0.85,
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
        del gpu_id, server_args
        from qwen_tts import Qwen3TTSModel
        from transformers import AutoProcessor

        model = model_worker.model_runner.model
        speech_tokenizer = qwen3_stages._load_qwen3_tts_tokenizer(
            checkpoint_dir,
            device=device,
            dtype=self.dtype,
            attn_implementation=self.attn_implementation,
        )
        model.load_speech_tokenizer(speech_tokenizer)
        processor = AutoProcessor.from_pretrained(
            checkpoint_dir,
            fix_mistral_regex=True,
        )
        self.wrapper = Qwen3TTSModel(
            model=model,
            processor=processor,
            generate_defaults=qwen3_stages._load_qwen3_tts_generate_defaults(
                checkpoint_dir
            ),
        )
        request_builders.set_qwen3_tts_preprocessing_context(
            model=model,
            wrapper=self.wrapper,
        )

    def compile_model(self, model: Any, server_args: Any) -> None:
        if bool(getattr(server_args, "enable_torch_compile", False)):
            qwen3_stages._compile_qwen3_tts_backbone(model)
            server_args.enable_torch_compile = False

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        model_runner_mod = importlib.import_module(
            "sglang_omni.models.qwen3_tts.model_runner"
        )

        return model_runner_mod.Qwen3TTSModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        return request_builders.make_qwen3_tts_scheduler_adapters(
            model=model,
            wrapper=self.wrapper,
        )

    def make_abort_callback(self) -> Any | None:
        return request_builders.cleanup_prepared_qwen3_tts_request
