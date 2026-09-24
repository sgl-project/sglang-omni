# SPDX-License-Identifier: Apache-2.0
"""Chatterbox-Turbo T3 SGLang engine builder."""

from __future__ import annotations

import gc
import os
from typing import Any

import torch

from sglang_omni.models.chatterbox import request_builders
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder


class ChatterboxT3EngineBuilder(TtsEngineBuilder):
    model_name = "Chatterbox-Turbo"
    context_length = 8196

    def __init__(self, *, max_new_tokens: int) -> None:
        from sglang_omni.models.chatterbox.hf_config import (
            CHATTERBOX_T3_MODEL_ARCH_OVERRIDE,
        )

        self.model_arch_override = CHATTERBOX_T3_MODEL_ARCH_OVERRIDE
        self.max_new_tokens = int(max_new_tokens)
        self.tokenizer: Any | None = None
        self._stream_output_builder: Any | None = None

    def _uses_torch_mps(self) -> bool:
        from sglang.srt.hardware_backend.mlx.runtime import use_mlx

        return (
            not use_mlx()
            and self.device is not None
            and torch.device(self.device).type == "mps"
        )

    def validate_before_infrastructure(self, server_args: Any) -> None:
        from sglang.srt.arg_groups.model_override_base import resolved_view

        cfg = resolved_view(server_args)
        if self._uses_torch_mps() and cfg.max_running_requests != 1:
            raise ValueError(
                "Chatterbox-Turbo Torch MPS requires max_running_requests=1"
            )
        super().validate_before_infrastructure(server_args)

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        from sglang_omni.models.chatterbox.hf_config import (
            register_chatterbox_hf_config,
        )

        register_chatterbox_hf_config()
        # The checkpoint ships a t3_turbo_v1.yaml instead of a config.json;
        # write a minimal config.json so SGLang's AutoConfig can load the
        # registered chatterbox_t3 config.
        config_path = os.path.join(checkpoint_dir, "config.json")
        if not os.path.exists(config_path):
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(
                    '{"model_type": "chatterbox_t3", '
                    '"architectures": ["ChatterboxT3SGLangModel"]}\n'
                )

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        from sglang.srt.hardware_backend.mlx.runtime import use_mlx

        if use_mlx():
            return {
                "max_running_requests": 1,
                "disable_cuda_graph": True,
                "disable_overlap_schedule": True,
                "disable_radix_cache": True,
                "enable_torch_compile": False,
                "chunked_prefill_size": -1,
                "mem_fraction_static": 0.85,
                "attention_backend": "torch_native",
                "sampling_backend": "pytorch",
                "dtype": dtype,
                "mlx_enable_sampling": True,
            }
        if self._uses_torch_mps():
            return {
                "max_running_requests": 1,
                "disable_cuda_graph": True,
                "disable_overlap_schedule": True,
                "disable_radix_cache": True,
                "enable_torch_compile": False,
                "chunked_prefill_size": -1,
                "mem_fraction_static": 0.85,
                "attention_backend": "torch_native",
                "sampling_backend": "pytorch",
                "dtype": dtype,
            }
        return {
            "max_running_requests": 64,
            "disable_cuda_graph": False,
            "mem_fraction_static": 0.85,
            "chunked_prefill_size": 8192,
            "dtype": "bfloat16",
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
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

        if self._uses_torch_mps():
            from sglang_omni.models.chatterbox.torch_mps_runner import (
                install_torch_mps_t3_model,
            )

            model_worker.model_runner.model = install_torch_mps_t3_model(
                checkpoint_dir, device, model_worker.model_runner.model
            )
            gc.collect()
            torch.mps.empty_cache()

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        from sglang.srt.hardware_backend.mlx.runtime import use_mlx

        if use_mlx():
            from sglang_omni.model_runner.mlx_model_worker import (
                MlxSchedulerModelRunner,
            )

            return MlxSchedulerModelRunner(model_worker, output_proc)
        if self._uses_torch_mps():
            from sglang_omni.models.chatterbox.torch_mps_runner import (
                ChatterboxT3TorchMpsModelRunner,
            )

            return ChatterboxT3TorchMpsModelRunner(model_worker, output_proc)
        raise NotImplementedError("Chatterbox-Turbo CUDA backend is not wired yet")

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        del model
        request_builder, result_adapter, self._stream_output_builder = (
            request_builders.make_tts_scheduler_adapters(
                tokenizer=self.tokenizer,
                max_new_tokens_cap=self.max_new_tokens,
            )
        )
        return request_builder, result_adapter

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {"stream_output_builder": self._stream_output_builder}
