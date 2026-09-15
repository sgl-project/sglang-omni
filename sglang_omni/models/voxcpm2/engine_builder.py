# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 engine builder for the AR stage."""

from __future__ import annotations

import logging
from typing import Any

from sglang_omni.models.voxcpm2 import constants as C
from sglang_omni.models.voxcpm2.hf_config import VOXCPM2_MODEL_ARCH_OVERRIDE
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder

logger = logging.getLogger(__name__)


class VoxCPM2EngineBuilder(TtsEngineBuilder):
    """Builds the SGLang engine that runs VoxCPM2's two AR stacks."""

    model_name = "voxcpm2"
    # note (Xinhao Tan): the shared builder needs an explicit context limit;
    # use the checkpoint's max_length for the combined text/audio sequence.
    context_length = 8192

    def __init__(
        self,
        *,
        inference_timesteps: int = C.DEFAULT_INFERENCE_TIMESTEPS,
        cfg_value: float = C.DEFAULT_CFG_VALUE,
        max_running_requests: int = 1,
    ) -> None:
        self.inference_timesteps = int(inference_timesteps)
        self.cfg_value = float(cfg_value)
        self.max_running_requests = int(max_running_requests)
        if self.inference_timesteps <= 0:
            raise ValueError("VoxCPM2 inference_timesteps must be positive")
        if self.max_running_requests <= 0:
            raise ValueError("VoxCPM2 max_running_requests must be positive")
        self.model_arch_override = VOXCPM2_MODEL_ARCH_OVERRIDE
        self.model_runner: Any | None = None
        self.tokenizer: Any | None = None

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        from transformers import AutoTokenizer

        from sglang_omni.models.voxcpm2.hf_config import register_voxcpm2_hf_config

        register_voxcpm2_hf_config()
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_dir, trust_remote_code=True
        )

    def resolve_checkpoint(self, model_path: str) -> str:
        from sglang_omni.models.voxcpm2.hf_config import stage_checkpoint_for_autoconfig

        return stage_checkpoint_for_autoconfig(super().resolve_checkpoint(model_path))

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        return {
            # note (Xinhao Tan): the runner builds full-prompt embeddings and
            # masks, but a radix hit schedules only the uncached suffix.
            # Prefix reuse is unsupported until those layouts are reconciled.
            "disable_radix_cache": True,
            "disable_cuda_graph": True,
            "disable_overlap_schedule": True,
            "enable_torch_compile": False,
            "max_running_requests": self.max_running_requests,
            "chunked_prefill_size": 0,
            "mem_fraction_static": 0.60,
            "dtype": dtype,
            "trust_remote_code": True,
        }

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        requested = int(
            overrides.get("max_running_requests", self.max_running_requests)
        )
        if requested <= 0:
            raise ValueError("VoxCPM2 max_running_requests must be positive")
        self.max_running_requests = requested
        if not bool(overrides.get("disable_cuda_graph", True)):
            overrides["enable_return_hidden_states"] = True

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        if not getattr(server_args, "disable_cuda_graph", False):
            capacity = max(
                self.max_running_requests,
                getattr(server_args, "cuda_graph_max_bs", None) or 1,
                max(getattr(server_args, "cuda_graph_bs", None) or [1]),
            )
            model_worker.model_runner.model.enable_graph_feedback(capacity)

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        from sglang_omni.models.voxcpm2.model_runner import VoxCPM2ModelRunner

        self.model_runner = VoxCPM2ModelRunner(model_worker, output_proc)
        return self.model_runner

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        from sglang_omni.models.voxcpm2.request_builders import (
            apply_voxcpm2_result,
            build_sglang_voxcpm2_request,
        )

        def build_request(payload: Any) -> Any:
            parameter = next(model.parameters())
            return build_sglang_voxcpm2_request(
                payload,
                tokenizer=self.tokenizer,
                patch_size=model.patch_size,
                feat_dim=model.feat_dim,
                vocab_size=int(model.config.vocab_size),
                device=parameter.device,
                dtype=parameter.dtype,
            )

        return build_request, apply_voxcpm2_result

    def make_abort_callback(self) -> Any | None:
        # note (Xinhao Tan): returning None is deliberate, not an omission.
        # The runner keeps no per-request state of its own - a step's patches
        # live on request.data, which the scheduler frees with the request -
        # and the streaming vocoder registers its own cleanup.
        return None

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        from sglang_omni.models.voxcpm2.request_builders import build_stream_output

        return {"stream_output_builder": build_stream_output}


__all__ = ["VoxCPM2EngineBuilder"]
