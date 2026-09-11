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
        self._model_runner: Any | None = None
        self._tokenizer: Any | None = None

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        from transformers import AutoTokenizer

        from sglang_omni.models.voxcpm2.hf_config import register_voxcpm2_hf_config

        register_voxcpm2_hf_config()
        self._tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_dir, trust_remote_code=True
        )

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        del checkpoint_dir, device, gpu_id
        if not getattr(server_args, "disable_cuda_graph", False):
            model_worker.model_runner.model.enable_graph_feedback(
                self.max_running_requests
            )

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        from sglang_omni.models.voxcpm2.model_runner import VoxCPM2ModelRunner

        self._model_runner = VoxCPM2ModelRunner(model_worker, output_proc)
        return self._model_runner

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        from sglang_omni.models.voxcpm2.request_builders import (
            apply_voxcpm2_result,
            build_sglang_voxcpm2_request,
        )

        def _build_request(payload: Any) -> Any:
            return build_sglang_voxcpm2_request(
                payload,
                tokenizer=self._tokenizer,
                patch_size=model.patch_size,
                feat_dim=model.feat_dim,
                vocab_size=int(model.config.vocab_size),
            )

        return _build_request, apply_voxcpm2_result

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
