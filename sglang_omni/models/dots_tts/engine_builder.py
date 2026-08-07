# SPDX-License-Identifier: Apache-2.0
"""dots.tts SGLang engine builder."""

from __future__ import annotations

import logging
from typing import Any

from sglang_omni.scheduling.engine_factory import TtsEngineBuilder

logger = logging.getLogger(__name__)


class DotsTTSEngineBuilder(TtsEngineBuilder):
    model_name = "dots.tts"
    context_length = 2048

    def __init__(
        self,
        *,
        optimize: bool = False,
        num_steps: int = 4,
        max_audio_patches: int = 500,
        max_running_requests: int = 16,
    ) -> None:
        from sglang_omni.models.dots_tts.hf_config import DOTS_TTS_MODEL_ARCH_OVERRIDE

        self.model_arch_override = DOTS_TTS_MODEL_ARCH_OVERRIDE
        self.optimize = bool(optimize)
        self.num_steps = int(num_steps)
        self.max_audio_patches = int(max_audio_patches)
        self.max_running_requests = int(max_running_requests)
        if min(self.num_steps, self.max_audio_patches, self.max_running_requests) <= 0:
            raise ValueError("dots.tts batching limits must be positive")
        self._model_runner: Any | None = None

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        del checkpoint_dir
        from sglang_omni.models.dots_tts.hf_config import register_dots_tts_hf_config

        register_dots_tts_hf_config()

    def customize_server_args(self, server_args: Any) -> None:
        # The compiled DiT path only serves max_running_requests=1; the batched
        # tail is eager, so skip the process-global compile policy otherwise.
        # The policy must exist before SGLang builds the model; applying it in
        # setup_model nests Dynamo under FX.
        if self.optimize and int(server_args.max_running_requests) == 1:
            from sglang_omni.models.dots_tts.stages import _configure_optimized_kernels

            _configure_optimized_kernels()

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        return {
            "disable_cuda_graph": True,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_torch_compile": False,
            "max_running_requests": self.max_running_requests,
            "chunked_prefill_size": 0,
            "mem_fraction_static": 0.20,
            "dtype": dtype,
            "trust_remote_code": False,
        }

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        if int(overrides.get("tp_size", 1)) != 1:
            raise ValueError("dots.tts base support does not implement TP")
        requested = int(
            overrides.get("max_running_requests", self.max_running_requests)
        )
        if requested <= 0:
            raise ValueError("dots.tts max_running_requests must be positive")
        self.max_running_requests = requested
        overrides["disable_radix_cache"] = True
        overrides["chunked_prefill_size"] = 0
        if bool(overrides.get("enable_torch_compile", False)):
            raise ValueError(
                "dots.tts uses its DiT compile path; SGLang backbone compile is disabled"
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
        model = model_worker.model_runner.model
        max_running_requests = int(server_args.max_running_requests)
        model.flow.optimize = self.optimize and max_running_requests == 1
        if self.optimize and max_running_requests > 1:
            logger.info(
                "dots.tts optimize with max_running_requests=%d runs the eager "
                "fused batched tail; the compiled DiT path requires "
                "max_running_requests=1",
                max_running_requests,
            )
        if max_running_requests > 1:
            model.flow.init_batched_tail(
                num_slots=max_running_requests,
                nfe=self.num_steps,
                max_audio_patches=self.max_audio_patches,
            )
        model.eval()

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        from sglang_omni.models.dots_tts.model_runner import DotsTTSModelRunner

        self._model_runner = DotsTTSModelRunner(model_worker, output_proc)
        return self._model_runner

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        from sglang_omni.models.dots_tts.request_builders import (
            apply_latent_result,
            build_sglang_dots_tts_request,
        )

        def _build_request(payload: Any) -> Any:
            data = build_sglang_dots_tts_request(payload)
            model.flow.validate_request(
                num_steps=data.state.num_steps,
                ode_method=data.state.ode_method,
                prompt_patch_count=int(data.prompt_span_positions.numel()),
                total_span_count=int(data.span_positions.numel()),
            )
            return data

        return _build_request, apply_latent_result

    def make_abort_callback(self) -> Any | None:
        assert self._model_runner is not None
        return self._model_runner.reset_request

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        from sglang_omni.models.dots_tts.request_builders import build_stream_output

        return {
            "stream_output_builder": build_stream_output,
            "enable_async_decode": False,
        }


__all__ = ["DotsTTSEngineBuilder"]
