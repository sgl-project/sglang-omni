# SPDX-License-Identifier: Apache-2.0
"""EasyMagpie SGLang AR engine builder."""

from __future__ import annotations

from typing import Any

from sglang.kernels.ops.mamba.triton_ops import (
    initialize_mamba_selective_state_update_backend,
)
from sglang.srt.environ import envs

from sglang_omni.models.easymagpie_tts.model_runner import EasyMagpieTTSModelRunner
from sglang_omni.models.easymagpie_tts.request_builders import (
    apply_easymagpie_result,
    build_sglang_easymagpie_request,
)
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder

EASYMAGPIE_ARCH = "EasyMagpieTTSForConditionalGeneration"
EASYMAGPIE_CONTEXT_LENGTH = 8192


class EasyMagpieTTSEngineBuilder(TtsEngineBuilder):
    model_name = "EasyMagpie-TTS"
    context_length = EASYMAGPIE_CONTEXT_LENGTH
    model_arch_override = EASYMAGPIE_ARCH

    def __init__(
        self, *, max_running_requests: int = 8, mem_fraction_static: float = 0.72
    ) -> None:
        self.max_running_requests = max_running_requests
        self.mem_fraction_static = mem_fraction_static

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        del checkpoint_dir
        # note (Yashwant Hayaran): the Mamba conv cache dtype is only
        # configurable through this SGLang env and defaults to bfloat16; a
        # mismatch with the model dtype breaks the causal-conv kernels.
        envs.SGLANG_MAMBA_CONV_DTYPE.set(self.dtype)

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        return {
            "dtype": dtype,
            "max_running_requests": self.max_running_requests,
            "max_total_tokens": self.max_running_requests * EASYMAGPIE_CONTEXT_LENGTH,
            "mem_fraction_static": self.mem_fraction_static,
            # Per-step phoneme and acoustic feedback lives on request data and
            # has no rollback, so a non-final prefill chunk would corrupt it.
            "chunked_prefill_size": 0,
            "disable_cuda_graph": True,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_torch_compile": False,
            "sampling_backend": "pytorch",
            "trust_remote_code": False,
        }

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        if int(overrides.get("tp_size", 1)) != 1:
            raise ValueError("EasyMagpie TTS supports tp_size=1 only")
        else:
            pass

    def customize_server_args(self, server_args: Any) -> None:
        initialize_mamba_selective_state_update_backend(server_args)

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        del checkpoint_dir, device, gpu_id, server_args
        model_worker.model_runner.model.eval()

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        return EasyMagpieTTSModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        del model
        return build_sglang_easymagpie_request, apply_easymagpie_result


__all__ = ["EASYMAGPIE_ARCH", "EasyMagpieTTSEngineBuilder"]
