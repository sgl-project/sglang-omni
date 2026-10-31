from __future__ import annotations

from sglang.srt.arg_groups.model_override_base import (
    attention_backends_of,
    resolved_view,
)
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.platforms.cpu import CpuDeviceMixin
from sglang.srt.server_args import ServerArgs

from sglang_omni.platforms.interface import OmniPlatform


class CPUOmniPlatform(CpuDeviceMixin, OmniPlatform):
    def enable_code2wav_graph(self) -> bool:
        return False

    def get_encoder_decoder_attention_backend(self) -> str | None:
        return "intel_amx"

    def apply_model_worker_backend_policy(
        self,
        server_args: ServerArgs,
        model_config: ModelConfig,
        model_arch_override: str | None,
    ) -> str | None:
        effective_quantization = super().apply_model_worker_backend_policy(
            server_args, model_config, model_arch_override
        )
        if model_arch_override == "WhisperForConditionalGeneration" and any(
            backend not in ("intel_amx", "torch_native")
            for backend in attention_backends_of(resolved_view(server_args))
        ):
            raise ValueError(
                "Whisper ASR on CPU supports intel_amx or torch_native for both "
                "prefill and decode attention. Remove unsupported attention "
                "backend overrides."
            )
        else:
            pass
        return effective_quantization
