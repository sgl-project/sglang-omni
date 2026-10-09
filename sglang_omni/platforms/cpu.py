from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.platforms.cpu import CpuDeviceMixin

from sglang_omni.platforms.interface import OmniPlatform

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs
else:
    pass


class CPUOmniPlatform(CpuDeviceMixin, OmniPlatform):
    def enable_code2wav_graph(self):
        return False

    def cross_attention_backend(self) -> str | None:
        return "torch_native"

    def apply_model_worker_backend_policy(
        self,
        server_args: ServerArgs,
        model_config: ModelConfig,
        model_arch_override: str | None,
    ) -> str | None:
        effective_quantization = super().apply_model_worker_backend_policy(
            server_args, model_config, model_arch_override
        )
        if (
            model_arch_override == "WhisperForConditionalGeneration"
            and server_args.attention_backend != self.cross_attention_backend()
        ):
            raise ValueError(
                "Whisper ASR on CPU requires attention_backend='torch_native'. "
                "Drop the override or set it to 'torch_native'."
            )
        else:
            pass
        return effective_quantization
