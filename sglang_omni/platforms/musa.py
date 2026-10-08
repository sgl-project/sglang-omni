# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from sglang.srt.platforms.device_mixin import PlatformEnum

from sglang_omni.platforms.cuda import CUDAOmniPlatform
from sglang_omni.platforms.interface import JointRopeInplaceKernel, OmniPlatform


class MUSAOmniPlatform(CUDAOmniPlatform):
    _enum = PlatformEnum.MUSA
    device_name = "musa"
    device_type = "musa"

    def get_fused_qk_norm_rope(self):
        # sgl-kernel's AOT fused_qk_norm_rope op is CUDA-only today.
        # Use the native QK-norm + RoPE path on MUSA.
        return None

    def get_joint_rope_inplace_kernel(self) -> JointRopeInplaceKernel:
        try:
            import torchada  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "MUSA joint RoPE requires torchada with SGLang JIT support"
            ) from exc
        return super().get_joint_rope_inplace_kernel()

    def enable_codec_decode_graph(self) -> bool:
        return False

    def apply_model_worker_backend_policy(
        self,
        server_args: ServerArgs,
        model_config: ModelConfig,
        model_arch_override: str | None,
    ) -> str | None:
        return OmniPlatform.apply_model_worker_backend_policy(
            self, server_args, model_config, model_arch_override
        )
