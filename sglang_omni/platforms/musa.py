# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.platforms.device_mixin import PlatformEnum

from sglang_omni.platforms.cuda import CUDAOmniPlatform

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

try:
    import torchada  # noqa: F401
except ImportError as exc:
    logger.warning(
        f"Failed to import torchada: {exc}. MUSA platform compatibility will not work."
    )


class MUSAOmniPlatform(CUDAOmniPlatform):
    _enum = PlatformEnum.MUSA
    device_name = "musa"
    device_type = "musa"

    def get_device(self, local_rank: int) -> "torch.device":
        import torch

        return torch.device("musa", local_rank)

    def set_device(self, device: "torch.device | int") -> None:
        import torch

        index = device.index if isinstance(device, torch.device) else int(device)
        torch.musa.set_device(0 if index is None else index)

    def get_device_name(self, device_id: int = 0) -> str:
        import torch

        return torch.musa.get_device_name(device_id)

    def get_device_total_memory(self, device_id: int = 0) -> int:
        import torch

        return torch.musa.get_device_properties(device_id).total_memory

    def get_current_memory_usage(self, device: "torch.device | None" = None) -> float:
        import torch

        if device is None:
            device = torch.device("musa", torch.musa.current_device())
        return float(torch.musa.memory_allocated(device))

    def get_fused_qk_norm_rope(self):
        # sgl-kernel's AOT fused_qk_norm_rope op is CUDA-only today.
        # Use the native QK-norm + RoPE path on MUSA.
        return None

    def apply_model_worker_backend_policy(
        self,
        server_args: ServerArgs,
        model_config: ModelConfig,
        model_arch_override: str | None,
    ) -> str | None:
        return super().apply_model_worker_backend_policy(
            server_args, model_config, model_arch_override
        )
