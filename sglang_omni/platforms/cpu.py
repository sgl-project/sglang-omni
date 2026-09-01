from __future__ import annotations

import torch
from sglang.srt.platforms.device_mixin import PlatformEnum

try:
    from sglang.srt.platforms.cpu import CpuDeviceMixin
except ModuleNotFoundError:
    from sglang.srt.platforms.interface import SRTPlatform

    class CpuDeviceMixin(SRTPlatform):
        _enum = PlatformEnum.CPU
        device_name = "cpu"
        device_type = "cpu"

        def get_device(self, local_rank: int) -> torch.device:
            return torch.device("cpu")

        def get_device_name(self, device_id: int = 0) -> str:
            return "cpu"

        def get_device_total_memory(self, device_id: int = 0) -> int:
            return 0

        def get_current_memory_usage(
            self, device: torch.device | None = None
        ) -> float:
            return 0.0

from sglang_omni.platforms.interface import OmniPlatform


class CPUOmniPlatform(CpuDeviceMixin, OmniPlatform):
    def enable_code2wav_graph(self):
        return False
