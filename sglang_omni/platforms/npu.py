from __future__ import annotations

import torch
from sglang.srt.platforms.device_mixin import PlatformEnum

from sglang_omni.platforms.interface import OmniPlatform


class NPUOmniPlatform(OmniPlatform):
    _enum: PlatformEnum = PlatformEnum.NPU
    device_name: str = "npu"
    device_type: str = "npu"

    def get_device(self, local_rank: int) -> "torch.device":
        return torch.device("npu", local_rank)

    def set_device(self, device: "torch.device") -> None:
        torch.npu.set_device(device)

    def enable_code2wav_graph(self):
        return False

    def supports_torchaudio_resample(self) -> bool:
        """Disabled as it run on CPU and faced errors during inference for now"""
        return False
