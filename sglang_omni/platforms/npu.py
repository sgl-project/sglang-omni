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

    def enable_sglang_cuda_graph(self):
        # Ascend NPU: decode CUDA graph capture of some model forward paths
        # (e.g. Qwen3-ASR MLP down_proj) fails inside the ATB matmul operator,
        # so run eager instead of capturing SGLang generation graphs.
        return False

    def enable_torch_compile(self):
        # Ascend NPU has no native torch.compile backend (torch_npu relies on
        # eager + ACL graph capture); avoid torch.compile wrappers on NPU.
        return False
