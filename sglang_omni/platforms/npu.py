from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from sglang.srt.platforms.device_mixin import PlatformEnum

from sglang_omni.platforms.interface import OmniPlatform, RMSNormKernel

if TYPE_CHECKING:
    from sglang_omni.platforms.device_graph import DeviceGraphBackend
else:
    pass


class NPUOmniPlatform(OmniPlatform):
    _enum: PlatformEnum = PlatformEnum.NPU
    device_name: str = "npu"
    device_type: str = "npu"

    def _get_device_graph_backend(self) -> DeviceGraphBackend:
        from sglang_omni.platforms.device_graph import NpuDeviceGraphBackend

        return NpuDeviceGraphBackend()

    def get_device(self, local_rank: int) -> "torch.device":
        return torch.device("npu", local_rank)

    def set_device(self, device: "torch.device") -> None:
        torch.npu.set_device(device)

    def enable_code2wav_graph(self):
        return False

    def get_rmsnorm(self) -> RMSNormKernel:
        import torch_npu

        def npu_rmsnorm(
            hidden_states: torch.Tensor, weight: torch.Tensor, variance_epsilon: float
        ) -> torch.Tensor:
            return torch_npu.npu_rms_norm(hidden_states, weight, variance_epsilon)[0]

        return npu_rmsnorm

    def enable_mimi_decode_graph(self) -> bool:
        """Capture PersonaPlex's Mimi streaming decode in NPU graphs."""
        return True

    def enable_tts_predictor_graph(self) -> bool:
        return True

    def enable_tts_vocoder_fast_path(self) -> bool:
        return False

    def supports_fp8_moe(self) -> bool:
        """No fp8 expert kernel is registered for the SoCs this backend targets,
        so serving MoE in fp8 fails while loading the weights."""
        return False

    def enable_zonos2_torch_compile(self) -> bool:
        """The torch_npu inductor backend cannot compile ZONOS2's graphs, so
        enabling compile would abort startup; the decode NPU graph still carries
        the backbone launches."""
        return False

    def supports_torchaudio_resample(self) -> bool:
        """Disabled as it run on CPU and faced errors during inference for now"""
        return False

    def get_torch_profiler(self) -> TorchProfiler:
        from sglang_omni.profiler.torch_profiler import TorchNPUProfiler

        return TorchNPUProfiler
