from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from sglang.srt.platforms.device_mixin import PlatformEnum

from sglang_omni.platforms.interface import OmniPlatform, RMSNormKernel

if TYPE_CHECKING:
    from sglang_omni.platforms.device_graph import DeviceGraphBackend
    from sglang_omni.platforms.interface import JointRopeInplaceKernel
else:
    pass


def joint_rope_inplace(
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_neox: bool,
) -> None:
    """Rotate interleaved Q/K pairs in place with the Ascend fused op."""
    if is_neox:
        raise ValueError(
            "The Ascend joint-RoPE kernel implements interleaved rotation only"
        )
    else:
        pass
    import torch_npu

    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)
    # The fused op applies one cos/sin value per rotated pair.
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(1).to(query.dtype)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(1).to(query.dtype)
    # The fused op reads contiguous BSND input, while packed Q/K are views.
    query_out = torch_npu.npu_rotary_mul(
        query.contiguous().unsqueeze(0),
        cos.unsqueeze(0),
        sin.unsqueeze(0),
        "interleave",
    )
    key_out = torch_npu.npu_rotary_mul(
        key.contiguous().unsqueeze(0),
        cos.unsqueeze(0),
        sin.unsqueeze(0),
        "interleave",
    )
    query.copy_(query_out.squeeze(0))
    key.copy_(key_out.squeeze(0))


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

    def supports_torchaudio_resample(self) -> bool:
        """Disabled as it run on CPU and faced errors during inference for now"""
        return False

    def get_joint_rope_inplace_kernel(self) -> JointRopeInplaceKernel:
        """The Ascend fused rotary op, wrapped to the joint in-place ABI.

        The Ascend operator set exposes interleaved rotation only through the
        single-tensor `npu_rotary_mul`, so the wrapper rotates Q and K with two
        launches.
        """
        return joint_rope_inplace

    def get_torch_profiler(self) -> TorchProfiler:
        from sglang_omni.profiler.torch_profiler import TorchNPUProfiler

        return TorchNPUProfiler
