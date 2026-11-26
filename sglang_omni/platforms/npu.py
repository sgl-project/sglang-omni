from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from sglang.srt.platforms.device_mixin import PlatformEnum

from sglang_omni.platforms.interface import (
    JointRopeInplaceKernel,
    OmniPlatform,
    OverlapAddImpl,
    RMSNormAddKernel,
    RMSNormKernel,
)

if TYPE_CHECKING:
    from sglang_omni.platforms.device_graph import DeviceGraphBackend
else:
    pass


def validate_npu_graph_environment(model_name: str) -> None:
    """Reject Host queue settings that would stall NPU graph streams."""
    if (
        os.environ.get("PER_STREAM_QUEUE") != "1"
        or os.environ.get("TASK_QUEUE_ENABLE", "1") != "1"
        or os.environ.get("ASCEND_LAUNCH_BLOCKING") == "1"
    ):
        raise ValueError(
            f"{model_name} NPU graphs require PER_STREAM_QUEUE=1, "
            "TASK_QUEUE_ENABLE unset or 1, and ASCEND_LAUNCH_BLOCKING unset or 0 "
            "before starting the server"
        )
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
    """Rotate half-split or interleaved Q/K pairs in place with the Ascend op."""
    import torch_npu

    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)
    if is_neox:
        # A half-split pair is (i, i + D / 2), so one coefficient covers a half.
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
        rotary_mode = "half"
    else:
        # An interleaved pair is (2i, 2i + 1), so one coefficient covers a pair.
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
        rotary_mode = "interleave"
    cos = cos.unsqueeze(1).unsqueeze(0).to(query.dtype)
    sin = sin.unsqueeze(1).unsqueeze(0).to(query.dtype)
    # The fused op reads contiguous BSND input, while packed Q/K are views.
    for tensor in (query, key):
        rotated = torch_npu.npu_rotary_mul(
            tensor.contiguous().unsqueeze(0), cos, sin, rotary_mode
        )
        tensor.copy_(rotated.squeeze(0))


def overlap_add_by_shifts(frames: torch.Tensor, *, hop_length: int) -> torch.Tensor:
    """Rebuild the overlap-add from stride-aligned slice sums.

    `torch.nn.functional.fold` faults the Ascend vector core once its kernel
    reaches 3071 samples, which Ming's n_fft=3528 exceeds. Splitting the window
    into hop-sized taps and adding each tap at its own offset matches fold and
    keeps every step on the vector core's elementwise path.
    """
    batch, win_length, frame_count = frames.shape
    taps = -(-win_length // hop_length)
    padded = taps * hop_length
    span = frame_count * hop_length
    if padded != win_length:
        tail = frames.new_zeros(batch, padded - win_length, frame_count)
        frames = torch.cat((frames, tail), dim=1)
    else:
        pass
    blocks = frames.reshape(batch, taps, hop_length, frame_count)
    merged = frames.new_zeros(batch, (frame_count - 1) * hop_length + padded)
    for tap in range(taps):
        start = tap * hop_length
        merged[:, start : start + span] += (
            blocks[:, tap].transpose(1, 2).reshape(batch, span)
        )
    # Zero padding contributes nothing, so the folded tail is dropped here.
    return merged[:, : (frame_count - 1) * hop_length + win_length]


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

    @classmethod
    def is_float64_supported(cls) -> bool:
        return False

    def enable_code2wav_graph(self):
        return False

    def get_rmsnorm(self) -> RMSNormKernel:
        import torch_npu

        def npu_rmsnorm(
            hidden_states: torch.Tensor, weight: torch.Tensor, variance_epsilon: float
        ) -> torch.Tensor:
            return torch_npu.npu_rms_norm(hidden_states, weight, variance_epsilon)[0]

        return npu_rmsnorm

    def get_add_rmsnorm(self) -> RMSNormAddKernel:
        import torch_npu

        def npu_add_rmsnorm(
            hidden_states: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
            variance_epsilon: float,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            normalized, _, residual_sum = torch_npu.npu_add_rms_norm(
                hidden_states, residual, weight, variance_epsilon
            )
            return normalized, residual_sum

        return npu_add_rmsnorm

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

    def get_overlap_add_impl(self) -> OverlapAddImpl:
        """Slice-summed overlap-add, used because fold faults on large windows."""
        return overlap_add_by_shifts

    def get_joint_rope_inplace_kernel(self) -> JointRopeInplaceKernel:
        """The Ascend fused rotary op, wrapped to the joint in-place ABI.

        The single-tensor `npu_rotary_mul` covers both rotation layouts, so the
        wrapper rotates Q and K with two launches.
        """
        return joint_rope_inplace

    def get_torch_profiler(self) -> TorchProfiler:
        from sglang_omni.profiler.torch_profiler import TorchNPUProfiler

        return TorchNPUProfiler
