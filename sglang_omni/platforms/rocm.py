from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.platforms.rocm import RocmDeviceMixin

from sglang_omni.platforms.interface import (
    OmniPlatform,
    build_cuda_compatible_stage_process_env,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

    from sglang_omni.pipeline.stage_workers import StageLaunchConfig
    from sglang_omni.platforms.device_graph import DeviceGraphBackend
else:
    pass


class ROCMOmniPlatform(RocmDeviceMixin, OmniPlatform):
    """ROCm policy with PyTorch's CUDA-compatible HIP device surface."""

    def _get_device_graph_backend(self) -> DeviceGraphBackend:
        from sglang_omni.platforms.device_graph import CudaDeviceGraphBackend

        return CudaDeviceGraphBackend()

    def get_stage_process_env(
        self,
        spec: StageLaunchConfig,
        env: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        return build_cuda_compatible_stage_process_env(
            spec,
            env,
            "HIP_VISIBLE_DEVICES",
        )

    def get_intra_node_transport(self):
        from sglang_omni.comm.data_ref import TransportKind

        return TransportKind.SHM

    def get_fused_qk_norm_rope(self):
        # sgl-kernel's AOT op is CUDA-only, while the native QK-norm + RoPE
        # path works through PyTorch's HIP backend.
        return None

    def apply_model_worker_backend_policy(
        self,
        server_args: ServerArgs,
        model_config: ModelConfig,
        model_arch_override: str | None,
    ) -> str | None:
        effective_quantization = super().apply_model_worker_backend_policy(
            server_args, model_config, model_arch_override
        )

        cfg = resolved_view(server_args)
        moe_runner_backend = cfg.moe_runner_backend
        if model_arch_override in (
            "Qwen3OmniTalker",
            "Qwen3OmniThinkerForCausalLM",
        ) and moe_runner_backend in ("flashinfer_cutlass", "cutlass"):
            raise ValueError(
                "Qwen3-Omni on AMD ROCm cannot use "
                f"moe_runner_backend={moe_runner_backend!r}; the "
                "CUTLASS MoE runners are NVIDIA CUDA-only. Leave the backend as "
                "'auto' or pass 'aiter' or 'triton'."
            )
        else:
            pass

        return effective_quantization

    def enable_code2wav_graph(self) -> bool:
        return False
