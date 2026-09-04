from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

from sglang.srt.platforms.rocm import RocmDeviceMixin

from sglang_omni.platforms.interface import OmniPlatform

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

    from sglang_omni.pipeline.stage_workers import StageLaunchConfig


class ROCMOmniPlatform(RocmDeviceMixin, OmniPlatform):
    """ROCm policy with PyTorch's CUDA-compatible HIP device surface."""

    def get_stage_process_env(
        self,
        spec: StageLaunchConfig,
        env: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        if spec.tp_size <= 1:
            return {}

        source_env = env if env is not None else os.environ
        hip_visible = source_env.get("HIP_VISIBLE_DEVICES")
        visibility_var = (
            "HIP_VISIBLE_DEVICES" if hip_visible else "CUDA_VISIBLE_DEVICES"
        )
        original_visible = source_env.get(visibility_var)
        if spec.gpu_id is None:
            raise ValueError(f"tp stage {spec.stage_name!r} requires a GPU id")
        if original_visible:
            visible_devices = [item.strip() for item in original_visible.split(",")]
            if spec.gpu_id >= len(visible_devices):
                raise ValueError(
                    f"tp stage {spec.stage_name!r} assigned gpu_id={spec.gpu_id}, "
                    f"but {visibility_var} only exposes {visible_devices}"
                )
            mapped_gpu = visible_devices[spec.gpu_id]
        else:
            mapped_gpu = str(spec.gpu_id)

        env_updates = {
            "CUDA_VISIBLE_DEVICES": mapped_gpu,
            "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS": "true",
            "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
        }
        if hip_visible:
            # Note (zijiecode): the HIP runtime prefers HIP_VISIBLE_DEVICES over the
            # CUDA alias, so the child must be narrowed through it; the alias
            # is kept for the startup lock and SGLang's physical-device helpers.
            env_updates["HIP_VISIBLE_DEVICES"] = mapped_gpu
        return env_updates

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

        if model_arch_override in (
            "Qwen3OmniTalker",
            "Qwen3OmniThinkerForCausalLM",
        ) and server_args.moe_runner_backend in ("flashinfer_cutlass", "cutlass"):
            raise ValueError(
                "Qwen3-Omni on AMD ROCm cannot use "
                f"moe_runner_backend={server_args.moe_runner_backend!r}; the "
                "CUTLASS MoE runners are NVIDIA CUDA-only. Leave the backend as "
                "'auto' or pass 'aiter' or 'triton'."
            )

        return effective_quantization

    def enable_code2wav_graph(self) -> bool:
        return False
