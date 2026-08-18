# SPDX-License-Identifier: Apache-2.0
"""MUSA platform hooks for SGLang Omni."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from sglang.srt.platforms.device_mixin import PlatformEnum

from sglang_omni.platforms.interface import OmniPlatform
from sglang_omni.utils.misc import normalize_quantization
from sglang_omni.vendor.sglang.server_args import override_server_args

if TYPE_CHECKING:
    from sglang_omni.pipeline.stage_workers import StageLaunchConfig


class MUSAOmniPlatform(OmniPlatform):
    _enum = PlatformEnum.MUSA
    device_name = "musa"
    device_type = "musa"

    def get_device(self, local_rank: int) -> torch.device:
        return torch.device("musa", local_rank)

    def set_device(self, device: torch.device | int) -> None:
        torch.musa.set_device(device)

    def get_device_name(self, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return int(torch.cuda.get_device_properties(device_id).total_memory)

    def get_current_memory_usage(self, device: torch.device | None = None) -> float:
        torch.cuda.reset_peak_memory_stats(device)
        return float(torch.cuda.max_memory_allocated(device))

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        return tuple(int(v) for v in torch.cuda.mem_get_info(device_id))

    def get_torch_distributed_backend_str(self) -> str:
        return "mccl"

    def get_stage_process_env(
        self,
        spec: StageLaunchConfig,
        env: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        if spec.tp_size <= 1:
            return {}

        source_env = env if env is not None else os.environ
        original_visible = source_env.get("MUSA_VISIBLE_DEVICES")
        if spec.gpu_id is None:
            raise ValueError(f"tp stage {spec.stage_name!r} requires a GPU id")
        if original_visible:
            visible_devices = [item.strip() for item in original_visible.split(",")]
            if spec.gpu_id >= len(visible_devices):
                raise ValueError(
                    f"tp stage {spec.stage_name!r} assigned gpu_id={spec.gpu_id}, "
                    f"but MUSA_VISIBLE_DEVICES only exposes {visible_devices}"
                )
            mapped_gpu = visible_devices[spec.gpu_id]
        else:
            mapped_gpu = str(spec.gpu_id)

        return {
            "MUSA_VISIBLE_DEVICES": mapped_gpu,
            "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS": "true",
            "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
        }

    def get_intra_node_transport(self) -> TransportKind:
        from sglang_omni.comm.data_ref import TransportKind

        return TransportKind.CUDA_IPC

    def apply_model_worker_backend_policy(
        self,
        server_args: ServerArgs,
        model_config: ModelConfig,
        model_arch_override: str | None,
    ) -> str | None:
        effective_quantization = super().apply_model_worker_backend_policy(
            server_args, model_config, model_arch_override
        )

        if server_args.moe_runner_backend == "auto":
            override_server_args(
                server_args,
                "sglang-omni-musa-backend-policy",
                moe_runner_backend="triton",
            )

        if normalize_quantization(server_args.fp8_gemm_runner_backend) in (
            None,
            "auto",
        ):
            override_server_args(
                server_args,
                "sglang-omni-musa-backend-policy",
                fp8_gemm_runner_backend="deep_gemm",
            )

        return effective_quantization
