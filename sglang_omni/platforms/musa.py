# SPDX-License-Identifier: Apache-2.0
"""MUSA platform hooks for SGLang Omni."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from sglang_omni.platforms.interface import OmniPlatform
from sglang_omni.utils.misc import normalize_quantization
from sglang_omni.vendor.sglang.server_args import override_server_args

if TYPE_CHECKING:
    from sglang_omni.pipeline.stage_workers import StageLaunchConfig


class MUSAOmniPlatform(OmniPlatform):
    device_name = "musa"
    device_type = "musa"

    def is_musa(self) -> bool:
        try:
            import torchada  # noqa: F401
        except ImportError:
            return False
        return hasattr(torch.version, "musa") and torch.version.musa is not None

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
