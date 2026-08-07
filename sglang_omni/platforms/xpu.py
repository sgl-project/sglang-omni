from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from sglang.srt.platforms.device_mixin import PlatformEnum

from sglang_omni.platforms.interface import OmniPlatform
from sglang_omni.vendor.sglang.server_args import override_server_args

if TYPE_CHECKING:
    from sglang_omni.pipeline.stage_workers import StageLaunchConfig

logger = logging.getLogger(__name__)


class XPUOmniPlatform(OmniPlatform):
    _enum: PlatformEnum = PlatformEnum.XPU
    device_name: str = "xpu"
    device_type: str = "xpu"

    def get_device(self, local_rank: int) -> "torch.device":
        return torch.device("xpu", local_rank)

    def set_device(self, device: "torch.device | int") -> None:
        # torch.xpu wants an index, not a device object. Resolve it up front so a
        # present index 0 is not confused with an absent one.
        index = device.index if isinstance(device, torch.device) else int(device)
        torch.xpu.set_device(0 if index is None else index)

    def apply_model_worker_backend_policy(
        self,
        server_args: ServerArgs,
        model_config: ModelConfig,
        model_arch_override: str | None,
    ) -> str | None:
        effective_quantization = super().apply_model_worker_backend_policy(
            server_args, model_config, model_arch_override
        )

        from sglang_omni.utils.xpu_sglang_compat import (
            patch_available_gpu_memory_for_xpu,
        )

        # Correct XPU free memory before the KV pool is sized against it. Graph
        # capture is left to SGLang, which dispatches XPUGraphRunner by device
        # (model_runner_components/cuda_graph_setup.py).
        patch_available_gpu_memory_for_xpu()

        if model_arch_override in (
            "Qwen3OmniTalker",
            "Qwen3OmniThinkerForCausalLM",
        ) and server_args.moe_runner_backend in (
            "auto",
            "flashinfer_cutlass",
            "cutlass",
        ):
            # SGLang's XPU MoE path asserts the runner is 'triton' (see
            # layers/quantization/unquant.py forward_xpu); 'auto' and the CUTLASS
            # runners fail that assert, so pin triton here.
            override_server_args(
                server_args,
                "sglang-omni-xpu-backend-policy",
                moe_runner_backend="triton",
            )
            logger.info("Selecting 'triton' MoE runner (XPU MoE path requires it)")

        return effective_quantization

    def get_stage_process_env(
        self,
        spec: StageLaunchConfig,
        env: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """No per-rank visibility env: unlike CUDA_VISIBLE_DEVICES with NCCL,
        ZE_AFFINITY_MASK isolation hides peer cards and hangs XCCL discovery, so
        TP ranks keep every card visible and address theirs by gpu_id."""
        del env
        if spec.tp_size <= 1:
            return {}
        if spec.gpu_id is None:
            raise ValueError(f"tp stage {spec.stage_name!r} requires a GPU id")
        return {"SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false"}
