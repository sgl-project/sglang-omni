from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from sglang.srt.platforms.device_mixin import PlatformEnum

from sglang_omni.platforms.interface import OmniPlatform

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

    from sglang_omni.pipeline.stage_workers import StageLaunchConfig


class XPUOmniPlatform(OmniPlatform):
    _enum: PlatformEnum = PlatformEnum.XPU
    device_name: str = "xpu"
    device_type: str = "xpu"

    def get_device(self, local_rank: int) -> "torch.device":
        return torch.device("xpu", local_rank)

    def set_device(self, device: "torch.device | int") -> None:
        index = device.index if isinstance(device, torch.device) else int(device)
        torch.xpu.set_device(0 if index is None else index)

    def enable_code2wav_graph(self):
        return False

    def get_fused_qk_norm_rope_with_cos_sin_cache(self):
        try:
            from sgl_kernel import fused_inplace_qknorm_rope
        except ImportError as exc:
            logger.info(
                f"XPU sgl_kernel has no cos/sin-cache fused QK-norm-RoPE kernel "
                f"({exc}); falling back to the unfused QK-norm and RoPE path"
            )
            return None
        return fused_inplace_qknorm_rope

    def enable_talker_graph(self) -> bool:
        # The predictor's default SDPA dispatch is not capturable here.
        return False

    def enable_thinker_decode_graph(self) -> bool:
        # Capture leaves the scheduler thread's stream recording; host reads fail.
        return False

    def get_decode_cuda_graph_backend(self) -> str | None:
        # SGLang leaves XPU decode capture opt-in and accepts only full.
        from sglang.srt.model_executor.cuda_graph_config import Backend

        return Backend.FULL

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
                f"Qwen3-Omni on Intel XPU cannot use "
                f"moe_runner_backend={server_args.moe_runner_backend!r}; the CUTLASS "
                "MoE runners are CUDA-only. Leave the backend as 'auto' or pass "
                "'triton'."
            )

        return effective_quantization

    def get_stage_process_env(
        self,
        spec: StageLaunchConfig,
        env: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Keep every card visible, preserving a group-wide ZE_AFFINITY_MASK."""
        if spec.tp_size <= 1:
            return {}
        if spec.gpu_id is None:
            raise ValueError(f"tp stage {spec.stage_name!r} requires a GPU id")

        updates = {"SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false"}
        source_env = env if env is not None else os.environ
        mask = (source_env.get("ZE_AFFINITY_MASK") or "").strip()
        if not mask:
            return updates

        visible = [item.strip() for item in mask.split(",") if item.strip()]
        if len(visible) < spec.tp_size:
            raise ValueError(
                f"tp stage {spec.stage_name!r} needs tp_size={spec.tp_size} cards, but "
                f"ZE_AFFINITY_MASK={mask!r} exposes {len(visible)}. Widen the mask to "
                "cover the whole TP group: every rank must see its peers for XCCL "
                "discovery, and dropping the mask instead would relocate the stage "
                "onto different physical cards."
            )
        if spec.gpu_id >= len(visible):
            raise ValueError(
                f"tp stage {spec.stage_name!r} assigned gpu_id={spec.gpu_id}, but "
                f"ZE_AFFINITY_MASK={mask!r} exposes only {len(visible)} cards "
                f"({', '.join(visible)}). gpu_id indexes into the mask, not the host."
            )
        return updates
