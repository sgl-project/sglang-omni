from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.platforms.cuda import CudaDeviceMixin

from sglang_omni.platforms.interface import OmniPlatform
from sglang_omni.quantization import resolve_quant_config
from sglang_omni.utils.misc import model_config_has_moe, normalize_quantization
from sglang_omni.vendor.sglang.server_args import override_server_args

if TYPE_CHECKING:
    from sglang_omni.pipeline.stage_workers import StageLaunchConfig
    from sglang_omni.platforms.device_graph import DeviceGraphBackend
    from sglang_omni.platforms.interface import JointRopeInplaceKernel

logger = logging.getLogger(__name__)


def _is_h20_device() -> bool:
    """True only on NVIDIA H20 (word-boundary match so "H200" isn't caught)."""
    try:
        import re

        import torch

        if not torch.cuda.is_available():
            return False
        return bool(re.search(r"\bH20\b", torch.cuda.get_device_name(0)))
    except Exception:
        return False


def _is_fp8_cutlass_moe_supported() -> bool:
    """Mirror SGLang's CUTLASS FP8 MoE assertions."""
    from sglang.srt.layers.quantization.fp8_utils import cutlass_fp8_supported
    from sglang.srt.utils import (
        is_sm90_supported,
        is_sm100_supported,
        is_sm120_supported,
    )

    return bool(
        cutlass_fp8_supported()
        and (is_sm90_supported() or is_sm100_supported() or is_sm120_supported())
    )


class CUDAOmniPlatform(CudaDeviceMixin, OmniPlatform):
    def _get_device_graph_backend(self) -> DeviceGraphBackend:
        from sglang_omni.platforms.device_graph import CudaDeviceGraphBackend

        return CudaDeviceGraphBackend()

    def get_stage_process_env(
        self,
        spec: StageLaunchConfig,
        env: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        if spec.tp_size <= 1:
            return {}

        source_env = env if env is not None else os.environ
        original_visible = source_env.get("CUDA_VISIBLE_DEVICES")
        if spec.gpu_id is None:
            raise ValueError(f"tp stage {spec.stage_name!r} requires a GPU id")
        if original_visible:
            visible_devices = [item.strip() for item in original_visible.split(",")]
            if spec.gpu_id >= len(visible_devices):
                raise ValueError(
                    f"tp stage {spec.stage_name!r} assigned gpu_id={spec.gpu_id}, "
                    f"but CUDA_VISIBLE_DEVICES only exposes {visible_devices}"
                )
            mapped_gpu = visible_devices[spec.gpu_id]
        else:
            mapped_gpu = str(spec.gpu_id)

        env_updates = {
            "CUDA_VISIBLE_DEVICES": mapped_gpu,
            "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS": "true",
            "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
        }
        # note (ratish): NVLS multicast binding is not available on every host,
        # and NCCL 2.29 fails communicator init instead of falling back. A
        # value from the shell or the stage configuration stands.
        if "NCCL_NVLS_ENABLE" not in source_env and (
            "NCCL_NVLS_ENABLE" not in spec.env_defaults
        ):
            env_updates["NCCL_NVLS_ENABLE"] = "0"
        return env_updates

    def get_intra_node_transport(self) -> TransportKind:
        from sglang_omni.comm.data_ref import TransportKind

        return TransportKind.CUDA_IPC

    def get_fused_qk_norm_rope(self):
        from sgl_kernel import fused_qk_norm_rope

        return fused_qk_norm_rope

    def get_joint_rope_inplace_kernel(self) -> JointRopeInplaceKernel:
        from sglang.kernels.ops.attention.rope import apply_rope_inplace

        return apply_rope_inplace

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
        is_qwen3_omni_arch = model_arch_override in (
            "Qwen3OmniTalker",
            "Qwen3OmniThinkerForCausalLM",
        )
        has_moe = model_config_has_moe(model_config)
        quant_dict = resolve_quant_config(model_config.hf_config)
        has_native_fp8_block_quant = (
            quant_dict is not None
            and normalize_quantization(quant_dict.get("quant_method")) == "fp8"
            and quant_dict.get("weight_block_size") is not None
        )

        if (
            model_arch_override == "Qwen3OmniTalker"
            and effective_quantization is None
            and moe_runner_backend == "auto"
        ):
            # Note:(Chenchen Hong) flashinfer_cutlass MoE deadlocks CUDA-graph
            # capture on H20 (no H20 kernel coverage); triton captures cleanly there.
            moe_runner_backend = "triton" if _is_h20_device() else "flashinfer_cutlass"
            override_server_args(
                server_args,
                "sglang-omni-qwen3-backend-policy",
                moe_runner_backend=moe_runner_backend,
            )

        if (
            is_qwen3_omni_arch
            and effective_quantization == "fp8"
            and has_moe
            and moe_runner_backend == "auto"
            and has_native_fp8_block_quant
            and _is_fp8_cutlass_moe_supported()
        ):
            moe_runner_backend = "cutlass"
            override_server_args(
                server_args,
                "sglang-omni-qwen3-backend-policy",
                moe_runner_backend=moe_runner_backend,
            )

        if (
            is_qwen3_omni_arch
            and effective_quantization == "fp8"
            and has_moe
            and moe_runner_backend == "cutlass"
        ):
            if not has_native_fp8_block_quant:
                raise ValueError(
                    "Qwen3-Omni FP8 CUTLASS MoE requires a native serialized "
                    "block-FP8 checkpoint with weight_block_size."
                )

        if (
            is_qwen3_omni_arch
            and effective_quantization == "fp8"
            and moe_runner_backend == "flashinfer_cutlass"
        ):
            raise ValueError(
                "Qwen3-Omni native FP8 checkpoints cannot use "
                "moe_runner_backend='flashinfer_cutlass'. Leave the backend as "
                "'auto' so Omni selects a native-FP8-compatible MoE runner."
            )

        fp8_gemm_backend = normalize_quantization(cfg.fp8_gemm_runner_backend)
        if (
            is_qwen3_omni_arch
            and effective_quantization == "fp8"
            and has_native_fp8_block_quant
            and fp8_gemm_backend in (None, "auto")
        ):
            # Prefill has request-dependent FP8 dense GEMM shapes outside CUDA
            # graph replay, and DeepGEMM compiles one kernel per shape after
            # readiness. note (ratish): on H100 the thinker measured 3 to 6
            # percent more throughput on Triton at c1 and c16 with equal
            # accuracy over 5000 MMSU prompts, and no post-ready compiles.
            fp8_gemm_backend = "triton"
            override_server_args(
                server_args,
                "sglang-omni-qwen3-backend-policy",
                fp8_gemm_runner_backend=fp8_gemm_backend,
            )

        server_quantization = cfg.quantization
        logger.info(
            f"Configured SGLang backend policy: arch={model_arch_override} "
            f"effective_quantization={effective_quantization} "
            f"server_quantization={server_quantization} "
            f"moe_runner_backend={moe_runner_backend} "
            f"fp8_gemm_backend={fp8_gemm_backend}"
        )
        return effective_quantization
