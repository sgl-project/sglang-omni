from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

from sglang.srt.platforms.cuda import CudaDeviceMixin
from sglang.srt.platforms.rocm import RocmDeviceMixin

from sglang_omni.platforms.interface import OmniPlatform
from sglang_omni.quantization import resolve_quant_config
from sglang_omni.utils.misc import model_config_has_moe, normalize_quantization
from sglang_omni.vendor.sglang.server_args import override_server_args

if TYPE_CHECKING:
    from sglang_omni.pipeline.stage_workers import StageLaunchConfig

logger = logging.getLogger(__name__)

_ROCM_VISIBLE_DEVICE_VARIABLES = (
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)


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
    """Mirror SGLang 0.5.16's CUTLASS FP8 MoE assertions."""
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


def _split_visible_devices(value: str) -> list[str]:
    devices = [item.strip() for item in value.split(",")]
    if not devices or any(not item for item in devices):
        raise ValueError(f"invalid visible-device list {value!r}")
    return devices


def _map_rocm_visible_device(
    *, spec: StageLaunchConfig, source_env: Mapping[str, str]
) -> str:
    """Map a logical rank without applying HIP visibility aliases twice.

    ROCr applies its physical mask before HIP/CUDA aliases. Consequently,
    ``ROCR_VISIBLE_DEVICES=3,4`` must pair with aliases ``0,1`` rather than
    ``3,4``; repeating the physical ids can hide every device except rank 0.
    """

    configured = {
        name: _split_visible_devices(value)
        for name in _ROCM_VISIBLE_DEVICE_VARIABLES
        if (value := (source_env.get(name) or "").strip())
    }
    if not configured:
        return str(spec.gpu_id)

    canonical_name = next(
        name for name in _ROCM_VISIBLE_DEVICE_VARIABLES if name in configured
    )
    visible_devices = configured[canonical_name]
    logical_devices = [str(index) for index in range(len(visible_devices))]
    for name, devices in configured.items():
        if devices not in (visible_devices, logical_devices):
            raise ValueError(
                f"tp stage {spec.stage_name!r} has conflicting accelerator "
                f"visibility: {canonical_name}={','.join(visible_devices)!r}, "
                f"{name}={','.join(devices)!r}"
            )
    if spec.gpu_id >= len(visible_devices):
        raise ValueError(
            f"tp stage {spec.stage_name!r} assigned gpu_id={spec.gpu_id}, "
            f"but {canonical_name} only exposes {visible_devices}"
        )
    return visible_devices[spec.gpu_id]


class CUDAOmniPlatform(CudaDeviceMixin, OmniPlatform):
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

        return {
            "CUDA_VISIBLE_DEVICES": mapped_gpu,
            "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS": "true",
            "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
        }

    def get_intra_node_transport(self) -> TransportKind:
        from sglang_omni.comm.data_ref import TransportKind

        return TransportKind.CUDA_IPC

    def supports_device_graphs(self) -> bool:
        return True

    def get_fused_qk_norm_rope(self):
        from sgl_kernel import fused_qk_norm_rope

        return fused_qk_norm_rope

    def apply_model_worker_backend_policy(
        self,
        server_args: ServerArgs,
        model_config: ModelConfig,
        model_arch_override: str | None,
    ) -> str | None:

        effective_quantization = super().apply_model_worker_backend_policy(
            server_args, model_config, model_arch_override
        )

        moe_runner_backend = server_args.moe_runner_backend
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
            override_server_args(
                server_args,
                "sglang-omni-qwen3-backend-policy",
                moe_runner_backend=(
                    "triton" if _is_h20_device() else "flashinfer_cutlass"
                ),
            )
            moe_runner_backend = server_args.moe_runner_backend

        if (
            is_qwen3_omni_arch
            and effective_quantization == "fp8"
            and has_moe
            and moe_runner_backend == "auto"
            and has_native_fp8_block_quant
            and _is_fp8_cutlass_moe_supported()
        ):
            override_server_args(
                server_args,
                "sglang-omni-qwen3-backend-policy",
                moe_runner_backend="cutlass",
            )
            moe_runner_backend = server_args.moe_runner_backend

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

        fp8_gemm_backend = normalize_quantization(server_args.fp8_gemm_runner_backend)
        if (
            model_arch_override == "Qwen3OmniTalker"
            and effective_quantization == "fp8"
            and has_native_fp8_block_quant
            and fp8_gemm_backend in (None, "auto")
        ):
            # Projected talker prefill has request-dependent FP8 dense GEMM shapes
            # outside decode CUDA graph replay; DeepGEMM can otherwise JIT there.
            override_server_args(
                server_args,
                "sglang-omni-qwen3-backend-policy",
                fp8_gemm_runner_backend="triton",
            )
            fp8_gemm_backend = server_args.fp8_gemm_runner_backend

        server_quantization = server_args.quantization
        logger.info(
            f"Configured SGLang backend policy: arch={model_arch_override} "
            f"effective_quantization={effective_quantization} "
            f"server_quantization={server_quantization} "
            f"moe_runner_backend={moe_runner_backend} "
            f"fp8_gemm_backend={fp8_gemm_backend}"
        )
        return effective_quantization


class ROCMOmniPlatform(RocmDeviceMixin, OmniPlatform):
    """ROCm policy kept separate from NVIDIA CUDA backend selection."""

    def get_stage_process_env(
        self,
        spec: StageLaunchConfig,
        env: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        if spec.tp_size <= 1:
            return {}

        source_env = env if env is not None else os.environ
        mapped_gpu = _map_rocm_visible_device(spec=spec, source_env=source_env)
        return {
            # ROCr uses physical ids; HIP and CUDA aliases operate on the
            # resulting logical namespace. A rank-local process therefore
            # sees physical ``mapped_gpu`` as logical device zero.
            "ROCR_VISIBLE_DEVICES": mapped_gpu,
            "HIP_VISIBLE_DEVICES": "0",
            "CUDA_VISIBLE_DEVICES": "0",
            "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS": "true",
            "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
        }

    def get_intra_node_transport(self) -> TransportKind:
        from sglang_omni.comm.data_ref import TransportKind

        # PyTorch exposes CUDA-compatible storage and event IPC APIs on ROCm.
        return TransportKind.CUDA_IPC

    def get_default_remote_transport(self) -> TransportKind:
        from sglang_omni.comm.data_ref import TransportKind

        return TransportKind.NIXL

    def enable_code2wav_graph(self):
        # The owned code2wav runner directly constructs CUDA graph objects. Keep
        # ROCm eager until that path is qualified independently on both targets.
        return False

    def supports_device_graphs(self) -> bool:
        # Individual SGLang graph paths may work through torch.cuda on HIP, but
        # Omni does not advertise the primitive until its owned graph runners pass.
        return False

    def apply_model_worker_backend_policy(
        self,
        server_args: ServerArgs,
        model_config: ModelConfig,
        model_arch_override: str | None,
    ) -> str | None:
        effective_quantization = super().apply_model_worker_backend_policy(
            server_args, model_config, model_arch_override
        )

        moe_backend = str(server_args.moe_runner_backend or "").strip().lower()
        if moe_backend in ("cutlass", "flashinfer_cutlass"):
            raise ValueError(
                f"{model_arch_override or 'model'} on ROCm cannot use "
                f"moe_runner_backend={server_args.moe_runner_backend!r}; CUTLASS "
                "and FlashInfer runners are NVIDIA-only. Use 'auto', 'aiter', "
                "or 'triton'."
            )

        fp8_backend = str(server_args.fp8_gemm_runner_backend or "").strip().lower()
        if fp8_backend in ("cutlass", "deep_gemm"):
            raise ValueError(
                f"{model_arch_override or 'model'} on ROCm cannot use "
                f"fp8_gemm_runner_backend={server_args.fp8_gemm_runner_backend!r}; "
                "use 'auto', 'aiter', or 'triton'."
            )

        logger.info(
            "Configured ROCm backend policy: arch=%s effective_quantization=%s "
            "moe_runner_backend=%s fp8_gemm_backend=%s",
            model_arch_override,
            effective_quantization,
            server_args.moe_runner_backend,
            server_args.fp8_gemm_runner_backend,
        )
        return effective_quantization
