from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

from sglang.srt.platforms.rocm import RocmDeviceMixin

from sglang_omni.platforms.interface import OmniPlatform

if TYPE_CHECKING:
    from sglang_omni.comm.data_ref import TransportKind
    from sglang_omni.pipeline.stage_workers import StageLaunchConfig


_VISIBLE_DEVICE_VARIABLES = (
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)


def _parse_visible_devices(name: str, value: str) -> list[str]:
    devices = [item.strip() for item in value.split(",")]
    if not devices or any(not item for item in devices):
        raise ValueError(f"invalid {name} value {value!r}")
    return devices


def _resolve_physical_visible_devices(source_env: Mapping[str, str]) -> list[str]:
    configured = {
        name: _parse_visible_devices(name, value)
        for name in _VISIBLE_DEVICE_VARIABLES
        if (value := (source_env.get(name) or "").strip())
    }
    if not configured:
        return []

    if "ROCR_VISIBLE_DEVICES" in configured:
        physical = configured["ROCR_VISIBLE_DEVICES"]
    else:
        first_name = next(
            name for name in _VISIBLE_DEVICE_VARIABLES if name in configured
        )
        physical = configured[first_name]

    logical = [str(index) for index in range(len(physical))]
    for name, devices in configured.items():
        if devices not in (physical, logical):
            raise ValueError(
                "conflicting ROCm visibility namespaces: "
                f"ROCR/HIP/CUDA masks resolve to {physical!r}, but {name}={devices!r}"
            )
    return physical


class ROCMOmniPlatform(RocmDeviceMixin, OmniPlatform):
    """AMD ROCm implementation of Omni's shared accelerator contract.

    ROCm intentionally does not inherit NVIDIA CUDA policy. PyTorch exposes HIP
    devices through the CUDA-shaped device API, while backend selection,
    visibility, transport qualification, and model capabilities remain
    platform-specific.
    """

    def get_stage_process_env(
        self,
        spec: StageLaunchConfig,
        env: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        if spec.tp_size <= 1:
            return {}
        if spec.gpu_id is None:
            raise ValueError(f"tp stage {spec.stage_name!r} requires a GPU id")

        source_env = env if env is not None else os.environ
        visible_devices = _resolve_physical_visible_devices(source_env)
        if visible_devices:
            if spec.gpu_id >= len(visible_devices):
                raise ValueError(
                    f"tp stage {spec.stage_name!r} assigned gpu_id={spec.gpu_id}, "
                    f"but ROCm visibility only exposes {visible_devices}"
                )
            physical_device = visible_devices[spec.gpu_id]
        else:
            physical_device = str(spec.gpu_id)

        return {
            # ROCr owns the physical-device selection. HIP and CUDA compatibility
            # aliases then address the single exposed device as logical device 0.
            "ROCR_VISIBLE_DEVICES": physical_device,
            "HIP_VISIBLE_DEVICES": "0",
            "CUDA_VISIBLE_DEVICES": "0",
            "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS": "true",
            "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
        }

    def get_intra_node_transport(self) -> TransportKind:
        from sglang_omni.comm.data_ref import TransportKind

        # CUDA-shaped HIP IPC APIs are not sufficient evidence for Omni's relay
        # lifetime and failure semantics. Use the host-staged fallback until the
        # ROCm IPC path is qualified independently.
        return TransportKind.SHM

    def get_remote_transport(self) -> TransportKind | None:
        # Cross-node ROCm transport is a separate qualification and dependency
        # layer. Fail before constructing CUDA-oriented Mooncake state.
        return None
