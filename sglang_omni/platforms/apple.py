# SPDX-License-Identifier: Apache-2.0
"""Apple Silicon platform policy for PyTorch MPS and the opt-in MLX backend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from sglang.srt.platforms.device_mixin import PlatformEnum

from sglang_omni.platforms.interface import OmniPlatform

if TYPE_CHECKING:
    from sglang_omni.pipeline.stage_workers import StageLaunchConfig


class AppleOmniPlatform(OmniPlatform):
    """Single-device Apple Metal platform.

    SGLang enables its MLX runner through SGLANG_USE_MLX=1 but does not
    register an MPS SRTPlatform. Omni still needs concrete device operations
    because every accelerator-backed stage binds its scheduler thread through
    current_platform.get_device and set_device.
    """

    _enum: PlatformEnum = PlatformEnum.MPS
    device_name: str = "mps"
    device_type: str = "mps"

    @staticmethod
    def _validate_device_id(device_id: int) -> None:
        if int(device_id) != 0:
            raise ValueError(
                f"Apple Silicon exposes one Metal device, got device_id={device_id}"
            )

    def get_device(self, device_id: int = 0) -> torch.device:
        self._validate_device_id(device_id)
        return torch.device("mps")

    def set_device(self, device: torch.device | int) -> None:
        if isinstance(device, torch.device):
            if device.type != "mps":
                raise ValueError(f"Expected an MPS device, got {device}")
            index = 0 if device.index is None else device.index
        else:
            index = int(device)
        self._validate_device_id(index)
        # note (yexiaodong): PyTorch MPS and MLX share one process-global Metal
        # device, so there is no CUDA-style device selection to perform.

    def get_device_name(self, device_id: int = 0) -> str:
        self._validate_device_id(device_id)
        return "Apple Metal"

    def get_device_total_memory(self, device_id: int = 0) -> int:
        self._validate_device_id(device_id)
        from sglang.srt.utils.tensor_bridge import use_mlx

        if use_mlx():
            import mlx.core as mx

            return int(mx.device_info()["max_recommended_working_set_size"])
        return int(torch.mps.recommended_max_memory())

    def get_current_memory_usage(self, device: torch.device | None = None) -> float:
        if device is not None and device.type != "mps":
            raise ValueError(f"Expected an MPS device, got {device}")
        from sglang.srt.utils.tensor_bridge import use_mlx

        if use_mlx():
            import mlx.core as mx

            return float(mx.get_active_memory())
        return float(torch.mps.current_allocated_memory())

    def get_stage_process_env(
        self,
        spec: StageLaunchConfig,
        env: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        del env
        if spec.tp_size > 1:
            raise ValueError(
                f"Apple Silicon stage {spec.stage_name!r} requires tp_size=1; "
                f"got tp_size={spec.tp_size}"
            )
        if spec.gpu_id is not None:
            self._validate_device_id(spec.gpu_id)
        return {}

    def get_intra_node_transport(self):
        from sglang_omni.comm.data_ref import TransportKind

        return TransportKind.SHM

    def empty_cache(self) -> None:
        from sglang.srt.utils.tensor_bridge import use_mlx

        if use_mlx():
            import mlx.core as mx

            mx.clear_cache()
        else:
            torch.mps.empty_cache()

    def synchronize(self) -> None:
        from sglang.srt.utils.tensor_bridge import use_mlx

        if use_mlx():
            import mlx.core as mx

            mx.synchronize()
        else:
            torch.mps.synchronize()

    def enable_code2wav_graph(self) -> bool:
        return False


__all__ = ["AppleOmniPlatform"]
