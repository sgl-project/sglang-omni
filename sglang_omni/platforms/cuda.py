from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

from sglang.srt.platforms.cuda import CudaDeviceMixin
from sglang.srt.platforms.rocm import RocmDeviceMixin

from sglang_omni.platforms.interface import OmniPlatform

if TYPE_CHECKING:
    from sglang_omni.pipeline.stage_workers import StageLaunchConfig


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


class ROCMOmniPlatform(RocmDeviceMixin, CUDAOmniPlatform):
    pass
