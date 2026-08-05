# SPDX-License-Identifier: Apache-2.0
"""CUDA platform implementation."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from typing import Any

import torch

from sglang_omni.platforms.interface import (
    DeviceMixin,
    OmniPlatform,
    PlatformEnum,
    TransferPolicy,
)
from sglang_omni.utils.gpu_compat import get_gpu_compat_env_defaults


class CudaDeviceMixin(DeviceMixin):
    _enum = PlatformEnum.CUDA
    device_name = "cuda"
    device_type = "cuda"

    def device_count(self) -> int:
        return int(torch.cuda.device_count())

    def set_device(self, device: torch.device) -> None:
        torch.cuda.set_device(device)

    def get_device_properties(self, device_id: int = 0) -> Any:
        return torch.cuda.get_device_properties(device_id)

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()


class CudaOmniPlatform(CudaDeviceMixin, OmniPlatform):
    device_control_env_var = "CUDA_VISIBLE_DEVICES"
    distributed_backend = "nccl"

    def transfer_policy(self) -> TransferPolicy:
        return TransferPolicy.CUDA_IPC

    def support_same_device_weight_sharing(self) -> bool:
        return True

    def support_cross_node_transport(self) -> bool:
        return True

    def compatibility_env_defaults(self, env: Mapping[str, str]) -> dict[str, str]:
        return get_gpu_compat_env_defaults(env)

    def reclaim_process_memory(
        self, device: torch.device, *, suppress_errors: bool = False
    ) -> None:
        self.set_device(device)
        if suppress_errors:
            with suppress(Exception):
                self.synchronize()
        else:
            self.synchronize()
        self.empty_cache()
        if suppress_errors:
            with suppress(Exception):
                torch.cuda.ipc_collect()
        else:
            torch.cuda.ipc_collect()

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        return tuple(int(value) for value in torch.cuda.mem_get_info(device_id))
