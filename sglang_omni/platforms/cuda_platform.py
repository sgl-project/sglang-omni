# SPDX-License-Identifier: Apache-2.0
"""CUDA and ROCm device mixins and concrete platforms."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import torch

from sglang_omni.platforms.interface import DeviceMixin, Platform, PlatformEnum


class CudaDeviceMixin(DeviceMixin):
    _enum = PlatformEnum.CUDA
    device_name = "cuda"
    device_type = "cuda"

    def __init__(self, torch_module: Any = torch) -> None:
        self._torch = torch_module

    def device_count(self) -> int:
        return int(self._torch.cuda.device_count())

    def set_device(self, device: torch.device) -> None:
        self._torch.cuda.set_device(device)

    def get_device_properties(self, device_id: int = 0) -> Any:
        return self._torch.cuda.get_device_properties(device_id)

    def synchronize(self) -> None:
        self._torch.cuda.synchronize()

    def empty_cache(self) -> None:
        self._torch.cuda.empty_cache()


class CudaPlatform(CudaDeviceMixin, Platform):
    device_control_env_var = "CUDA_VISIBLE_DEVICES"

    def reclaim_process_memory(
        self,
        device: torch.device,
        *,
        suppress_errors: bool = False,
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
                self._torch.cuda.ipc_collect()
        else:
            self._torch.cuda.ipc_collect()


class RocmDeviceMixin(CudaDeviceMixin):
    """ROCm device operations exposed through PyTorch's CUDA-compatible API."""

    _enum = PlatformEnum.ROCM
    device_name = "rocm"
    # device_type remains "cuda": torch.device("rocm") is not valid.


class RocmPlatform(RocmDeviceMixin, CudaPlatform):
    device_control_env_var = "ROCR_VISIBLE_DEVICES"
