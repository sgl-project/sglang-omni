# SPDX-License-Identifier: Apache-2.0
"""Shared device abstraction for SGLang-Omni platforms."""

from __future__ import annotations

import enum
from typing import Any

import torch


class PlatformEnum(enum.Enum):
    """Known platform types, following SGLang's platform convention."""

    CUDA = enum.auto()
    ROCM = enum.auto()
    CPU = enum.auto()
    OOT = enum.auto()
    UNSPECIFIED = enum.auto()


class DeviceMixin:
    """Device identity and operations shared by Omni platform implementations."""

    _enum: PlatformEnum = PlatformEnum.UNSPECIFIED
    device_name: str = "unknown"
    device_type: str = "cpu"

    def is_cuda(self) -> bool:
        return self._enum is PlatformEnum.CUDA

    def is_rocm(self) -> bool:
        return self._enum is PlatformEnum.ROCM

    def is_cpu(self) -> bool:
        return self._enum is PlatformEnum.CPU

    def get_device(self, device_id: int = 0) -> torch.device:
        return torch.device(self.device_type, device_id)

    def set_device(self, device: torch.device) -> None:
        raise NotImplementedError

    def device_count(self) -> int:
        raise NotImplementedError

    def get_device_properties(self, device_id: int = 0) -> Any:
        raise NotImplementedError

    def empty_cache(self) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device_name})"


class Platform(DeviceMixin):
    """Omni's process-lifecycle additions to SGLang's device vocabulary."""

    device_control_env_var: str | None = None

    def reclaim_process_memory(
        self,
        device: torch.device,
        *,
        suppress_errors: bool = False,
    ) -> None:
        pass


class CpuDeviceMixin(DeviceMixin):
    _enum = PlatformEnum.CPU
    device_name = "cpu"
    device_type = "cpu"

    def get_device(self, device_id: int = 0) -> torch.device:
        del device_id
        return torch.device("cpu")

    def set_device(self, device: torch.device) -> None:
        del device

    def device_count(self) -> int:
        return 0

    def get_device_properties(self, device_id: int = 0) -> Any:
        del device_id
        raise RuntimeError("CPU has no accelerator device properties")


class CpuPlatform(CpuDeviceMixin, Platform):
    pass
