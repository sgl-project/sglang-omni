# SPDX-License-Identifier: Apache-2.0
"""Shared device abstraction for SGLang-Omni platforms."""

from __future__ import annotations

import enum
import logging
from collections.abc import Mapping, MutableMapping
from typing import Any

import torch

logger = logging.getLogger(__name__)


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

    def visible_device_value(self, env: Mapping[str, str]) -> str | None:
        """
        Return the platform's raw device-visibility value from ``env``.
        """
        if self.device_control_env_var is None:
            return None
        return env.get(self.device_control_env_var)

    def visible_devices(self, env: Mapping[str, str]) -> list[int | str]:
        """Parse the platform's visible device selectors from ``env``.

        Numeric selectors are returned as integers. Opaque selectors, such as
        device UUIDs, retain their string representation and ordering.

        e.g. ``CUDA_VISIBLE_DEVICES=3,4`` returns ``[3, 4]``
        """
        value = self.visible_device_value(env)
        if not value:
            return []

        devices: list[int | str] = []
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                devices.append(int(item))
            except ValueError:
                devices.append(item)
        return devices

    def worker_device_env(
        self,
        logical_device_id: int,
        env: Mapping[str, str],
    ) -> dict[str, str]:
        """Return the visibility override that isolates one worker device.

        ``logical_device_id`` indexes the selectors currently visible through
        ``env``. For example, logical device 0 under
        ``CUDA_VISIBLE_DEVICES=3,4`` produces
        ``{"CUDA_VISIBLE_DEVICES": "3"}``. Without an existing visibility
        mask, the logical ID is used as the physical selector.

        Raises:
            RuntimeError: If the platform has no device-control variable.
            ValueError: If the device ID is negative or outside the current
                visibility mask.
        """
        env_var = self.device_control_env_var
        if env_var is None:
            raise RuntimeError("Accelerator worker requires a device control variable")
        if logical_device_id < 0:
            raise ValueError(f"Invalid device id {logical_device_id}")

        visible_devices = self.visible_devices(env)
        if visible_devices:
            if logical_device_id >= len(visible_devices):
                raise ValueError(
                    f"Device id {logical_device_id} is not visible: {env_var} "
                    f"only exposes {visible_devices}"
                )
            selector = visible_devices[logical_device_id]
        else:
            selector = logical_device_id
        return {env_var: str(selector)}

    def compatibility_env_defaults(
        self,
        env: Mapping[str, str],
    ) -> dict[str, str]:
        """Return unset environment defaults required by this platform.
        e.g. _FLASHINFER_USE_CUDA_NORM
        """
        return {}

    def apply_compatibility_env_defaults(
        self,
        env: MutableMapping[str, str],
    ) -> dict[str, str]:
        """
        Apply and return this platform's compatibility defaults in ``env``.
        """
        overrides = self.compatibility_env_defaults(env)
        for key, value in overrides.items():
            env[key] = value
            logger.info(
                "Applied device compatibility env override: %s=%s",
                key,
                value,
            )
        return overrides

    def reclaim_process_memory(
        self,
        device: torch.device,
        *,
        suppress_errors: bool = False,
    ) -> None:
        """Release process-scoped accelerator resources for ``device``."""


class CpuDeviceMixin(DeviceMixin):
    _enum = PlatformEnum.CPU
    device_name = "cpu"
    device_type = "cpu"

    def get_device(self, device_id: int = 0) -> torch.device:
        return torch.device("cpu")

    def set_device(self, device: torch.device) -> None:
        pass

    def device_count(self) -> int:
        return 0

    def get_device_properties(self, device_id: int = 0) -> Any:
        raise RuntimeError("CPU has no accelerator device properties")


class CpuPlatform(CpuDeviceMixin, Platform):
    pass
