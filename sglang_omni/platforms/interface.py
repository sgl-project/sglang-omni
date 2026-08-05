# SPDX-License-Identifier: Apache-2.0
"""Shared device abstraction for SGLang-Omni platforms."""

from __future__ import annotations

import enum
import logging
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


class PlatformEnum(str, enum.Enum):
    """Known platform types, following SGLang's platform convention."""

    CUDA = "cuda"
    CPU = "cpu"
    UNSPECIFIED = "unspecified"


class TransferPolicy(str, enum.Enum):
    CUDA_IPC = "cuda_ipc"
    HOST_STAGED = "host_staged"


@dataclass(frozen=True)
class ResolvedPlatformSpec:
    platform_type: PlatformEnum
    device_type: str
    distributed_backend: str

    def __post_init__(self) -> None:
        actual = (
            self.platform_type,
            self.device_type,
            self.distributed_backend,
        )
        supported = {
            (PlatformEnum.CUDA, "cuda", "nccl"),
            (PlatformEnum.CPU, "cpu", "gloo"),
        }
        if actual not in supported:
            raise ValueError(f"Unsupported platform spec: {actual!r}")


class DeviceMixin:
    """Device identity and operations shared by Omni platform implementations."""

    _enum: PlatformEnum = PlatformEnum.UNSPECIFIED
    device_name: str = "unknown"
    device_type: str = "cpu"

    def is_cuda(self) -> bool:
        return self._enum is PlatformEnum.CUDA

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


class OmniPlatform(DeviceMixin):
    """Omni's process-lifecycle additions to SGLang's device vocabulary."""

    device_control_env_var: str | None = None
    distributed_backend: str = "gloo"

    @staticmethod
    def detect(torch_module: Any = torch) -> "OmniPlatform":
        from sglang_omni.platforms.cuda_platform import CudaOmniPlatform

        runtime = getattr(torch_module, "cuda", None)
        if runtime is not None and runtime.is_available():
            return CudaOmniPlatform()
        return CpuOmniPlatform()

    @staticmethod
    def from_spec(spec: ResolvedPlatformSpec) -> "OmniPlatform":
        from sglang_omni.platforms.cuda_platform import CudaOmniPlatform

        if spec.platform_type is PlatformEnum.CUDA:
            return CudaOmniPlatform()
        if spec.platform_type is PlatformEnum.CPU:
            return CpuOmniPlatform()
        raise ValueError(f"Unsupported platform type {spec.platform_type!r}")

    def to_spec(self) -> ResolvedPlatformSpec:
        return ResolvedPlatformSpec(
            platform_type=self._enum,
            device_type=self.device_type,
            distributed_backend=self.distributed_backend,
        )

    def transfer_policy(self) -> TransferPolicy:
        return TransferPolicy.HOST_STAGED

    def support_same_device_weight_sharing(self) -> bool:
        return False

    def support_worker_visibility_isolation(self) -> bool:
        return False

    def support_cross_node_transport(self) -> bool:
        return False

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
        self, logical_device_id: int, env: Mapping[str, str]
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

    def initialize_worker(self) -> None:
        pass

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        raise NotImplementedError

    def compatibility_env_defaults(self, env: Mapping[str, str]) -> dict[str, str]:
        """Return unset environment defaults required by this platform.
        e.g. _FLASHINFER_USE_CUDA_NORM
        """
        return {}

    def apply_compatibility_env_defaults(
        self, env: MutableMapping[str, str]
    ) -> dict[str, str]:
        """
        Apply and return this platform's compatibility defaults in ``env``.
        """
        overrides = self.compatibility_env_defaults(env)
        for key, value in overrides.items():
            env[key] = value
            logger.info("Applied device compatibility env override: %s=%s", key, value)
        return overrides

    def reclaim_process_memory(
        self, device: torch.device, *, suppress_errors: bool = False
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


class CpuOmniPlatform(CpuDeviceMixin, OmniPlatform):
    pass
