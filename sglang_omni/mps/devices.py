# SPDX-License-Identifier: Apache-2.0
"""CUDA/NVML physical-device identity and MPS capability checks."""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

_MIN_COMPUTE_CAPABILITY = (7, 0)


@dataclass(frozen=True)
class MpsPhysicalDevice:
    gpu_uuid: str | None
    unsupported_reason: str | None = None


def _check_cuda(status: Any, operation: str) -> None:
    if int(status) == 0:
        return
    detail = getattr(status, "name", str(int(status)))
    raise RuntimeError(f"{operation} failed with {detail}")


def _resolve_cuda_device_uuids(
    gpu_ids: Iterable[int],
    driver=None,
) -> tuple[dict[int, str], dict[int, str]]:
    """Resolve parent-visible CUDA ordinals without creating a context."""

    ordinals = tuple(sorted(set(gpu_ids)))
    if any(ordinal < 0 for ordinal in ordinals):
        raise ValueError(f"CUDA device ordinals must be non-negative: {ordinals}")
    if driver is None:
        from cuda.bindings import driver

    (status,) = driver.cuInit(0)
    _check_cuda(status, "cuInit")

    resolved: dict[int, str] = {}
    errors: dict[int, str] = {}
    for ordinal in ordinals:
        try:
            status, device = driver.cuDeviceGet(ordinal)
            _check_cuda(status, f"cuDeviceGet({ordinal})")
            status, device_uuid = driver.cuDeviceGetUuid(device)
            _check_cuda(status, f"cuDeviceGetUuid({ordinal})")
            raw_uuid = bytes(device_uuid.bytes)
            if len(raw_uuid) != 16:
                raise RuntimeError(
                    f"cuDeviceGetUuid({ordinal}) returned {len(raw_uuid)} bytes, "
                    "expected 16"
                )
            resolved[ordinal] = f"GPU-{uuid.UUID(bytes=raw_uuid)}"
        except Exception as exc:
            errors[ordinal] = str(exc)
    return resolved, errors


class NvmlDeviceInfo:
    def inspect(self, gpu_ids: Iterable[int]) -> dict[int, MpsPhysicalDevice]:
        """Resolve unique parent-visible ordinals, then inspect by CUDA UUID."""

        ordinals = tuple(sorted(set(gpu_ids)))
        if not ordinals:
            return {}
        try:
            uuid_by_ordinal, cuda_errors = _resolve_cuda_device_uuids(ordinals)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            return {
                ordinal: MpsPhysicalDevice(
                    None,
                    f"CUDA driver query failed: {exc}",
                )
                for ordinal in ordinals
            }

        devices = {
            ordinal: MpsPhysicalDevice(
                None,
                f"CUDA driver query failed: {error}",
            )
            for ordinal, error in cuda_errors.items()
        }
        if not uuid_by_ordinal:
            return devices

        try:
            import pynvml
        except ImportError:
            devices.update(
                {
                    ordinal: MpsPhysicalDevice(
                        gpu_uuid,
                        "pynvml is not installed",
                    )
                    for ordinal, gpu_uuid in uuid_by_ordinal.items()
                }
            )
            return devices

        try:
            pynvml.nvmlInit()
        except (pynvml.NVMLError, OSError) as exc:
            devices.update(
                {
                    ordinal: MpsPhysicalDevice(
                        gpu_uuid,
                        f"NVML query failed: {exc}",
                    )
                    for ordinal, gpu_uuid in uuid_by_ordinal.items()
                }
            )
            return devices

        for ordinal, gpu_uuid in uuid_by_ordinal.items():
            try:
                handle = pynvml.nvmlDeviceGetHandleByUUID(gpu_uuid.encode())
                nvml_uuid = pynvml.nvmlDeviceGetUUID(handle)
                physical_uuid = (
                    nvml_uuid.decode() if isinstance(nvml_uuid, bytes) else nvml_uuid
                )
                if physical_uuid.startswith("MIG-"):
                    devices[ordinal] = MpsPhysicalDevice(
                        gpu_uuid,
                        "MIG devices are not validated for native MPS in "
                        "SGLang Omni",
                    )
                    continue
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                if (major, minor) < _MIN_COMPUTE_CAPABILITY:
                    devices[ordinal] = MpsPhysicalDevice(
                        gpu_uuid,
                        (
                            f"compute capability {major}.{minor} is pre-Volta; "
                            "per-client isolation requires Volta or newer"
                        ),
                    )
                    continue
                try:
                    mig_current, _ = pynvml.nvmlDeviceGetMigMode(handle)
                    if mig_current == pynvml.NVML_DEVICE_MIG_ENABLE:
                        devices[ordinal] = MpsPhysicalDevice(
                            gpu_uuid,
                            (
                                "MIG mode is enabled; native MPS is not "
                                "validated for MIG deployments in SGLang "
                                "Omni, run with mps=off"
                            ),
                        )
                        continue
                except pynvml.NVMLError_NotSupported:
                    pass
                devices[ordinal] = MpsPhysicalDevice(gpu_uuid)
            except (pynvml.NVMLError, OSError, ValueError) as exc:
                devices[ordinal] = MpsPhysicalDevice(
                    gpu_uuid,
                    f"NVML query failed: {exc}",
                )
        return devices
