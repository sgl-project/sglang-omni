# SPDX-License-Identifier: Apache-2.0
"""Process-scoped GPU memory accounting helpers."""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class _InvalidGpuDeviceError(RuntimeError):
    pass


def parse_cuda_visible_devices(value: str | None = None) -> list[int | str]:
    """Parse CUDA_VISIBLE_DEVICES into physical indices, UUIDs, or MIG ids."""

    if value is None:
        value = os.environ.get("CUDA_VISIBLE_DEVICES")
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


def resolve_visible_device_id(
    logical_gpu_id: int,
    visible_devices: list[int | str],
) -> int | str:
    """Map a CUDA logical GPU id to the corresponding NVML device id."""

    if logical_gpu_id < 0:
        raise _InvalidGpuDeviceError(f"Invalid GPU device {logical_gpu_id}")
    if not visible_devices:
        return logical_gpu_id
    if logical_gpu_id >= len(visible_devices):
        raise _InvalidGpuDeviceError(
            f"Invalid GPU device {logical_gpu_id}. CUDA_VISIBLE_DEVICES exposes "
            f"{len(visible_devices)} device(s): {visible_devices}"
        )
    return visible_devices[logical_gpu_id]


def is_process_scoped_memory_available() -> bool:
    """Return whether NVML process-scoped memory queries are available."""

    pynvml = _try_import_pynvml()
    if pynvml is None:
        return False
    try:
        pynvml.nvmlInit()
        return True
    except Exception:
        return False
    finally:
        _shutdown_nvml(pynvml)


def get_process_gpu_memory_bytes(logical_gpu_id: int) -> int | None:
    """Return current-process GPU memory on a CUDA logical device.

    The returned value is in bytes. ``None`` means NVML is unavailable or the
    process-scoped query failed. Invalid device mappings raise ``RuntimeError``
    because those are launch/configuration errors.
    """

    visible_devices = parse_cuda_visible_devices()
    device_id = resolve_visible_device_id(logical_gpu_id, visible_devices)

    pynvml = _try_import_pynvml()
    if pynvml is None:
        return None

    try:
        pynvml.nvmlInit()
    except Exception as exc:
        logger.warning("NVML init failed; process GPU memory is unavailable: %s", exc)
        return None

    try:
        if visible_devices:
            try:
                handle = _get_device_handle(pynvml, device_id)
            except Exception as exc:
                raise _InvalidGpuDeviceError(
                    f"Failed to get NVML handle for visible device {device_id!r} "
                    f"(logical_gpu_id={logical_gpu_id}). Check CUDA_VISIBLE_DEVICES "
                    "and stage GPU placement."
                ) from exc
        else:
            device_count = pynvml.nvmlDeviceGetCount()
            if logical_gpu_id >= device_count:
                raise _InvalidGpuDeviceError(
                    f"Invalid GPU device {logical_gpu_id}. Only {device_count} "
                    "GPU(s) are visible to NVML."
                )
            handle = pynvml.nvmlDeviceGetHandleByIndex(logical_gpu_id)

        pid = os.getpid()
        for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            if proc.pid == pid:
                return int(proc.usedGpuMemory)
        return 0
    except _InvalidGpuDeviceError:
        raise
    except Exception as exc:
        logger.warning("NVML query failed; process GPU memory is unavailable: %s", exc)
        return None
    finally:
        _shutdown_nvml(pynvml)


def format_bytes_gib(value: int | None) -> str:
    if value is None:
        return "None"
    return f"{value / (1024**3):.2f}GiB"


def _try_import_pynvml() -> Any | None:
    try:
        return importlib.import_module("pynvml")
    except ModuleNotFoundError:
        return None


def _get_device_handle(pynvml: Any, device_id: int | str) -> Any:
    if isinstance(device_id, int):
        return pynvml.nvmlDeviceGetHandleByIndex(device_id)

    get_by_uuid = pynvml.nvmlDeviceGetHandleByUUID
    try:
        return get_by_uuid(device_id)
    except TypeError:
        return get_by_uuid(device_id.encode("utf-8"))


def _shutdown_nvml(pynvml: Any) -> None:
    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass
