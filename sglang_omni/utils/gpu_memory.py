# SPDX-License-Identifier: Apache-2.0
"""Process-scoped GPU memory accounting helpers."""

from __future__ import annotations

import importlib
import logging
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class _InvalidGpuDeviceError(RuntimeError):
    pass


@dataclass(frozen=True)
class GpuDeviceInfo:
    logical_gpu_id: int
    device_id: int | str | None
    name: str | None
    total_memory_bytes: int | None


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

    The returned value is in bytes. None means NVML is unavailable or the
    process-scoped query failed. Invalid device mappings raise RuntimeError
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
        logger.debug("NVML init failed; process GPU memory is unavailable: %s", exc)
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
        logger.debug("NVML query failed; process GPU memory is unavailable: %s", exc)
        return None
    finally:
        _shutdown_nvml(pynvml)


def get_gpu_device_info(logical_gpu_id: int) -> GpuDeviceInfo:
    """Return best-effort CUDA device metadata.

    NVML is preferred because it follows CUDA_VISIBLE_DEVICES mappings for
    physical ids and UUIDs. If NVML metadata is unavailable, PyTorch CUDA
    metadata is used for total memory when possible.
    """

    info = GpuDeviceInfo(
        logical_gpu_id=logical_gpu_id,
        device_id=None,
        name=None,
        total_memory_bytes=None,
    )
    visible_devices = parse_cuda_visible_devices()
    try:
        device_id = resolve_visible_device_id(logical_gpu_id, visible_devices)
    except _InvalidGpuDeviceError as exc:
        logger.debug(f"GPU device metadata is unavailable: {exc}")
        return info

    pynvml = _try_import_pynvml()
    if pynvml is None:
        return _get_torch_gpu_device_info(logical_gpu_id, device_id)

    try:
        pynvml.nvmlInit()
    except Exception as exc:
        logger.debug(
            f"NVML init failed; using PyTorch GPU metadata if available: {exc}"
        )
        return _get_torch_gpu_device_info(logical_gpu_id, device_id)

    try:
        handle = _get_device_handle(pynvml, device_id)
        name = _decode_nvml_string(pynvml.nvmlDeviceGetName(handle))
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return GpuDeviceInfo(
            logical_gpu_id=logical_gpu_id,
            device_id=device_id,
            name=name,
            total_memory_bytes=int(memory_info.total),
        )
    except Exception as exc:
        logger.debug(
            f"NVML metadata query failed; using PyTorch GPU metadata if available: "
            f"{exc}"
        )
        return _get_torch_gpu_device_info(logical_gpu_id, device_id)
    finally:
        _shutdown_nvml(pynvml)


def _get_torch_gpu_device_info(
    logical_gpu_id: int,
    device_id: int | str | None,
) -> GpuDeviceInfo:
    """Return CUDA device metadata available through PyTorch."""
    try:
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        properties = torch.cuda.get_device_properties(logical_gpu_id)
        return GpuDeviceInfo(
            logical_gpu_id=logical_gpu_id,
            device_id=device_id,
            name=getattr(properties, "name", None),
            total_memory_bytes=int(properties.total_memory),
        )
    except Exception as exc:
        logger.debug(f"PyTorch GPU device metadata is unavailable: {exc}")
        return GpuDeviceInfo(
            logical_gpu_id=logical_gpu_id,
            device_id=device_id,
            name=None,
            total_memory_bytes=None,
        )


def format_bytes_gib(value: int | None) -> str:
    if value is None:
        return "None"
    return f"{value / (1024**3):.2f}GiB"


def calculate_stage_budget_available_bytes(
    *,
    total_memory_bytes: int,
    accounted_memory_bytes: int,
    memory_fraction: float,
    accounted_memory_label: str = "accounted_used",
) -> int:
    """Return stage KV headroom under a total GPU memory fraction."""
    if total_memory_bytes <= 0:
        raise ValueError("total_memory_bytes must be positive")
    if accounted_memory_bytes < 0:
        raise ValueError("accounted_memory_bytes must be non-negative")
    if not 0.0 < memory_fraction <= 1.0:
        raise ValueError("memory_fraction must be in (0, 1]")

    requested_bytes = int(total_memory_bytes * memory_fraction)
    available_bytes = requested_bytes - accounted_memory_bytes
    if available_bytes <= 0:
        raise RuntimeError(
            "Colocated GPU memory budget leaves no KV-cache headroom: "
            f"total={format_bytes_gib(total_memory_bytes)}, "
            f"fraction={memory_fraction:.3f}, "
            f"budget={format_bytes_gib(requested_bytes)}, "
            f"{accounted_memory_label}={format_bytes_gib(accounted_memory_bytes)}"
        )
    return available_bytes


def calculate_stage_load_delta_bytes(
    *,
    pre_model_load_memory_gib: float,
    post_model_load_memory_gib: float,
) -> int:
    """Return GPU memory consumed between two free-memory samples."""
    if pre_model_load_memory_gib < 0:
        raise ValueError("pre_model_load_memory_gib must be non-negative")
    if post_model_load_memory_gib < 0:
        raise ValueError("post_model_load_memory_gib must be non-negative")
    if post_model_load_memory_gib > pre_model_load_memory_gib:
        raise RuntimeError(
            "Stage load memory delta is negative: "
            f"pre_load={pre_model_load_memory_gib:.2f}GiB, "
            f"post_load={post_model_load_memory_gib:.2f}GiB"
        )

    return int((pre_model_load_memory_gib - post_model_load_memory_gib) * (1024**3))


def get_gpu_startup_lock_path(
    logical_gpu_id: int,
    *,
    env: dict[str, str] | None = None,
    base_dir: str | Path | None = None,
) -> Path:
    """Return the launch-time lock path for a CUDA logical GPU id."""

    source_env = env if env is not None else os.environ
    visible_devices = parse_cuda_visible_devices(source_env.get("CUDA_VISIBLE_DEVICES"))
    visible_device = resolve_visible_device_id(logical_gpu_id, visible_devices)
    safe_device = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(visible_device)).strip("_")
    if not safe_device:
        safe_device = str(logical_gpu_id)
    lock_dir = Path(base_dir) if base_dir is not None else Path(tempfile.gettempdir())
    return lock_dir / f"sglang_omni_gpu_{safe_device}_startup.lock"


@contextmanager
def gpu_startup_lock(logical_gpu_id: int):
    """Serialize heavyweight scheduler construction on one visible GPU."""

    import fcntl

    lock_path = get_gpu_startup_lock_path(logical_gpu_id)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield lock_path
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


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


def _decode_nvml_string(value: str | bytes) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _shutdown_nvml(pynvml: Any) -> None:
    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass
