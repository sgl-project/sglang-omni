# SPDX-License-Identifier: Apache-2.0
"""GPU-specific runtime compatibility helpers."""

from __future__ import annotations

import importlib
import logging
import os
from typing import Mapping

from sglang_omni.utils.gpu_memory import (
    _get_device_handle,
    _shutdown_nvml,
    _try_import_pynvml,
    get_gpu_device_info,
    parse_cuda_visible_devices,
    resolve_visible_device_id,
)

logger = logging.getLogger(__name__)

_FLASHINFER_USE_CUDA_NORM = "FLASHINFER_USE_CUDA_NORM"
_BLACKWELL_GPU_NAME_MARKERS = ("B200", "BLACKWELL")
_BLACKWELL_MIN_MAJOR_COMPUTE_CAPABILITY = 10


def _is_blackwell_gpu_name(gpu_name: str | None) -> bool:
    if not gpu_name:
        return False
    normalized = gpu_name.upper()
    return any(marker in normalized for marker in _BLACKWELL_GPU_NAME_MARKERS)


def _is_blackwell_compute_capability(major: int, minor: int = 0) -> bool:
    del minor
    return major >= _BLACKWELL_MIN_MAJOR_COMPUTE_CAPABILITY


def _get_compute_capability(logical_gpu_id: int) -> tuple[int, int] | None:
    visible_devices = parse_cuda_visible_devices()
    try:
        device_id = resolve_visible_device_id(logical_gpu_id, visible_devices)
    except Exception:
        return None

    pynvml = _try_import_pynvml()
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            if visible_devices:
                handle = _get_device_handle(pynvml, device_id)
            else:
                handle = pynvml.nvmlDeviceGetHandleByIndex(logical_gpu_id)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            return int(major), int(minor)
        except Exception as exc:
            logger.debug(
                "NVML compute capability query failed for gpu_id=%s: %s",
                logical_gpu_id,
                exc,
            )
        finally:
            _shutdown_nvml(pynvml)

    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(logical_gpu_id)
            return int(properties.major), int(properties.minor)
    except Exception as exc:
        logger.debug(
            "PyTorch compute capability query failed for gpu_id=%s: %s",
            logical_gpu_id,
            exc,
        )
    return None


def _visible_gpu_ids() -> list[int]:
    visible_devices = parse_cuda_visible_devices()
    if visible_devices:
        return list(range(len(visible_devices)))
    return [0]


def visible_gpus_need_flashinfer_cuda_norm() -> bool:
    """Return whether any visible CUDA device needs the FlashInfer CUDA norm workaround."""
    for gpu_id in _visible_gpu_ids():
        device_info = get_gpu_device_info(gpu_id)
        if _is_blackwell_gpu_name(device_info.name):
            return True
        capability = _get_compute_capability(gpu_id)
        if capability is not None and _is_blackwell_compute_capability(*capability):
            return True
    return False


def get_gpu_compat_env_defaults(
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return env overrides needed for the current visible GPU topology."""
    source_env = os.environ if env is None else env
    if source_env.get(_FLASHINFER_USE_CUDA_NORM) is not None:
        return {}
    if not visible_gpus_need_flashinfer_cuda_norm():
        return {}
    return {_FLASHINFER_USE_CUDA_NORM: "1"}


def apply_gpu_compat_env_defaults(
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Apply GPU compatibility env overrides to the current process."""
    overrides = get_gpu_compat_env_defaults(env)
    for key, value in overrides.items():
        os.environ[key] = value
        logger.info("Applied GPU compatibility env override: %s=%s", key, value)
    return overrides
