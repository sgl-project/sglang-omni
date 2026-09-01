# SPDX-License-Identifier: Apache-2.0
"""Fast-AR attention backend policy for FishAudio S2-Pro."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sglang_omni.utils.gpu_compat import get_visible_gpu_sm_version


@dataclass(frozen=True)
class FastARBackendPlan:
    backend: str
    enable_torch_compile: bool
    device_type: str
    enable_cuda_graph: bool = True


_CUDA_BACKENDS_BY_SM = {
    89: FastARBackendPlan(
        backend="flashinfer", enable_torch_compile=False, device_type="cuda"
    ),
    90: FastARBackendPlan(
        backend="fa3", enable_torch_compile=True, device_type="cuda"
    ),
    100: FastARBackendPlan(
        backend="flashinfer", enable_torch_compile=False, device_type="cuda"
    ),
    120: FastARBackendPlan(
        backend="flashinfer", enable_torch_compile=False, device_type="cuda"
    ),
}

_MUSA_BACKEND = FastARBackendPlan(
    backend="fa3",
    enable_torch_compile=True,
    device_type="musa",
)


def _current_device_type() -> str:
    try:
        from sglang_omni.platforms import current_platform

        return current_platform.device_type
    except Exception:
        return "cuda"


def resolve_fast_ar_backend_plan(
    *,
    gpu_id: int,
    device_type: str | None = None,
    sm_resolver: Callable[[int], int | None] = get_visible_gpu_sm_version,
    flashinfer_available: Callable[[], bool] | None = None,
) -> FastARBackendPlan:
    """Resolve the FishAudio Fast-AR backend for the active accelerator."""

    device_type = device_type or _current_device_type()
    if device_type == "musa":
        return _MUSA_BACKEND
    if device_type != "cuda":
        raise RuntimeError(
            "FishAudio S2-Pro Fast-AR attention supports CUDA and MUSA; "
            f"got device_type={device_type!r}."
        )

    sm_version = sm_resolver(gpu_id)
    if sm_version is None:
        raise RuntimeError(
            "FishAudio S2-Pro cannot validate Fast-AR attention because "
            f"CUDA compute capability for gpu_id={gpu_id} could not be detected."
        )

    plan = _CUDA_BACKENDS_BY_SM.get(sm_version)
    if plan is None:
        raise RuntimeError(
            f"FishAudio S2-Pro Fast-AR does not support SM{sm_version}; "
            "supported architectures are SM89, SM90, SM100, and SM120. "
            "A Slow-AR attention_backend override cannot bypass this requirement."
        )

    if flashinfer_available is None:
        from sglang_omni.vendor.sglang.utils import is_flashinfer_available

        flashinfer_available = is_flashinfer_available

    if plan.backend == "flashinfer" and not flashinfer_available():
        raise RuntimeError(
            f"FishAudio S2-Pro Fast-AR requires FlashInfer on SM{sm_version}, but "
            "FlashInfer is unavailable. Install and enable FlashInfer and ensure "
            "SGLANG_IS_FLASHINFER_AVAILABLE is not false; a Slow-AR "
            "attention_backend override cannot bypass this requirement."
        )
    return plan


def cuda_fast_ar_uses_fa3(
    *,
    device_index: int,
    capability_resolver: Callable[[int], tuple[int, int]],
) -> bool:
    major, minor = capability_resolver(device_index)
    sm_version = major * 10 + minor
    return (
        resolve_fast_ar_backend_plan(
            gpu_id=device_index,
            device_type="cuda",
            sm_resolver=lambda _gpu_id: sm_version,
            flashinfer_available=lambda: True,
        ).backend
        == "fa3"
    )
