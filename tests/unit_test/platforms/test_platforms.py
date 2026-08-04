# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import sglang_omni.platforms.cuda_platform as cuda_platform_module
from sglang_omni.platforms import PlatformEnum, resolve_current_platform


class _Runtime:
    def __init__(self, available: bool = True):
        self._available = available
        self.calls: list[tuple] = []
        self.properties = SimpleNamespace(name="Test", total_memory=1024)

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return 2

    def set_device(self, device) -> None:
        self.calls.append(("set_device", device))

    def get_device_properties(self, device_id: int):
        self.calls.append(("get_device_properties", device_id))
        return self.properties

    def synchronize(self) -> None:
        self.calls.append(("synchronize",))

    def empty_cache(self) -> None:
        self.calls.append(("empty_cache",))

    def ipc_collect(self) -> None:
        self.calls.append(("ipc_collect",))


def _torch_runtime(
    *,
    cuda: _Runtime | None = None,
    hip: str | None = None,
):
    return SimpleNamespace(
        version=SimpleNamespace(hip=hip),
        cuda=cuda,
    )


def test_rocm_identity_is_distinct_from_torch_device_type() -> None:
    platform = resolve_current_platform(_torch_runtime(cuda=_Runtime(), hip="7.2"))

    assert platform._enum is PlatformEnum.ROCM
    assert platform.is_rocm()
    assert platform.device_name == "rocm"
    assert platform.device_type == "cuda"
    assert platform.get_device(3) == torch.device("cuda", 3)


def test_cuda_is_resolved_from_an_available_cuda_runtime() -> None:
    platform = resolve_current_platform(_torch_runtime(cuda=_Runtime()))

    assert platform._enum is PlatformEnum.CUDA
    assert platform.is_cuda()


def test_cpu_is_the_fallback_when_no_runtime_is_usable() -> None:
    platform = resolve_current_platform(_torch_runtime(cuda=_Runtime(False)))

    assert platform.is_cpu()
    device = platform.get_device()
    assert device == torch.device("cpu")
    assert platform.set_device(device) is None
    assert platform.empty_cache() is None
    assert platform.synchronize() is None


def test_platforms_define_one_device_control_variable() -> None:
    cuda = resolve_current_platform(_torch_runtime(cuda=_Runtime()))
    platform = resolve_current_platform(_torch_runtime(cuda=_Runtime(), hip="7.2"))

    assert cuda.device_control_env_var == "CUDA_VISIBLE_DEVICES"
    assert platform.device_control_env_var == "ROCR_VISIBLE_DEVICES"


def test_platform_resolves_worker_device_environment() -> None:
    platform = resolve_current_platform(_torch_runtime(cuda=_Runtime(), hip="7.2"))

    assert (
        platform.visible_device_value({"ROCR_VISIBLE_DEVICES": "3,GPU-abc"})
        == "3,GPU-abc"
    )
    assert platform.visible_devices({"ROCR_VISIBLE_DEVICES": "3,GPU-abc"}) == [
        3,
        "GPU-abc",
    ]
    assert platform.worker_device_env(1, {"ROCR_VISIBLE_DEVICES": "3,GPU-abc"}) == {
        "ROCR_VISIBLE_DEVICES": "GPU-abc"
    }


def test_platform_rejects_device_outside_visibility_mask() -> None:
    platform = resolve_current_platform(_torch_runtime(cuda=_Runtime()))

    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES only exposes"):
        platform.worker_device_env(1, {"CUDA_VISIBLE_DEVICES": "0"})


def test_cuda_owns_compatibility_environment_policy(monkeypatch) -> None:
    calls: list[dict[str, str]] = []

    def _defaults(env):
        calls.append(dict(env))
        return {"FLASHINFER_USE_CUDA_NORM": "1"}

    monkeypatch.setattr(
        cuda_platform_module,
        "get_gpu_compat_env_defaults",
        _defaults,
    )
    cuda = resolve_current_platform(_torch_runtime(cuda=_Runtime()))
    rocm = resolve_current_platform(_torch_runtime(cuda=_Runtime(), hip="7.2"))

    env = {"CUDA_VISIBLE_DEVICES": "0"}
    assert cuda.apply_compatibility_env_defaults(env) == {
        "FLASHINFER_USE_CUDA_NORM": "1"
    }
    assert env["FLASHINFER_USE_CUDA_NORM"] == "1"
    assert calls == [{"CUDA_VISIBLE_DEVICES": "0"}]
    assert rocm.compatibility_env_defaults({"ROCR_VISIBLE_DEVICES": "0"}) == {}


@pytest.mark.parametrize(
    ("hip", "expects_rocm"),
    [
        (None, False),
        ("7.2", True),
    ],
)
def test_runtime_lifecycle_contract(
    hip: str | None,
    expects_rocm: bool,
) -> None:
    runtime = _Runtime()
    platform = resolve_current_platform(_torch_runtime(cuda=runtime, hip=hip))
    device = platform.get_device(1)

    assert platform.is_rocm() is expects_rocm
    assert platform.device_count() == 2
    assert platform.get_device_properties(1) is runtime.properties
    platform.reclaim_process_memory(device)

    assert ("set_device", device) in runtime.calls
    assert ("synchronize",) in runtime.calls
    assert ("empty_cache",) in runtime.calls
    assert ("ipc_collect",) in runtime.calls


def test_reclaim_can_suppress_optional_cleanup_failures() -> None:
    class _FailingOptionalCleanupRuntime(_Runtime):
        def synchronize(self) -> None:
            self.calls.append(("synchronize",))
            raise RuntimeError("synchronize failed")

        def ipc_collect(self) -> None:
            self.calls.append(("ipc_collect",))
            raise RuntimeError("ipc collect failed")

    runtime = _FailingOptionalCleanupRuntime()
    platform = resolve_current_platform(_torch_runtime(cuda=runtime))

    platform.reclaim_process_memory(
        platform.get_device(0),
        suppress_errors=True,
    )

    assert ("synchronize",) in runtime.calls
    assert ("empty_cache",) in runtime.calls
    assert ("ipc_collect",) in runtime.calls


def test_reclaim_propagates_cleanup_failures_by_default() -> None:
    class _FailingRuntime(_Runtime):
        def synchronize(self) -> None:
            raise RuntimeError("synchronize failed")

    platform = resolve_current_platform(_torch_runtime(cuda=_FailingRuntime()))

    with pytest.raises(RuntimeError, match="synchronize failed"):
        platform.reclaim_process_memory(platform.get_device(0))
