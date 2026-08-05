# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest
import torch

import sglang_omni.platforms.cuda_platform as cuda_platform_module
from sglang_omni.platforms import (
    OmniPlatform,
    PlatformEnum,
    ResolvedPlatformSpec,
    TransferPolicy,
    resolve_current_platform,
)


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

    def mem_get_info(self, device_id: int):
        return (512, 1024)


def _torch_runtime(cuda: _Runtime | None = None):
    return SimpleNamespace(cuda=cuda)


def test_cuda_is_resolved_from_an_available_cuda_runtime() -> None:
    platform = resolve_current_platform(_torch_runtime(_Runtime()))

    assert platform._enum is PlatformEnum.CUDA
    assert platform.is_cuda()
    assert platform.transfer_policy() is TransferPolicy.CUDA_IPC


def test_cpu_is_the_fallback_when_cuda_is_unavailable() -> None:
    platform = resolve_current_platform(_torch_runtime(_Runtime(False)))

    assert platform.is_cpu()
    assert platform.get_device() == torch.device("cpu")
    assert platform.transfer_policy() is TransferPolicy.HOST_STAGED


def test_resolved_platform_spec_round_trip() -> None:
    platform = resolve_current_platform(_torch_runtime(_Runtime()))
    spec = pickle.loads(pickle.dumps(platform.to_spec()))
    restored = OmniPlatform.from_spec(spec)

    assert spec == ResolvedPlatformSpec(PlatformEnum.CUDA, "cuda", "nccl")
    assert restored.is_cuda()


def test_platform_resolves_worker_device_environment() -> None:
    platform = resolve_current_platform(_torch_runtime(_Runtime()))

    assert platform.visible_devices({"CUDA_VISIBLE_DEVICES": "3,GPU-abc"}) == [
        3,
        "GPU-abc",
    ]
    assert platform.worker_device_env(1, {"CUDA_VISIBLE_DEVICES": "3,GPU-abc"}) == {
        "CUDA_VISIBLE_DEVICES": "GPU-abc"
    }


def test_platform_rejects_device_outside_visibility_mask() -> None:
    platform = resolve_current_platform(_torch_runtime(_Runtime()))

    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES only exposes"):
        platform.worker_device_env(1, {"CUDA_VISIBLE_DEVICES": "0"})


def test_cuda_owns_compatibility_environment_policy(monkeypatch) -> None:
    monkeypatch.setattr(
        cuda_platform_module,
        "get_gpu_compat_env_defaults",
        lambda env: {"FLASHINFER_USE_CUDA_NORM": "1"},
    )
    platform = resolve_current_platform(_torch_runtime(_Runtime()))
    env = {"CUDA_VISIBLE_DEVICES": "0"}

    assert platform.apply_compatibility_env_defaults(env) == {
        "FLASHINFER_USE_CUDA_NORM": "1"
    }
    assert env["FLASHINFER_USE_CUDA_NORM"] == "1"


def test_runtime_lifecycle_contract(monkeypatch) -> None:
    runtime = _Runtime()
    monkeypatch.setattr(cuda_platform_module.torch, "cuda", runtime)
    platform = resolve_current_platform(_torch_runtime(runtime))
    device = platform.get_device(1)

    assert platform.device_count() == 2
    assert platform.get_device_properties(1) is runtime.properties
    assert platform.get_available_memory(1) == (512, 1024)
    platform.reclaim_process_memory(device)

    assert ("set_device", device) in runtime.calls
    assert ("synchronize",) in runtime.calls
    assert ("empty_cache",) in runtime.calls
    assert ("ipc_collect",) in runtime.calls


def test_reclaim_can_suppress_optional_cleanup_failures(monkeypatch) -> None:
    class _FailingRuntime(_Runtime):
        def synchronize(self) -> None:
            raise RuntimeError("synchronize failed")

        def ipc_collect(self) -> None:
            raise RuntimeError("ipc collect failed")

    runtime = _FailingRuntime()
    monkeypatch.setattr(cuda_platform_module.torch, "cuda", runtime)
    platform = resolve_current_platform(_torch_runtime(runtime))
    platform.reclaim_process_memory(platform.get_device(), suppress_errors=True)


def test_reclaim_propagates_cleanup_failures_by_default(monkeypatch) -> None:
    class _FailingRuntime(_Runtime):
        def synchronize(self) -> None:
            raise RuntimeError("synchronize failed")

    runtime = _FailingRuntime()
    monkeypatch.setattr(cuda_platform_module.torch, "cuda", runtime)
    platform = resolve_current_platform(_torch_runtime(runtime))
    with pytest.raises(RuntimeError, match="synchronize failed"):
        platform.reclaim_process_memory(platform.get_device())
