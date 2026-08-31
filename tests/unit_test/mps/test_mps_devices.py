# SPDX-License-Identifier: Apache-2.0
"""Parent-process CUDA identity and NVML capability tests."""

from __future__ import annotations

import sys
import uuid
from enum import IntEnum
from types import ModuleType, SimpleNamespace

from sglang_omni.mps.devices import NvmlDeviceInfo

GPU_A = "GPU-aaaaaaaa-bbbb-cccc-dddd-000000000001"
GPU_B = "GPU-aaaaaaaa-bbbb-cccc-dddd-000000000007"


class _CudaStatus(IntEnum):
    SUCCESS = 0
    INVALID_DEVICE = 101


class _FakeDriver:
    def __init__(
        self,
        uuids: dict[int, str],
        *,
        init_status: _CudaStatus = _CudaStatus.SUCCESS,
    ):
        self.uuids = uuids
        self.init_status = init_status

    def cuInit(self, flags):
        return (self.init_status,)

    def cuDeviceGet(self, ordinal):
        if ordinal not in self.uuids:
            return _CudaStatus.INVALID_DEVICE, None
        return _CudaStatus.SUCCESS, ordinal

    def cuDeviceGetUuid(self, device):
        raw_uuid = uuid.UUID(self.uuids[device].removeprefix("GPU-")).bytes
        return _CudaStatus.SUCCESS, SimpleNamespace(bytes=raw_uuid)


def _fake_pynvml(failed_uuids: set[str] | None = None):
    class NvmlError(Exception):
        pass

    class NvmlNotSupported(NvmlError):
        pass

    handles: list[str] = []
    failures = failed_uuids or set()

    def by_uuid(raw_uuid):
        gpu_uuid = raw_uuid.decode()
        if gpu_uuid in failures:
            raise NvmlError(f"cannot inspect {gpu_uuid}")
        handles.append(gpu_uuid)
        return gpu_uuid

    return SimpleNamespace(
        handles=handles,
        NVMLError=NvmlError,
        NVMLError_NotSupported=NvmlNotSupported,
        NVML_DEVICE_MIG_ENABLE=1,
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByUUID=by_uuid,
        nvmlDeviceGetUUID=lambda handle: handle,
        nvmlDeviceGetCudaComputeCapability=lambda _handle: (9, 0),
        nvmlDeviceGetMigMode=lambda _handle: (0, 0),
    )


def _install_cuda_driver(monkeypatch, driver) -> None:
    cuda = ModuleType("cuda")
    bindings = ModuleType("cuda.bindings")
    cuda.bindings = bindings
    bindings.driver = driver
    monkeypatch.setitem(sys.modules, "cuda", cuda)
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)


def test_nvml_uses_driver_uuid_when_cuda_order_differs_from_nvml_index(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml())
    _install_cuda_driver(monkeypatch, _FakeDriver({0: GPU_B, 1: GPU_A}))

    devices = NvmlDeviceInfo().inspect([1, 0, 1])

    assert {ordinal: device.gpu_uuid for ordinal, device in devices.items()} == {
        0: GPU_B,
        1: GPU_A,
    }


def test_cuda_resolution_failure_never_falls_back_to_nvml_index(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml())
    _install_cuda_driver(
        monkeypatch,
        _FakeDriver({}, init_status=_CudaStatus.INVALID_DEVICE),
    )

    device = NvmlDeviceInfo().inspect([2])[2]

    assert device.gpu_uuid is None
    assert "INVALID_DEVICE" in device.unsupported_reason


def test_nvml_failure_preserves_driver_resolved_uuid(monkeypatch):
    pynvml = _fake_pynvml({GPU_B})
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)
    _install_cuda_driver(monkeypatch, _FakeDriver({0: GPU_A, 1: GPU_B}))

    inspected = NvmlDeviceInfo().inspect([0, 1])

    assert inspected[0].gpu_uuid == GPU_A
    assert inspected[0].unsupported_reason is None
    assert inspected[1].gpu_uuid == GPU_B
    assert "NVML query failed" in inspected[1].unsupported_reason
