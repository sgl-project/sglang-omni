# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import fcntl
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

import sglang_omni.utils.gpu_memory as gpu_memory


class _FakeNVML(ModuleType):
    def __init__(
        self,
        *,
        device_count: int = 4,
        device_name: str | bytes = b"NVIDIA H200",
        total_memory: int = 141 * 1024**3,
        processes: list[SimpleNamespace] | None = None,
        init_error: Exception | None = None,
        query_error: Exception | None = None,
        uuid_requires_bytes: bool = False,
    ) -> None:
        super().__init__("pynvml")
        self.device_count = device_count
        self.device_name = device_name
        self.total_memory = total_memory
        self.processes = processes or []
        self.init_error = init_error
        self.query_error = query_error
        self.uuid_requires_bytes = uuid_requires_bytes
        self.shutdown_called = False
        self.index_handles: list[int] = []
        self.uuid_handles: list[str | bytes] = []

    def nvmlInit(self) -> None:
        if self.init_error is not None:
            raise self.init_error

    def nvmlShutdown(self) -> None:
        self.shutdown_called = True

    def nvmlDeviceGetCount(self) -> int:
        return self.device_count

    def nvmlDeviceGetHandleByIndex(self, device_id: int) -> str:
        self.index_handles.append(device_id)
        return f"index:{device_id}"

    def nvmlDeviceGetHandleByUUID(self, device_id: str | bytes) -> str:
        if self.uuid_requires_bytes and isinstance(device_id, str):
            raise TypeError("uuid must be bytes")
        self.uuid_handles.append(device_id)
        return f"uuid:{device_id}"

    def nvmlDeviceGetComputeRunningProcesses(
        self,
        handle: str,
    ) -> list[SimpleNamespace]:
        if self.query_error is not None:
            raise self.query_error
        return self.processes

    def nvmlDeviceGetName(self, handle: str) -> str | bytes:
        if self.query_error is not None:
            raise self.query_error
        return self.device_name

    def nvmlDeviceGetMemoryInfo(self, handle: str) -> SimpleNamespace:
        if self.query_error is not None:
            raise self.query_error
        return SimpleNamespace(total=self.total_memory)


def _install_fake_nvml(monkeypatch: pytest.MonkeyPatch, fake: _FakeNVML) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", fake)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, []),
        ("", []),
        (
            " 0, 2, GPU-deadbeef, MIG-GPU-deadbeef/1/2,, ",
            [0, 2, "GPU-deadbeef", "MIG-GPU-deadbeef/1/2"],
        ),
    ],
)
def test_parse_cuda_visible_devices_handles_supported_forms(
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
    expected: list[int | str],
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert gpu_memory.parse_cuda_visible_devices(value) == expected


def test_parse_cuda_visible_devices_reads_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,GPU-abc")

    assert gpu_memory.parse_cuda_visible_devices() == [3, "GPU-abc"]


@pytest.mark.parametrize(
    ("logical_gpu_id", "visible_devices", "expected"),
    [
        (2, [], 2),
        (1, [4, "GPU-abc"], "GPU-abc"),
    ],
)
def test_resolve_visible_device_id_maps_logical_gpu_id(
    logical_gpu_id: int,
    visible_devices: list[int | str],
    expected: int | str,
) -> None:
    assert (
        gpu_memory.resolve_visible_device_id(logical_gpu_id, visible_devices)
        == expected
    )


@pytest.mark.parametrize(
    ("logical_gpu_id", "visible_devices", "match"),
    [
        (-1, [], "Invalid GPU device -1"),
        (1, [0], "CUDA_VISIBLE_DEVICES exposes 1"),
    ],
)
def test_resolve_visible_device_id_rejects_invalid_mapping(
    logical_gpu_id: int,
    visible_devices: list[int | str],
    match: str,
) -> None:
    with pytest.raises(RuntimeError, match=match):
        gpu_memory.resolve_visible_device_id(logical_gpu_id, visible_devices)


def test_process_scoped_memory_available_uses_nvml_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeNVML()
    _install_fake_nvml(monkeypatch, fake)

    assert gpu_memory.is_process_scoped_memory_available() is True
    assert fake.shutdown_called is True


def test_process_scoped_memory_unavailable_when_nvml_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_module_not_found(name: str) -> None:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(gpu_memory.importlib, "import_module", _raise_module_not_found)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert gpu_memory.is_process_scoped_memory_available() is False
    assert gpu_memory.get_process_gpu_memory_bytes(0) is None


def test_process_scoped_memory_unavailable_when_nvml_init_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeNVML(init_error=RuntimeError("driver unavailable"))
    _install_fake_nvml(monkeypatch, fake)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert gpu_memory.is_process_scoped_memory_available() is False
    assert gpu_memory.get_process_gpu_memory_bytes(0) is None


def test_get_process_gpu_memory_uses_current_pid_and_visible_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeNVML(
        processes=[
            SimpleNamespace(pid=111, usedGpuMemory=256),
            SimpleNamespace(pid=os.getpid(), usedGpuMemory=1024),
        ]
    )
    _install_fake_nvml(monkeypatch, fake)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    assert gpu_memory.get_process_gpu_memory_bytes(0) == 1024
    assert fake.index_handles == [3]
    assert fake.shutdown_called is True


def test_get_process_gpu_memory_uses_visible_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeNVML(processes=[SimpleNamespace(pid=os.getpid(), usedGpuMemory=2048)])
    _install_fake_nvml(monkeypatch, fake)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc")

    assert gpu_memory.get_process_gpu_memory_bytes(0) == 2048
    assert fake.uuid_handles == ["GPU-abc"]


def test_get_process_gpu_memory_retries_uuid_as_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeNVML(
        processes=[SimpleNamespace(pid=os.getpid(), usedGpuMemory=4096)],
        uuid_requires_bytes=True,
    )
    _install_fake_nvml(monkeypatch, fake)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc")

    assert gpu_memory.get_process_gpu_memory_bytes(0) == 4096
    assert fake.uuid_handles == [b"GPU-abc"]


def test_get_process_gpu_memory_returns_zero_when_pid_not_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeNVML(processes=[SimpleNamespace(pid=os.getpid() + 1, usedGpuMemory=1)])
    _install_fake_nvml(monkeypatch, fake)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert gpu_memory.get_process_gpu_memory_bytes(0) == 0
    assert fake.index_handles == [0]


def test_get_process_gpu_memory_returns_none_on_query_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeNVML(query_error=RuntimeError("driver query failed"))
    _install_fake_nvml(monkeypatch, fake)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert gpu_memory.get_process_gpu_memory_bytes(0) is None


def test_get_gpu_device_info_falls_back_to_torch_when_nvml_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_properties(device_id: int) -> SimpleNamespace:
            assert device_id == 0
            return SimpleNamespace(name="NVIDIA H20", total_memory=96 * 1024**3)

    fake_torch = SimpleNamespace(cuda=_FakeCuda())

    def _import_module(name: str):
        if name == "pynvml":
            raise ModuleNotFoundError(name)
        if name == "torch":
            return fake_torch
        raise AssertionError(name)

    monkeypatch.setattr(gpu_memory.importlib, "import_module", _import_module)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    info = gpu_memory.get_gpu_device_info(0)

    assert info.device_id == 1
    assert info.name == "NVIDIA H20"
    assert info.total_memory_bytes == 96 * 1024**3


def test_get_process_gpu_memory_rejects_invalid_device_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES exposes 1"):
        gpu_memory.get_process_gpu_memory_bytes(1)

    fake = _FakeNVML(device_count=1)
    _install_fake_nvml(monkeypatch, fake)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    with pytest.raises(RuntimeError, match="Only 1 GPU"):
        gpu_memory.get_process_gpu_memory_bytes(2)


def test_format_bytes_gib() -> None:
    assert gpu_memory.format_bytes_gib(None) == "None"
    assert gpu_memory.format_bytes_gib(3 * 1024**3) == "3.00GiB"


def test_calculate_stage_budget_available_bytes_subtracts_accounted_memory() -> None:
    available = gpu_memory.calculate_stage_budget_available_bytes(
        total_memory_bytes=1000,
        accounted_memory_bytes=300,
        memory_fraction=0.5,
    )

    assert available == 200


def test_calculate_stage_budget_available_bytes_rejects_no_headroom() -> None:
    with pytest.raises(RuntimeError, match="stage_load_used"):
        gpu_memory.calculate_stage_budget_available_bytes(
            total_memory_bytes=1000,
            accounted_memory_bytes=500,
            memory_fraction=0.5,
            accounted_memory_label="stage_load_used",
        )


def test_calculate_stage_load_delta_bytes_uses_free_memory_samples() -> None:
    assert gpu_memory.calculate_stage_load_delta_bytes(
        pre_model_load_memory_gib=95.0,
        post_model_load_memory_gib=35.5,
    ) == int(59.5 * 1024**3)


def test_calculate_stage_load_delta_bytes_rejects_memory_growth() -> None:
    with pytest.raises(RuntimeError, match="delta is negative"):
        gpu_memory.calculate_stage_load_delta_bytes(
            pre_model_load_memory_gib=35.0,
            post_model_load_memory_gib=36.0,
        )


def test_gpu_startup_lock_path_uses_visible_device_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,MIG-GPU-deadbeef/1/2")

    first = gpu_memory.get_gpu_startup_lock_path(0, base_dir=tmp_path)
    second = gpu_memory.get_gpu_startup_lock_path(1, base_dir=tmp_path)

    assert first.name == "sglang_omni_gpu_3_startup.lock"
    assert second.name == "sglang_omni_gpu_MIG-GPU-deadbeef_1_2_startup.lock"


def test_gpu_startup_lock_releases_after_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(gpu_memory.tempfile, "gettempdir", lambda: str(tmp_path))

    with pytest.raises(RuntimeError, match="factory failed"):
        with gpu_memory.gpu_startup_lock(0):
            raise RuntimeError("factory failed")

    lock_path = gpu_memory.get_gpu_startup_lock_path(0, base_dir=tmp_path)
    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def test_get_gpu_device_info_reports_name_and_total_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeNVML(device_name=b"NVIDIA H200", total_memory=141 * 1024**3)
    _install_fake_nvml(monkeypatch, fake)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    info = gpu_memory.get_gpu_device_info(0)

    assert info.logical_gpu_id == 0
    assert info.device_id == 3
    assert info.name == "NVIDIA H200"
    assert info.total_memory_bytes == 141 * 1024**3
    assert fake.index_handles == [3]
    assert fake.shutdown_called is True


def test_get_gpu_device_info_returns_unknown_without_nvml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_module_not_found(name: str) -> None:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(gpu_memory.importlib, "import_module", _raise_module_not_found)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    info = gpu_memory.get_gpu_device_info(0)

    assert info.logical_gpu_id == 0
    assert info.device_id == 0
    assert info.name is None
    assert info.total_memory_bytes is None
