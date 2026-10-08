# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import fcntl
import os
import sys
from ctypes import POINTER, Array, Structure, c_uint32, c_uint64, c_void_p, cast
from types import ModuleType, SimpleNamespace

import pytest
import torch

import sglang_omni.utils.gpu_memory as gpu_memory
import sglang_omni.utils.xpu_management as xpu_management
from sglang_omni.utils.gpu_backend import gpu_device_type
from sglang_omni.utils.xpu_management import SysmanError, XpuDevice, XpuProcessMemory

ACCELERATOR_ONLY = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        or (hasattr(torch, "xpu") and torch.xpu.is_available())
    ),
    reason="requires cuda or xpu",
)


@pytest.fixture(autouse=True)
def use_cuda_accounting(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    if request.node.get_closest_marker("accelerator") is None:
        monkeypatch.setattr(gpu_memory, "gpu_device_type", lambda: "cuda")


@pytest.mark.parametrize("device_type", ["cuda", "xpu"])
def test_gpu_backend_detects_available_accelerator(device_type: str) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: device_type == "cuda"),
        xpu=SimpleNamespace(is_available=lambda: device_type == "xpu"),
    )
    assert gpu_device_type(fake_torch) == device_type


class FakeSysmanProcess(Structure):
    _fields_ = [
        ("stype", c_uint32),
        ("processId", c_uint32),
        ("memSize", c_uint64),
    ]


@pytest.mark.parametrize("initial_count", [0, 1])
@pytest.mark.parametrize("invalid_size", [False, True])
def test_xpu_process_query_retries_growth(
    monkeypatch: pytest.MonkeyPatch, initial_count: int, invalid_size: bool
) -> None:
    fetches = 0

    def query(
        handle: c_void_p, count: c_void_p, processes: Array[FakeSysmanProcess] | None
    ) -> int:
        nonlocal fetches
        count_pointer = cast(count, POINTER(c_uint32))
        if processes is None:
            count_pointer.contents.value = initial_count if fetches == 0 else 2
            return 0
        fetches += 1
        count_pointer.contents.value = 2
        if len(processes) < 2:
            return 99 if invalid_size else 0
        processes[0].processId, processes[0].memSize = 123, 2048
        processes[1].processId, processes[1].memSize = 456, 4096
        return 0

    monkeypatch.setattr(
        xpu_management,
        "pyzes",
        SimpleNamespace(
            zes_process_state_t=FakeSysmanProcess,
            ZES_STRUCTURE_TYPE_PROCESS_STATE=1,
            ZE_RESULT_ERROR_INVALID_SIZE=99,
            zesDeviceProcessesGetState=query,
        ),
    )
    device = XpuDevice(
        physical_index=0,
        handle=c_void_p(1),
        uuid="a",
        name="Intel",
        driver_version="1",
        pci_bus_id=None,
    )
    assert device.processes() == [
        XpuProcessMemory(pid=123, memory_bytes=2048),
        XpuProcessMemory(pid=456, memory_bytes=4096),
    ]
    assert fetches == 2


def test_xpu_process_query_does_not_report_zero_for_an_unstable_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def query(
        handle: c_void_p, count: c_void_p, processes: Array[FakeSysmanProcess] | None
    ) -> int:
        cast(count, POINTER(c_uint32)).contents.value = 0 if processes is None else 1
        return 0

    monkeypatch.setattr(
        xpu_management,
        "pyzes",
        SimpleNamespace(
            zes_process_state_t=FakeSysmanProcess,
            ZES_STRUCTURE_TYPE_PROCESS_STATE=1,
            ZE_RESULT_ERROR_INVALID_SIZE=99,
            zesDeviceProcessesGetState=query,
        ),
    )
    device = XpuDevice(
        physical_index=0,
        handle=c_void_p(1),
        uuid="a",
        name="Intel",
        driver_version="1",
        pci_bus_id=None,
    )
    with pytest.raises(SysmanError, match="kept changing"):
        device.processes()


class FakeNVML(ModuleType):
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


def install_fake_nvml(monkeypatch: pytest.MonkeyPatch, fake: FakeNVML) -> None:
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
    fake = FakeNVML()
    install_fake_nvml(monkeypatch, fake)

    assert gpu_memory.is_process_scoped_memory_available() is True
    assert fake.shutdown_called is True


def test_process_scoped_memory_unavailable_when_nvml_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_module_not_found(name: str) -> None:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(gpu_memory.importlib, "import_module", raise_module_not_found)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert gpu_memory.is_process_scoped_memory_available() is False
    assert gpu_memory.get_process_gpu_memory_bytes(0) is None


def test_process_scoped_memory_unavailable_when_nvml_init_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeNVML(init_error=RuntimeError("driver unavailable"))
    install_fake_nvml(monkeypatch, fake)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert gpu_memory.is_process_scoped_memory_available() is False
    assert gpu_memory.get_process_gpu_memory_bytes(0) is None


def test_get_process_gpu_memory_uses_current_pid_and_visible_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeNVML(
        processes=[
            SimpleNamespace(pid=111, usedGpuMemory=256),
            SimpleNamespace(pid=os.getpid(), usedGpuMemory=1024),
        ]
    )
    install_fake_nvml(monkeypatch, fake)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    assert gpu_memory.get_process_gpu_memory_bytes(0) == 1024
    assert fake.index_handles == [3]
    assert fake.shutdown_called is True


def test_get_process_gpu_memory_uses_visible_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeNVML(processes=[SimpleNamespace(pid=os.getpid(), usedGpuMemory=2048)])
    install_fake_nvml(monkeypatch, fake)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc")

    assert gpu_memory.get_process_gpu_memory_bytes(0) == 2048
    assert fake.uuid_handles == ["GPU-abc"]


def test_get_process_gpu_memory_retries_uuid_as_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeNVML(
        processes=[SimpleNamespace(pid=os.getpid(), usedGpuMemory=4096)],
        uuid_requires_bytes=True,
    )
    install_fake_nvml(monkeypatch, fake)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc")

    assert gpu_memory.get_process_gpu_memory_bytes(0) == 4096
    assert fake.uuid_handles == [b"GPU-abc"]


def test_get_process_gpu_memory_returns_zero_when_pid_not_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeNVML(processes=[SimpleNamespace(pid=os.getpid() + 1, usedGpuMemory=1)])
    install_fake_nvml(monkeypatch, fake)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert gpu_memory.get_process_gpu_memory_bytes(0) == 0
    assert fake.index_handles == [0]


def test_get_process_gpu_memory_returns_none_on_query_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeNVML(query_error=RuntimeError("driver query failed"))
    install_fake_nvml(monkeypatch, fake)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert gpu_memory.get_process_gpu_memory_bytes(0) is None


def test_xpu_memory_uses_current_process_and_ignores_cuda_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_memory, "gpu_device_type", lambda: "xpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    device = XpuDevice(
        physical_index=3,
        handle=c_void_p(4),
        uuid="uuid-a",
        name="Intel GPU",
        driver_version="1",
        pci_bus_id=None,
    )
    monkeypatch.setattr(XpuDevice, "from_logical_index", lambda index: device)
    monkeypatch.setattr(
        XpuDevice,
        "processes",
        lambda self: [
            XpuProcessMemory(pid=os.getpid(), memory_bytes=4096),
            XpuProcessMemory(pid=os.getpid() + 1, memory_bytes=8192),
        ],
    )
    monkeypatch.setattr(XpuDevice, "memory_bytes", lambda self: (8192, 16384))

    def refuse_nvml() -> None:
        raise AssertionError("XPU must not query NVML")

    monkeypatch.setattr(gpu_memory, "try_import_pynvml", refuse_nvml)

    assert gpu_memory.is_process_scoped_memory_available()
    assert gpu_memory.get_process_gpu_memory_bytes(0) == 4096
    metadata = gpu_memory.get_gpu_device_info(0)
    assert metadata.name == "Intel GPU"
    assert metadata.device_id == "uuid-a"
    assert metadata.total_memory_bytes == 16384
    monkeypatch.setattr(XpuDevice, "processes", lambda self: [])
    assert gpu_memory.get_process_gpu_memory_bytes(0) == 0


def test_xpu_unavailable_management_keeps_memory_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_memory, "gpu_device_type", lambda: "xpu")
    monkeypatch.setattr(xpu_management, "pyzes", None)
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.xpu, "get_device_properties", lambda index: SimpleNamespace(uuid="uuid-a")
    )
    assert not gpu_memory.is_process_scoped_memory_available()
    assert gpu_memory.get_process_gpu_memory_bytes(0) is None
    with pytest.raises(gpu_memory.InvalidGpuDeviceError, match="Invalid XPU"):
        gpu_memory.get_process_gpu_memory_bytes(1)


def test_xpu_mapping_uses_uuid_instead_of_sysman_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices = [
        XpuDevice(
            physical_index=index,
            handle=c_void_p(index + 1),
            uuid=uuid,
            name="Intel GPU",
            driver_version="1",
            pci_bus_id=None,
        )
        for index, uuid in enumerate(("uuid-b", "uuid-a"))
    ]
    monkeypatch.setattr(XpuDevice, "enumerate_devices", lambda: devices)
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.xpu, "get_device_properties", lambda index: SimpleNamespace(uuid="uuid-a")
    )
    assert XpuDevice.from_logical_index(0).physical_index == 1
    monkeypatch.setattr(
        torch.xpu,
        "get_device_properties",
        lambda index: SimpleNamespace(uuid="unknown"),
    )
    with pytest.raises(SysmanError, match="not found"):
        XpuDevice.from_logical_index(0)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.xpu.is_available(), reason="requires XPU and Sysman")
def test_process_memory_reports_real_xpu_allocation() -> None:
    pytest.importorskip("pyzes")
    allocation = torch.ones(1024 * 1024, dtype=torch.float32, device="xpu:0")
    torch.xpu.synchronize()
    memory_bytes = gpu_memory.get_process_gpu_memory_bytes(0)
    assert memory_bytes is not None
    assert memory_bytes >= allocation.numel() * allocation.element_size()


def test_get_gpu_device_info_falls_back_to_torch_when_nvml_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_properties(device_id: int) -> SimpleNamespace:
            assert device_id == 0
            return SimpleNamespace(name="NVIDIA H20", total_memory=96 * 1024**3)

    fake_torch = SimpleNamespace(get_device_module=lambda: FakeCuda())

    def import_module(name: str):
        if name == "pynvml":
            raise ModuleNotFoundError(name)
        if name == "torch":
            return fake_torch
        raise AssertionError(name)

    monkeypatch.setattr(gpu_memory.importlib, "import_module", import_module)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    info = gpu_memory.get_gpu_device_info(0)

    assert info.device_id == 1
    assert info.name == "NVIDIA H20"
    assert info.total_memory_bytes == 96 * 1024**3


@pytest.mark.accelerator
@ACCELERATOR_ONLY
def test_device_info_reports_real_memory_on_this_accelerator() -> None:
    """The live half, where an accelerator is actually present."""
    info = gpu_memory.get_gpu_device_info(0)

    assert info.name
    assert info.total_memory_bytes is not None and info.total_memory_bytes > 0


def test_get_process_gpu_memory_rejects_invalid_device_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES exposes 1"):
        gpu_memory.get_process_gpu_memory_bytes(1)

    fake = FakeNVML(device_count=1)
    install_fake_nvml(monkeypatch, fake)
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
    fake = FakeNVML(device_name=b"NVIDIA H200", total_memory=141 * 1024**3)
    install_fake_nvml(monkeypatch, fake)
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
    def raise_module_not_found(name: str) -> None:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(gpu_memory.importlib, "import_module", raise_module_not_found)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    info = gpu_memory.get_gpu_device_info(0)

    assert info.logical_gpu_id == 0
    assert info.device_id == 0
    assert info.name is None
    assert info.total_memory_bytes is None
