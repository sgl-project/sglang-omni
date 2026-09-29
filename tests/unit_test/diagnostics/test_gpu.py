# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from types import ModuleType, SimpleNamespace

from typer.testing import CliRunner

import sglang_omni.diagnostics.gpu as gpu_diagnostics
from sglang_omni.cli import app


class FakeCuda:
    def __init__(self) -> None:
        self.properties = [
            SimpleNamespace(
                name="GPU-A",
                major=12,
                minor=0,
                total_memory=32 * 1024**3,
                uuid="uuid-a",
            ),
            SimpleNamespace(
                name="GPU-B",
                major=12,
                minor=0,
                total_memory=32 * 1024**3,
                uuid="uuid-b",
            ),
        ]

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return len(self.properties)

    def get_device_properties(self, index: int):
        return self.properties[index]


class FakeTorch:
    __version__ = "2.11.0+cu130"
    version = SimpleNamespace(cuda="13.0")

    def __init__(self) -> None:
        self.cuda = FakeCuda()


class FakePlatform:
    """Stand in for the resolved platform so a report describes one arm."""

    def __init__(self, device_type: str) -> None:
        self.device_type = device_type

    def is_cuda_alike(self) -> bool:
        return self.device_type == "cuda"

    def is_cpu(self) -> bool:
        return self.device_type == "cpu"


def pin_platform(monkeypatch, device_type: str) -> None:
    monkeypatch.setattr(gpu_diagnostics, "current_platform", FakePlatform(device_type))


class FakeNVML(ModuleType):
    def __init__(self) -> None:
        super().__init__("pynvml")
        self.shutdown_called = False

    def nvmlInit(self) -> None:
        pass

    def nvmlShutdown(self) -> None:
        self.shutdown_called = True

    def nvmlDeviceGetCount(self) -> int:
        return 2

    def nvmlDeviceGetHandleByIndex(self, index: int) -> str:
        return f"handle:{index}"

    def nvmlDeviceGetMemoryInfo(self, handle: str) -> SimpleNamespace:
        index = int(handle.split(":")[1])
        return SimpleNamespace(total=32 * 1024**3, free=(30 - index) * 1024**3)

    def nvmlDeviceGetPciInfo(self, handle: str) -> SimpleNamespace:
        index = int(handle.split(":")[1])
        return SimpleNamespace(busId=f"00000000:{index + 1:02x}:00.0")

    def nvmlDeviceGetCudaComputeCapability(self, handle: str) -> tuple[int, int]:
        return (12, 0)

    def nvmlDeviceGetUUID(self, handle: str) -> str:
        index = int(handle.split(":")[1])
        return f"GPU-uuid-{'a' if index == 0 else 'b'}"

    def nvmlDeviceGetName(self, handle: str) -> str:
        index = int(handle.split(":")[1])
        return f"GPU-{'A' if index == 0 else 'B'}"

    def nvmlSystemGetDriverVersion(self) -> str:
        return "610.43.02"

    def nvmlSystemGetCudaDriverVersion_v2(self) -> int:
        return 13030


def test_collect_gpu_diagnostics_preserves_reordered_visible_mapping(
    monkeypatch,
) -> None:
    fake_nvml = FakeNVML()
    fake_torch = FakeTorch()
    fake_torch.cuda.properties.reverse()
    monkeypatch.setattr(gpu_diagnostics, "cuda_runtime_version", lambda: "13.3")
    monkeypatch.setattr(gpu_diagnostics, "backend_inventory", lambda: [])
    pin_platform(monkeypatch, "cuda")

    report = gpu_diagnostics.collect_gpu_diagnostics(
        env={"CUDA_VISIBLE_DEVICES": "1,0"},
        torch_module=fake_torch,
        pynvml_module=fake_nvml,
    )

    assert [gpu["logical_index"] for gpu in report["gpus"]] == [0, 1]
    assert [gpu["physical_index"] for gpu in report["gpus"]] == [1, 0]
    assert [gpu["uuid"] for gpu in report["gpus"]] == [
        "GPU-uuid-b",
        "GPU-uuid-a",
    ]
    assert report["environment"]["cuda_visible_devices"] == "1,0"
    assert set(report) == {
        "schema_version",
        "environment",
        "gpus",
        "backends",
        "warnings",
    }
    rendered = gpu_diagnostics.render_gpu_diagnostics(report)
    assert "logical 0 -> physical 1" in rendered
    assert fake_nvml.shutdown_called is True


def test_nvml_inventory_failure_is_isolated_per_physical_device(
    monkeypatch,
) -> None:
    class PartiallyFailingNVML(FakeNVML):
        def nvmlDeviceGetCount(self) -> int:
            return 3

        def nvmlDeviceGetHandleByIndex(self, index: int) -> str:
            if index == 1:
                raise RuntimeError("device is temporarily unavailable")
            return super().nvmlDeviceGetHandleByIndex(index)

    fake_nvml = PartiallyFailingNVML()
    fake_torch = FakeTorch()
    monkeypatch.setattr(gpu_diagnostics, "cuda_runtime_version", lambda: "13.3")
    monkeypatch.setattr(gpu_diagnostics, "backend_inventory", lambda: [])
    pin_platform(monkeypatch, "cuda")

    report = gpu_diagnostics.collect_gpu_diagnostics(
        env={"CUDA_VISIBLE_DEVICES": "0,2"},
        torch_module=fake_torch,
        pynvml_module=fake_nvml,
    )

    assert [gpu["physical_index"] for gpu in report["gpus"]] == [0, 2]
    assert any(
        "physical_index=1" in warning and "device is temporarily unavailable" in warning
        for warning in report["warnings"]
    )
    assert fake_nvml.shutdown_called is True


def test_nvml_inventory_failure_is_isolated_per_device_field(monkeypatch) -> None:
    class PartiallyFailingNVML(FakeNVML):
        def nvmlDeviceGetPciInfo(self, handle: str) -> SimpleNamespace:
            if handle == "handle:0":
                raise RuntimeError("PCI information is unavailable")
            return super().nvmlDeviceGetPciInfo(handle)

    fake_nvml = PartiallyFailingNVML()
    fake_torch = FakeTorch()
    monkeypatch.setattr(gpu_diagnostics, "cuda_runtime_version", lambda: "13.3")
    monkeypatch.setattr(gpu_diagnostics, "backend_inventory", lambda: [])
    pin_platform(monkeypatch, "cuda")

    report = gpu_diagnostics.collect_gpu_diagnostics(
        env={"CUDA_VISIBLE_DEVICES": "0,1"},
        torch_module=fake_torch,
        pynvml_module=fake_nvml,
    )

    assert [gpu["physical_index"] for gpu in report["gpus"]] == [0, 1]
    assert report["gpus"][0]["pci_bus_id"] is None
    assert report["gpus"][0]["free_memory_bytes"] == 30 * 1024**3
    assert report["gpus"][0]["uuid"] == "GPU-uuid-a"
    assert any(
        "PCI query failed for physical_index=0" in warning
        and "PCI information is unavailable" in warning
        for warning in report["warnings"]
    )
    assert fake_nvml.shutdown_called is True


def test_mig_visible_device_emits_unsupported_mapping_warning(monkeypatch) -> None:
    fake_nvml = FakeNVML()
    fake_torch = FakeTorch()
    fake_torch.cuda.properties = [
        SimpleNamespace(
            name="MIG Device",
            major=12,
            minor=0,
            total_memory=10 * 1024**3,
            uuid="MIG-instance-uuid",
        )
    ]
    monkeypatch.setattr(gpu_diagnostics, "cuda_runtime_version", lambda: "13.3")
    monkeypatch.setattr(gpu_diagnostics, "backend_inventory", lambda: [])
    pin_platform(monkeypatch, "cuda")

    report = gpu_diagnostics.collect_gpu_diagnostics(
        env={"CUDA_VISIBLE_DEVICES": "MIG-instance-uuid"},
        torch_module=fake_torch,
        pynvml_module=fake_nvml,
    )

    assert report["gpus"][0]["physical_index"] is None
    assert report["gpus"][0]["free_memory_bytes"] is None
    assert any(
        "MIG device" in warning and "unsupported" in warning
        for warning in report["warnings"]
    )


def test_backend_inventory_reports_installed_but_unimportable(monkeypatch) -> None:
    monkeypatch.setattr(
        gpu_diagnostics,
        "_BACKENDS",
        (("communication", "nixl", "nixl._api"),),
    )
    monkeypatch.setattr(
        gpu_diagnostics,
        "distribution_info",
        lambda module: ("nixl", "1.3.1"),
    )
    monkeypatch.setattr(
        gpu_diagnostics,
        "module_import_error",
        lambda module: "exited with code 1: OSError: libcudart.so",
    )

    backend = gpu_diagnostics.backend_inventory()[0]

    assert backend["installed"] is True
    assert backend["importable"] is False
    assert backend["distribution"] == "nixl"
    assert "failed to import: exited with code 1" in backend["reason"]


def test_backend_inventory_resolves_cuda_variant_distributions(monkeypatch) -> None:
    monkeypatch.setattr(
        gpu_diagnostics,
        "_BACKENDS",
        (
            ("communication", "nixl", "nixl._api"),
            ("communication", "mooncake", "mooncake.engine"),
        ),
    )
    monkeypatch.setattr(gpu_diagnostics, "module_import_error", lambda module: None)
    monkeypatch.setattr(
        gpu_diagnostics.importlib.metadata,
        "packages_distributions",
        lambda: {
            "nixl": ["nixl"],
            "mooncake": ["mooncake-transfer-engine"],
        },
    )
    versions = {
        "nixl": "1.2.0",
        "mooncake-transfer-engine": "0.3.10",
    }
    monkeypatch.setattr(
        gpu_diagnostics.importlib.metadata,
        "version",
        lambda distribution: versions[distribution],
    )

    backends = gpu_diagnostics.backend_inventory()

    assert [(backend["distribution"], backend["version"]) for backend in backends] == [
        ("nixl", "1.2.0"),
        ("mooncake-transfer-engine", "0.3.10"),
    ]
    assert all(backend["installed"] for backend in backends)
    assert all(backend["importable"] for backend in backends)


def test_module_import_probe_survives_hard_crash(tmp_path, monkeypatch) -> None:
    assert gpu_diagnostics.module_import_error("json") is None

    module = tmp_path / "crashing_backend.py"
    module.write_text(
        "import os\nimport signal\nos.kill(os.getpid(), signal.SIGKILL)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    error = gpu_diagnostics.module_import_error("crashing_backend")

    assert error is not None
    assert "terminated by signal" in error


def test_module_import_probe_reports_import_error() -> None:
    error = gpu_diagnostics.module_import_error("_missing_sglang_omni_backend")

    assert error is not None
    assert "ModuleNotFoundError" in error


class FakeXpu:
    """Two cards, keyed by physical index so a mask can be checked by uuid."""

    UUIDS = {0: "uuid-0", 1: "uuid-1", 2: "uuid-2", 5: "uuid-5"}

    def __init__(self, physical_indices: list[int]) -> None:
        self.physical_indices = physical_indices

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return len(self.physical_indices)

    def get_device_properties(self, index: int):
        physical = self.physical_indices[index]
        return SimpleNamespace(
            name="Intel(R) Arc(TM) Pro B60 Graphics",
            total_memory=24 * 1024**3,
            uuid=self.UUIDS[physical],
            driver_version="1.15.38646+6",
        )

    def mem_get_info(self, index: int) -> tuple[int, int]:
        return (20 * 1024**3, 24 * 1024**3)


class FakeXpuTorch:
    __version__ = "2.14.0.dev+xpu"
    version = SimpleNamespace(cuda=None)

    def __init__(self, physical_indices: list[int]) -> None:
        self.cuda = SimpleNamespace(is_available=lambda: False)
        self.xpu = FakeXpu(physical_indices)

    def get_device_module(self, device_type: str):
        return self.xpu


def test_a_non_cuda_accelerator_is_enumerated_through_its_device_module(
    monkeypatch,
) -> None:
    """check-gpu reported no GPU at all on a healthy Intel host, because both
    the inventory and the strict gate went through NVML and torch.cuda.
    """
    monkeypatch.setattr(gpu_diagnostics, "backend_inventory", lambda: [])
    pin_platform(monkeypatch, "xpu")

    report = gpu_diagnostics.collect_gpu_diagnostics(
        env={}, torch_module=FakeXpuTorch([0, 1])
    )

    environment = report["environment"]
    assert environment["device_type"] == "xpu"
    assert environment["visible_devices_variable"] == "ZE_AFFINITY_MASK"
    assert environment["accelerator_available"] is True
    assert environment["cuda_available"] is False
    assert environment["logical_device_count"] == 2
    assert environment["driver_version"] == "1.15.38646+6"
    assert [gpu["name"] for gpu in report["gpus"]] == [
        "Intel(R) Arc(TM) Pro B60 Graphics"
    ] * 2
    assert [gpu["free_memory_bytes"] for gpu in report["gpus"]] == [20 * 1024**3] * 2
    assert report["warnings"] == []
    assert "No xpu devices" not in gpu_diagnostics.render_gpu_diagnostics(report)


def test_a_cpu_platform_reports_no_devices_rather_than_inventing_one(
    monkeypatch,
) -> None:
    """torch.cpu answers is_available and device_count, so enumerating it the
    way an accelerator is enumerated would invent a device with no properties.
    """
    monkeypatch.setattr(gpu_diagnostics, "backend_inventory", lambda: [])
    pin_platform(monkeypatch, "cpu")

    report = gpu_diagnostics.collect_gpu_diagnostics(
        env={}, torch_module=FakeXpuTorch([0, 1])
    )

    assert report["gpus"] == []
    assert report["environment"]["accelerator_available"] is False
    assert report["warnings"] == []


def test_an_affinity_mask_maps_to_physical_cards_in_ascending_order(
    monkeypatch,
) -> None:
    """Level Zero admits the masked cards in ascending order and ignores the
    order they were written in, so ZE_AFFINITY_MASK=5,2 makes physical 2 the
    logical 0. Reading it like CUDA_VISIBLE_DEVICES names the wrong card.
    """
    monkeypatch.setattr(gpu_diagnostics, "backend_inventory", lambda: [])
    pin_platform(monkeypatch, "xpu")

    report = gpu_diagnostics.collect_gpu_diagnostics(
        env={"ZE_AFFINITY_MASK": "5,2"}, torch_module=FakeXpuTorch([2, 5])
    )

    assert [gpu["physical_index"] for gpu in report["gpus"]] == [2, 5]
    assert [gpu["uuid"] for gpu in report["gpus"]] == ["uuid-2", "uuid-5"]


def test_check_gpu_json_output(monkeypatch) -> None:
    check_gpu_module = importlib.import_module("sglang_omni.cli.check_gpu")
    report = {
        "schema_version": 1,
        "environment": {"logical_device_count": 0},
        "gpus": [],
    }
    monkeypatch.setattr(
        check_gpu_module,
        "collect_gpu_diagnostics",
        lambda: report,
    )

    result = CliRunner().invoke(app, ["check-gpu", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == report


def test_check_gpu_strict_fails_on_warning(monkeypatch) -> None:
    check_gpu_module = importlib.import_module("sglang_omni.cli.check_gpu")
    report = {
        "schema_version": 2,
        "environment": {
            "accelerator_available": True,
            "logical_device_count": 1,
        },
        "gpus": [{"logical_index": 0}],
        "backends": [],
        "warnings": ["Installed backend failed to import."],
    }
    monkeypatch.setattr(
        check_gpu_module,
        "collect_gpu_diagnostics",
        lambda: report,
    )

    result = CliRunner().invoke(app, ["check-gpu", "--json", "--strict"])

    assert result.exit_code == 1
    assert json.loads(result.stdout) == report


def test_check_gpu_strict_fails_without_a_visible_accelerator(monkeypatch) -> None:
    check_gpu_module = importlib.import_module("sglang_omni.cli.check_gpu")
    report = {
        "schema_version": 2,
        "environment": {
            "accelerator_available": False,
            "logical_device_count": 0,
        },
        "gpus": [],
        "backends": [],
        "warnings": [],
    }
    monkeypatch.setattr(
        check_gpu_module,
        "collect_gpu_diagnostics",
        lambda: report,
    )

    result = CliRunner().invoke(app, ["check-gpu", "--json", "--strict"])

    assert result.exit_code == 1
    assert json.loads(result.stdout) == report


def test_check_gpu_does_not_load_serving_entrypoints_in_subprocess() -> None:
    script = """
import importlib
import sys
from typer.testing import CliRunner
from sglang_omni.cli import app

check_gpu_module = importlib.import_module("sglang_omni.cli.check_gpu")
check_gpu_module.collect_gpu_diagnostics = lambda: {"schema_version": 1}
result = CliRunner().invoke(app, ["check-gpu", "--json"])
assert result.exit_code == 0, result.output
assert "sglang_omni.serve.launcher" not in sys.modules
assert "sglang_omni.serve.openai_api" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
