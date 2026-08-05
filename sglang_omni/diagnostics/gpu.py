# SPDX-License-Identifier: Apache-2.0
"""Model-free GPU and backend diagnostics."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import subprocess
import sys
from collections.abc import Mapping
from typing import Any

from sglang_omni.platforms import Platform, resolve_current_platform
from sglang_omni.utils.gpu_memory import (
    _decode_nvml_string,
    _shutdown_nvml,
    _try_import_pynvml,
    format_bytes_gib,
)

_BACKENDS = (
    ("attention", "flash-attn-4", "flash_attn.cute"),
    ("attention", "flashinfer", "flashinfer"),
    ("attention", "triton", "triton"),
    ("attention", "torch-sdpa", "torch.nn.functional"),
    ("gemm", "sgl-deep-gemm", "deep_gemm"),
    ("gemm", "sglang-kernel", "sgl_kernel"),
    ("gemm", "aiter", "aiter"),
    ("moe", "quack-kernels", "quack"),
    ("quantization", "torchao", "torchao"),
    (
        "quantization",
        "compressed-tensors",
        "compressed_tensors",
    ),
    ("communication", "nixl", "nixl._api"),
    ("communication", "mooncake", "mooncake.engine"),
)
_IMPORT_PROBE_TIMEOUT_SECONDS = 30.0
_IMPORT_PROBE_CODE = "import importlib, sys; importlib.import_module(sys.argv[1])"


def _cuda_version(value: int | None) -> str | None:
    if not value:
        return None
    return f"{value // 1000}.{(value % 1000) // 10}"


def _normalize_uuid(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    return normalized.removeprefix("gpu-") or None


def _probe_output(*values: str | bytes | None) -> str | None:
    for value in values:
        if value:
            text = (
                value.decode("utf-8", errors="replace")
                if isinstance(value, bytes)
                else value
            ).strip()
            if text:
                return text[-2000:]
    return None


def _module_import_error(module: str) -> str | None:
    try:
        result = subprocess.run(
            [sys.executable, "-c", _IMPORT_PROBE_CODE, module],
            check=False,
            capture_output=True,
            text=True,
            timeout=_IMPORT_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        detail = _probe_output(exc.stderr, exc.stdout)
        reason = f"timed out after {_IMPORT_PROBE_TIMEOUT_SECONDS:g}s"
        return f"{reason}: {detail}" if detail else reason

    if result.returncode == 0:
        return None

    reason = (
        f"terminated by signal {-result.returncode}"
        if result.returncode < 0
        else f"exited with code {result.returncode}"
    )
    detail = _probe_output(result.stderr, result.stdout)
    return f"{reason}: {detail}" if detail else reason


def _distribution_info(module: str) -> tuple[str | None, str | None]:
    package = module.partition(".")[0]
    try:
        distributions = importlib.metadata.packages_distributions().get(package, ())
    except Exception:
        return None, None

    for distribution in sorted(distributions):
        try:
            return distribution, importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:
            return distribution, None
    return None, None


def _backend_inventory() -> list[dict[str, Any]]:
    backends = []
    for category, name, module in _BACKENDS:
        import_error = _module_import_error(module)
        distribution, version = _distribution_info(module)
        importable = import_error is None
        installed = distribution is not None or importable
        reason = None
        if import_error is not None:
            if distribution is not None:
                reason = (
                    f"Distribution {distribution!r} is installed, but module "
                    f"{module!r} failed to import: {import_error}"
                )
            else:
                reason = f"Module {module!r} failed to import: {import_error}"
        backends.append(
            {
                "category": category,
                "name": name,
                "distribution": distribution,
                "version": version,
                "module": module,
                "installed": installed,
                "importable": importable,
                "reason": reason,
            }
        )
    return backends


def _cuda_runtime_version() -> str | None:
    try:
        runtime = importlib.import_module("cuda.bindings.runtime")
        status, version = runtime.cudaRuntimeGetVersion()
        return _cuda_version(int(version)) if int(status) == 0 else None
    except Exception:
        return None


def _nvml_inventory(
    pynvml: Any | None,
) -> tuple[list[dict[str, Any]], dict[str, str | None], list[str]]:
    system = {"driver_version": None, "cuda_driver_api_version": None}
    inventory: list[dict[str, Any]] = []
    warnings: list[str] = []
    if pynvml is None:
        return inventory, system, warnings

    try:
        pynvml.nvmlInit()
    except Exception as exc:
        return inventory, system, [f"NVML initialization failed: {exc}"]

    try:
        system["driver_version"] = _decode_nvml_string(
            pynvml.nvmlSystemGetDriverVersion()
        )
    except Exception as exc:
        warnings.append(f"NVML driver query failed: {exc}")
    try:
        system["cuda_driver_api_version"] = _cuda_version(
            int(pynvml.nvmlSystemGetCudaDriverVersion_v2())
        )
    except Exception as exc:
        warnings.append(f"NVML CUDA driver query failed: {exc}")

    try:
        count = int(pynvml.nvmlDeviceGetCount())
    except Exception as exc:
        warnings.append(f"NVML device count query failed: {exc}")
        return inventory, system, warnings

    for physical_index in range(count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
        except Exception as exc:
            warnings.append(
                f"NVML device handle query failed for physical_index="
                f"{physical_index}: {exc}"
            )
            continue

        device = {
            "physical_index": physical_index,
            "uuid": None,
            "pci_bus_id": None,
            "name": None,
            "compute_capability": None,
            "total_memory_bytes": None,
            "free_memory_bytes": None,
        }
        try:
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            device["total_memory_bytes"] = int(memory.total)
            device["free_memory_bytes"] = int(memory.free)
        except Exception as exc:
            warnings.append(
                f"NVML memory query failed for physical_index={physical_index}: {exc}"
            )
        try:
            pci = pynvml.nvmlDeviceGetPciInfo(handle)
            device["pci_bus_id"] = _decode_nvml_string(pci.busId)
        except Exception as exc:
            warnings.append(
                f"NVML PCI query failed for physical_index={physical_index}: {exc}"
            )
        try:
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            device["compute_capability"] = f"{int(major)}.{int(minor)}"
        except Exception as exc:
            warnings.append(
                f"NVML compute capability query failed for physical_index="
                f"{physical_index}: {exc}"
            )
        try:
            device["uuid"] = _decode_nvml_string(pynvml.nvmlDeviceGetUUID(handle))
        except Exception as exc:
            warnings.append(
                f"NVML UUID query failed for physical_index={physical_index}: {exc}"
            )
        try:
            device["name"] = _decode_nvml_string(pynvml.nvmlDeviceGetName(handle))
        except Exception as exc:
            warnings.append(
                f"NVML name query failed for physical_index={physical_index}: {exc}"
            )
        inventory.append(device)
    return inventory, system, warnings


def _physical_device(
    logical_index: int,
    properties: Any | None,
    visible_devices: list[int | str],
    by_index: dict[int, dict[str, Any]],
    by_uuid: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    torch_uuid = _normalize_uuid(getattr(properties, "uuid", None))
    if torch_uuid in by_uuid:
        return by_uuid[torch_uuid]

    if logical_index < len(visible_devices):
        visible = visible_devices[logical_index]
        if isinstance(visible, int):
            return by_index.get(visible, {})
        return by_uuid.get(_normalize_uuid(visible), {})

    return by_index.get(logical_index, {}) if not visible_devices else {}


def _logical_devices(
    platform: Platform,
    visible_devices: list[int | str],
    inventory: list[dict[str, Any]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    by_index = {device["physical_index"]: device for device in inventory}
    by_uuid = {
        uuid: device
        for device in inventory
        if (uuid := _normalize_uuid(device.get("uuid"))) is not None
    }
    devices = []
    for logical_index in range(platform.device_count()):
        try:
            properties = platform.get_device_properties(logical_index)
        except Exception as exc:
            warnings.append(f"PyTorch GPU {logical_index} query failed: {exc}")
            properties = None
        visible_device = (
            visible_devices[logical_index]
            if logical_index < len(visible_devices)
            else logical_index
        )
        physical = _physical_device(
            logical_index, properties, visible_devices, by_index, by_uuid
        )
        if (
            isinstance(visible_device, str)
            and visible_device.upper().startswith("MIG-")
            and not physical
        ):
            warnings.append(
                f"CUDA_VISIBLE_DEVICES entry {visible_device!r} is a MIG device; "
                "physical GPU mapping and free memory are unsupported."
            )
        architecture = None
        torch_cc = (
            f"{properties.major}.{properties.minor}" if properties is not None else None
        )
        architecture = (
            getattr(properties, "gcnArchName", None) if properties is not None else None
        )
        devices.append(
            {
                "logical_index": logical_index,
                "visible_device": visible_device,
                "physical_index": physical.get("physical_index"),
                "uuid": physical.get("uuid")
                or str(getattr(properties, "uuid", "") or "")
                or None,
                "pci_bus_id": physical.get("pci_bus_id"),
                "name": physical.get("name") or getattr(properties, "name", None),
                "compute_capability": physical.get("compute_capability") or torch_cc,
                "architecture": architecture,
                "total_memory_bytes": physical.get("total_memory_bytes")
                or getattr(properties, "total_memory", None),
                "free_memory_bytes": physical.get("free_memory_bytes"),
            }
        )
    return devices


def collect_gpu_diagnostics(
    *,
    env: Mapping[str, str] | None = None,
    torch_module: Any | None = None,
    pynvml_module: Any | None = None,
) -> dict[str, Any]:
    """Collect diagnostics without loading model configuration or weights."""

    source_env = os.environ if env is None else env
    torch = torch_module or importlib.import_module("torch")
    device_platform = resolve_current_platform(torch)

    platform = device_platform.device_name
    platform_visible_value = device_platform.visible_device_value(source_env)
    visible_devices = device_platform.visible_devices(source_env)
    # Importing/querying NVML in a ROCm environment can surface an unrelated
    # host NVIDIA driver and produce a misleading physical-device inventory.
    # NVML metadata is meaningful only for an NVIDIA PyTorch build.
    pynvml = (
        pynvml_module
        if pynvml_module is not None
        else (_try_import_pynvml() if device_platform.is_cuda() else None)
    )

    inventory, system, warnings = _nvml_inventory(pynvml)
    try:
        devices = _logical_devices(
            device_platform, visible_devices, inventory, warnings
        )
    finally:
        if pynvml is not None:
            _shutdown_nvml(pynvml)

    backends = _backend_inventory()
    warnings.extend(
        backend["reason"]
        for backend in backends
        if backend["installed"] and not backend["importable"]
    )
    return {
        "schema_version": 1,
        "environment": {
            "device_control_env_var": device_platform.device_control_env_var,
            "visible_devices": platform_visible_value,
            "accelerator_platform": platform,
            **system,
            "cuda_runtime_version": (
                _cuda_runtime_version() if device_platform.is_cuda() else None
            ),
            "pytorch_version": getattr(torch, "__version__", None),
            "pytorch_cuda_build": getattr(
                getattr(torch, "version", None), "cuda", None
            ),
            "pytorch_hip_build": getattr(getattr(torch, "version", None), "hip", None),
            "accelerator_available": device_platform.device_count() > 0,
            "logical_device_count": len(devices),
        },
        "gpus": devices,
        "backends": backends,
        "warnings": warnings,
    }


def render_gpu_diagnostics(report: Mapping[str, Any]) -> str:
    """Render a compact diagnostic summary for terminal output."""

    environment = report["environment"]
    control_env_var = environment.get("device_control_env_var")
    visible = environment.get("visible_devices")
    platform = environment.get("accelerator_platform", "cuda")
    is_amd = platform == "rocm"
    lines = [
        "SGLang-Omni GPU diagnostics (no model loaded)",
        f"Accelerator platform: {platform}",
        (
            f"{control_env_var or 'Device visibility'}: "
            f"{visible if visible is not None else '<unset>'}"
        ),
        f"Driver: {environment['driver_version'] or 'unavailable'}",
        (
            "CUDA driver/runtime: "
            f"{environment['cuda_driver_api_version'] or 'unavailable'} / "
            f"{environment['cuda_runtime_version'] or 'unavailable'}"
        ),
        (
            f"PyTorch/{'ROCm' if is_amd else 'CUDA'} build: "
            f"{environment['pytorch_version'] or 'unavailable'} / "
            f"{(environment.get('pytorch_hip_build') if is_amd else environment['pytorch_cuda_build']) or 'unavailable'}"
        ),
        "GPUs:",
    ]
    if not report["gpus"]:
        lines.append("  No accelerator devices are visible to PyTorch.")
    for device in report["gpus"]:
        lines.append(
            f"  logical {device['logical_index']} -> physical "
            f"{device['physical_index']} visible={device['visible_device']} "
            f"name={device['name'] or 'unknown'} "
            f"arch={device.get('architecture') or 'unknown'} "
            f"cc={device['compute_capability'] or 'unknown'} "
            f"memory={format_bytes_gib(device['free_memory_bytes'])}/"
            f"{format_bytes_gib(device['total_memory_bytes'])} "
            f"uuid={device['uuid'] or 'unknown'} "
            f"pci={device['pci_bus_id'] or 'unknown'}"
        )

    lines.append("Backends:")
    for backend in report["backends"]:
        status = (
            "available"
            if backend["importable"]
            else (
                "installed, module unavailable"
                if backend["installed"]
                else "not installed"
            )
        )
        lines.append(
            f"  {backend['category']}/{backend['name']}: {status}; "
            f"version={backend['version'] or '-'}"
        )

    if report["warnings"]:
        lines.append("Warnings:")
        lines.extend(f"  {warning}" for warning in report["warnings"])
    return "\n".join(lines)
