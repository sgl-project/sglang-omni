# SPDX-License-Identifier: Apache-2.0
"""Intel Sysman device metadata and resource counters."""

from __future__ import annotations

from ctypes import byref, c_uint32, c_void_p, cast, pointer
from dataclasses import dataclass
from types import ModuleType
from typing import TypedDict
from uuid import UUID

import torch

try:
    import pyzes
except (ImportError, OSError):
    pyzes = None


class SysmanError(RuntimeError):
    pass


PROCESS_QUERY_ATTEMPTS = 3


class XpuBenchmarkMetadata(TypedDict):
    physical_index: int
    name: str
    uuid: str
    pci_bus_id: str | None
    driver_version: str
    memory_total_bytes: int | None
    gpu_clock_megahertz: float | None
    memory_clock_megahertz: float | None
    power_limits: list[XpuPowerLimit] | None


class XpuPowerLimit(TypedDict):
    domain: int
    level: int
    enabled: bool
    limit_watts: float


def check_sysman_result(result: int, operation: str) -> None:
    if result != 0:
        raise SysmanError(f"{operation} failed with Level Zero status {result:#x}")
    else:
        pass


@dataclass(frozen=True, kw_only=True)
class XpuProcessMemory:
    pid: int
    memory_bytes: int


class SysmanDeviceInfo(TypedDict):
    physical_index: int
    handle: c_void_p
    uuid: str
    name: str
    driver_version: str
    pci_bus_id: str | None


def enumerate_xpu_devices() -> list[SysmanDeviceInfo]:
    if pyzes is None:
        raise SysmanError("pyzes>=0.1.2 and the Level Zero loader are required")
    else:
        pass
    check_sysman_result(pyzes.zesInit(0), "zesInit")
    driver_count = c_uint32()
    check_sysman_result(pyzes.zesDriverGet(byref(driver_count), None), "zesDriverGet")
    drivers = (pyzes.zes_driver_handle_t * driver_count.value)()
    check_sysman_result(
        pyzes.zesDriverGet(byref(driver_count), drivers), "zesDriverGet"
    )
    devices: list[SysmanDeviceInfo] = []
    for driver in drivers:
        device_count = c_uint32()
        check_sysman_result(
            pyzes.zesDeviceGet(driver, byref(device_count), None), "zesDeviceGet"
        )
        handles = (pyzes.zes_device_handle_t * device_count.value)()
        check_sysman_result(
            pyzes.zesDeviceGet(driver, byref(device_count), handles), "zesDeviceGet"
        )
        for handle in handles:
            properties = pyzes.zes_device_properties_t()
            properties.stype = pyzes.ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES
            check_sysman_result(
                pyzes.zesDeviceGetProperties(handle, byref(properties)),
                "zesDeviceGetProperties",
            )
            pci_properties = pyzes.zes_pci_properties_t()
            pci_properties.stype = pyzes.ZES_STRUCTURE_TYPE_PCI_PROPERTIES
            pci_status = pyzes.zesDevicePciGetProperties(handle, byref(pci_properties))
            if pci_status == 0:
                address = pci_properties.address
                pci_bus_id = (
                    f"{address.domain:04x}:{address.bus:02x}:"
                    f"{address.device:02x}.{address.function:x}"
                )
            else:
                pci_bus_id = None
            devices.append(
                {
                    "physical_index": len(devices),
                    "handle": handle,
                    "uuid": str(UUID(bytes=bytes(properties.core.uuid.id))),
                    "name": properties.core.name.decode(errors="replace"),
                    "driver_version": properties.driverVersion.decode(errors="replace"),
                    "pci_bus_id": pci_bus_id,
                }
            )
    return devices


def get_xpu_device_info(
    logical_index: int, torch_module: ModuleType = torch
) -> SysmanDeviceInfo:
    if not 0 <= logical_index < torch_module.xpu.device_count():
        raise ValueError(f"Invalid XPU logical device {logical_index}")
    else:
        pass
    device_uuid = str(torch_module.xpu.get_device_properties(logical_index).uuid)
    for device in enumerate_xpu_devices():
        if device["uuid"] == device_uuid:
            return device
        else:
            pass
    raise SysmanError(
        f"XPU logical device {logical_index} UUID {device_uuid} not found"
    )


def get_xpu_memory_bytes(handle: c_void_p) -> tuple[int, int]:
    """Return free and total device memory, excluding system memory."""
    module_count = c_uint32()
    check_sysman_result(
        pyzes.zesDeviceEnumMemoryModules(handle, byref(module_count), None),
        "zesDeviceEnumMemoryModules",
    )
    handles = (pyzes.zes_mem_handle_t * module_count.value)()
    check_sysman_result(
        pyzes.zesDeviceEnumMemoryModules(handle, byref(module_count), handles),
        "zesDeviceEnumMemoryModules",
    )
    root_memory: list[tuple[int, int]] = []
    tile_memory: list[tuple[int, int]] = []
    for handle in handles:
        properties = pyzes.zes_mem_properties_t()
        properties.stype = pyzes.ZES_STRUCTURE_TYPE_MEM_PROPERTIES
        check_sysman_result(
            pyzes.zesMemoryGetProperties(handle, byref(properties)),
            "zesMemoryGetProperties",
        )
        if properties.location != pyzes.ZES_MEM_LOC_DEVICE:
            continue
        else:
            pass
        state = pyzes.zes_mem_state_t()
        state.stype = pyzes.ZES_STRUCTURE_TYPE_MEM_STATE
        check_sysman_result(
            pyzes.zesMemoryGetState(handle, byref(state)), "zesMemoryGetState"
        )
        memory = tile_memory if properties.onSubdevice else root_memory
        memory.append((int(state.free), int(state.size)))
    memory = root_memory or tile_memory
    if not memory:
        raise SysmanError("No device memory modules reported by Sysman")
    else:
        pass
    return sum(free for free, _ in memory), sum(total for _, total in memory)


def get_xpu_processes(handle: c_void_p) -> list[XpuProcessMemory]:
    for _ in range(PROCESS_QUERY_ATTEMPTS):
        process_count = c_uint32()
        check_sysman_result(
            pyzes.zesDeviceProcessesGetState(handle, byref(process_count), None),
            "zesDeviceProcessesGetState",
        )
        capacity = process_count.value
        processes = (pyzes.zes_process_state_t * capacity)()
        for process in processes:
            process.stype = pyzes.ZES_STRUCTURE_TYPE_PROCESS_STATE
        status = pyzes.zesDeviceProcessesGetState(
            handle, byref(process_count), processes
        )
        if (
            status == pyzes.ZE_RESULT_ERROR_INVALID_SIZE
            or process_count.value > capacity
        ):
            continue
        else:
            check_sysman_result(status, "zesDeviceProcessesGetState")
            return [
                XpuProcessMemory(
                    pid=int(process.processId), memory_bytes=int(process.memSize)
                )
                for process in processes[: process_count.value]
            ]
    raise SysmanError("Process list kept changing during zesDeviceProcessesGetState")


def get_xpu_benchmark_metadata(device: SysmanDeviceInfo) -> XpuBenchmarkMetadata:
    """Read independently optional identity, capacity, clocks, and power limit."""
    try:
        _, memory_total_bytes = get_xpu_memory_bytes(device["handle"])
    except SysmanError:
        memory_total_bytes = None
    try:
        clocks = get_xpu_clocks_megahertz(device["handle"])
    except SysmanError:
        clocks = {}
    try:
        power_limits = get_xpu_power_limits(device["handle"])
    except SysmanError:
        power_limits = None
    return {
        "physical_index": device["physical_index"],
        "name": device["name"],
        "uuid": device["uuid"],
        "pci_bus_id": device["pci_bus_id"],
        "driver_version": device["driver_version"],
        "memory_total_bytes": memory_total_bytes,
        "gpu_clock_megahertz": clocks.get("gpu"),
        "memory_clock_megahertz": clocks.get("memory"),
        "power_limits": power_limits,
    }


def get_xpu_clocks_megahertz(handle: c_void_p) -> dict[str, float]:
    domain_count = c_uint32()
    check_sysman_result(
        pyzes.zesDeviceEnumFrequencyDomains(handle, byref(domain_count), None),
        "zesDeviceEnumFrequencyDomains",
    )
    handles = (pyzes.zes_freq_handle_t * domain_count.value)()
    check_sysman_result(
        pyzes.zesDeviceEnumFrequencyDomains(handle, byref(domain_count), handles),
        "zesDeviceEnumFrequencyDomains",
    )
    clocks: dict[str, float] = {}
    for handle in handles:
        properties = pyzes.zes_freq_properties_t()
        properties.stype = pyzes.ZES_STRUCTURE_TYPE_FREQ_PROPERTIES
        check_sysman_result(
            pyzes.zesFrequencyGetProperties(handle, byref(properties)),
            "zesFrequencyGetProperties",
        )
        if properties.onSubdevice or properties.type not in (
            pyzes.ZES_FREQ_DOMAIN_GPU,
            pyzes.ZES_FREQ_DOMAIN_MEMORY,
        ):
            continue
        else:
            pass
        state = pyzes.zes_freq_state_t()
        state.stype = pyzes.ZES_STRUCTURE_TYPE_FREQ_STATE
        if pyzes.zesFrequencyGetState(handle, byref(state)) == 0 and state.actual >= 0:
            domain = "gpu" if properties.type == pyzes.ZES_FREQ_DOMAIN_GPU else "memory"
            clocks[domain] = float(state.actual)
        else:
            pass
    return clocks


def get_xpu_power_domains(handle: c_void_p) -> list[tuple[c_void_p, int]]:
    power_count = c_uint32()
    check_sysman_result(
        pyzes.zesDeviceEnumPowerDomains(handle, byref(power_count), None),
        "zesDeviceEnumPowerDomains",
    )
    handles = (pyzes.zes_pwr_handle_t * power_count.value)()
    check_sysman_result(
        pyzes.zesDeviceEnumPowerDomains(handle, byref(power_count), handles),
        "zesDeviceEnumPowerDomains",
    )
    domains: list[tuple[c_void_p, int]] = []
    for handle in handles:
        properties = pyzes.zes_power_properties_t()
        properties.stype = pyzes.ZES_STRUCTURE_TYPE_POWER_PROPERTIES
        extension = pyzes.zes_power_ext_properties_t()
        extension.stype = pyzes.ZES_STRUCTURE_TYPE_POWER_EXT_PROPERTIES
        properties.pNext = cast(pointer(extension), c_void_p)
        check_sysman_result(
            pyzes.zesPowerGetProperties(handle, byref(properties)),
            "zesPowerGetProperties",
        )
        if not properties.onSubdevice:
            domains.append((handle, int(extension.domain)))
        else:
            pass
    return domains


def get_xpu_power_limits(handle: c_void_p) -> list[XpuPowerLimit]:
    limits_watts: list[XpuPowerLimit] = []
    for handle, domain in get_xpu_power_domains(handle):
        limit_count = c_uint32()
        check_sysman_result(
            pyzes.zesPowerGetLimitsExt(handle, byref(limit_count), None),
            "zesPowerGetLimitsExt",
        )
        limits = (pyzes.zes_power_limit_ext_desc_t * limit_count.value)()
        for limit in limits:
            limit.stype = pyzes.ZES_STRUCTURE_TYPE_POWER_LIMIT_EXT_DESC
        check_sysman_result(
            pyzes.zesPowerGetLimitsExt(handle, byref(limit_count), limits),
            "zesPowerGetLimitsExt",
        )
        for limit in limits:
            if limit.limitUnit == pyzes.ZES_LIMIT_UNIT_POWER and limit.limit >= 0:
                limits_watts.append(
                    {
                        "domain": domain,
                        "level": int(limit.level),
                        "enabled": bool(limit.enabled),
                        "limit_watts": float(limit.limit) / 1000.0,
                    }
                )
            else:
                pass
    return limits_watts


def get_xpu_activity_microseconds(handle: c_void_p) -> tuple[int, int] | None:
    """Return cumulative active time and its timestamp, both in microseconds."""
    engine_count = c_uint32()
    check_sysman_result(
        pyzes.zesDeviceEnumEngineGroups(handle, byref(engine_count), None),
        "zesDeviceEnumEngineGroups",
    )
    handles = (pyzes.zes_engine_handle_t * engine_count.value)()
    check_sysman_result(
        pyzes.zesDeviceEnumEngineGroups(handle, byref(engine_count), handles),
        "zesDeviceEnumEngineGroups",
    )
    for handle in handles:
        properties = pyzes.zes_engine_properties_t()
        properties.stype = pyzes.ZES_STRUCTURE_TYPE_ENGINE_PROPERTIES
        check_sysman_result(
            pyzes.zesEngineGetProperties(handle, byref(properties)),
            "zesEngineGetProperties",
        )
        if properties.type == pyzes.ZES_ENGINE_GROUP_ALL and not properties.onSubdevice:
            activity = pyzes.zes_engine_stats_t()
            check_sysman_result(
                pyzes.zesEngineGetActivity(handle, byref(activity)),
                "zesEngineGetActivity",
            )
            return int(activity.activeTime), int(activity.timestamp)
        else:
            pass
    return None


def get_xpu_energy_microjoules(handle: c_void_p) -> tuple[int, int] | None:
    """Return cumulative card energy in microjoules and timestamp in microseconds."""
    for power_handle, domain in get_xpu_power_domains(handle):
        if domain == pyzes.ZES_POWER_DOMAIN_CARD:
            energy = pyzes.zes_power_energy_counter_t()
            check_sysman_result(
                pyzes.zesPowerGetEnergyCounter(power_handle, byref(energy)),
                "zesPowerGetEnergyCounter",
            )
            return int(energy.energy), int(energy.timestamp)
        else:
            pass
    return None
