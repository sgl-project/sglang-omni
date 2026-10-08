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


@dataclass(kw_only=True)
class XpuDevice:
    physical_index: int
    handle: c_void_p
    uuid: str
    name: str
    driver_version: str
    pci_bus_id: str | None
    previous_activity: tuple[int, int] | None = None
    previous_energy: tuple[int, int] | None = None

    @classmethod
    def enumerate_devices(cls) -> list[XpuDevice]:
        if pyzes is None:
            raise SysmanError("pyzes>=0.1.2 and the Level Zero loader are required")
        else:
            pass
        check_sysman_result(pyzes.zesInit(0), "zesInit")
        driver_count = c_uint32()
        check_sysman_result(
            pyzes.zesDriverGet(byref(driver_count), None), "zesDriverGet"
        )
        drivers = (pyzes.zes_driver_handle_t * driver_count.value)()
        check_sysman_result(
            pyzes.zesDriverGet(byref(driver_count), drivers), "zesDriverGet"
        )
        devices: list[XpuDevice] = []
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
                pci_status = pyzes.zesDevicePciGetProperties(
                    handle, byref(pci_properties)
                )
                if pci_status == 0:
                    address = pci_properties.address
                    pci_bus_id = (
                        f"{address.domain:04x}:{address.bus:02x}:"
                        f"{address.device:02x}.{address.function:x}"
                    )
                else:
                    pci_bus_id = None
                devices.append(
                    cls(
                        physical_index=len(devices),
                        handle=handle,
                        uuid=str(UUID(bytes=bytes(properties.core.uuid.id))),
                        name=properties.core.name.decode(errors="replace"),
                        driver_version=properties.driverVersion.decode(
                            errors="replace"
                        ),
                        pci_bus_id=pci_bus_id,
                    )
                )
        return devices

    @classmethod
    def from_logical_index(
        cls, logical_index: int, torch_module: ModuleType = torch
    ) -> XpuDevice:
        if not 0 <= logical_index < torch_module.xpu.device_count():
            raise ValueError(f"Invalid XPU logical device {logical_index}")
        else:
            pass
        device_uuid = str(torch_module.xpu.get_device_properties(logical_index).uuid)
        for device in cls.enumerate_devices():
            if device.uuid == device_uuid:
                return device
            else:
                pass
        raise SysmanError(
            f"XPU logical device {logical_index} UUID {device_uuid} not found"
        )

    def memory_bytes(self) -> tuple[int, int]:
        """Return free and total device memory, excluding system memory."""
        module_count = c_uint32()
        check_sysman_result(
            pyzes.zesDeviceEnumMemoryModules(self.handle, byref(module_count), None),
            "zesDeviceEnumMemoryModules",
        )
        handles = (pyzes.zes_mem_handle_t * module_count.value)()
        check_sysman_result(
            pyzes.zesDeviceEnumMemoryModules(self.handle, byref(module_count), handles),
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

    def processes(self) -> list[XpuProcessMemory]:
        for _ in range(PROCESS_QUERY_ATTEMPTS):
            process_count = c_uint32()
            check_sysman_result(
                pyzes.zesDeviceProcessesGetState(
                    self.handle, byref(process_count), None
                ),
                "zesDeviceProcessesGetState",
            )
            capacity = process_count.value
            processes = (pyzes.zes_process_state_t * capacity)()
            for process in processes:
                process.stype = pyzes.ZES_STRUCTURE_TYPE_PROCESS_STATE
            status = pyzes.zesDeviceProcessesGetState(
                self.handle, byref(process_count), processes
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
        raise SysmanError(
            "Process list kept changing during zesDeviceProcessesGetState"
        )

    def benchmark_metadata(self) -> XpuBenchmarkMetadata:
        """Read independently optional identity, capacity, clocks, and power limit."""
        try:
            _, memory_total_bytes = self.memory_bytes()
        except SysmanError:
            memory_total_bytes = None
        try:
            clocks = self.clock_megahertz()
        except SysmanError:
            clocks = {}
        try:
            power_limits = self.power_limits()
        except SysmanError:
            power_limits = None
        return {
            "physical_index": self.physical_index,
            "name": self.name,
            "uuid": self.uuid,
            "pci_bus_id": self.pci_bus_id,
            "driver_version": self.driver_version,
            "memory_total_bytes": memory_total_bytes,
            "gpu_clock_megahertz": clocks.get("gpu"),
            "memory_clock_megahertz": clocks.get("memory"),
            "power_limits": power_limits,
        }

    def clock_megahertz(self) -> dict[str, float]:
        domain_count = c_uint32()
        check_sysman_result(
            pyzes.zesDeviceEnumFrequencyDomains(self.handle, byref(domain_count), None),
            "zesDeviceEnumFrequencyDomains",
        )
        handles = (pyzes.zes_freq_handle_t * domain_count.value)()
        check_sysman_result(
            pyzes.zesDeviceEnumFrequencyDomains(
                self.handle, byref(domain_count), handles
            ),
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
            if (
                pyzes.zesFrequencyGetState(handle, byref(state)) == 0
                and state.actual >= 0
            ):
                domain = (
                    "gpu" if properties.type == pyzes.ZES_FREQ_DOMAIN_GPU else "memory"
                )
                clocks[domain] = float(state.actual)
            else:
                pass
        return clocks

    def power_domains(self) -> list[tuple[c_void_p, int]]:
        power_count = c_uint32()
        check_sysman_result(
            pyzes.zesDeviceEnumPowerDomains(self.handle, byref(power_count), None),
            "zesDeviceEnumPowerDomains",
        )
        handles = (pyzes.zes_pwr_handle_t * power_count.value)()
        check_sysman_result(
            pyzes.zesDeviceEnumPowerDomains(self.handle, byref(power_count), handles),
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

    def power_limits(self) -> list[XpuPowerLimit]:
        limits_watts: list[XpuPowerLimit] = []
        for handle, domain in self.power_domains():
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

    def utilization_percent(self) -> float | None:
        engine_count = c_uint32()
        check_sysman_result(
            pyzes.zesDeviceEnumEngineGroups(self.handle, byref(engine_count), None),
            "zesDeviceEnumEngineGroups",
        )
        handles = (pyzes.zes_engine_handle_t * engine_count.value)()
        check_sysman_result(
            pyzes.zesDeviceEnumEngineGroups(self.handle, byref(engine_count), handles),
            "zesDeviceEnumEngineGroups",
        )
        for handle in handles:
            properties = pyzes.zes_engine_properties_t()
            properties.stype = pyzes.ZES_STRUCTURE_TYPE_ENGINE_PROPERTIES
            check_sysman_result(
                pyzes.zesEngineGetProperties(handle, byref(properties)),
                "zesEngineGetProperties",
            )
            if (
                properties.type == pyzes.ZES_ENGINE_GROUP_ALL
                and not properties.onSubdevice
            ):
                activity = pyzes.zes_engine_stats_t()
                check_sysman_result(
                    pyzes.zesEngineGetActivity(handle, byref(activity)),
                    "zesEngineGetActivity",
                )
                previous = self.previous_activity
                self.previous_activity = (
                    int(activity.activeTime),
                    int(activity.timestamp),
                )
                if previous is None:
                    return None
                else:
                    active_microseconds = activity.activeTime - previous[0]
                    elapsed_microseconds = activity.timestamp - previous[1]
                    if active_microseconds < 0 or elapsed_microseconds <= 0:
                        return None
                    else:
                        return min(
                            100.0, 100.0 * active_microseconds / elapsed_microseconds
                        )
            else:
                pass
        return None

    def power_watts(self) -> float | None:
        for handle, domain in self.power_domains():
            if domain == pyzes.ZES_POWER_DOMAIN_CARD:
                energy = pyzes.zes_power_energy_counter_t()
                check_sysman_result(
                    pyzes.zesPowerGetEnergyCounter(handle, byref(energy)),
                    "zesPowerGetEnergyCounter",
                )
                previous = self.previous_energy
                self.previous_energy = (int(energy.energy), int(energy.timestamp))
                if previous is None:
                    return None
                else:
                    energy_microjoules = energy.energy - previous[0]
                    elapsed_microseconds = energy.timestamp - previous[1]
                    if energy_microjoules < 0 or elapsed_microseconds <= 0:
                        return None
                    else:
                        return energy_microjoules / elapsed_microseconds
            else:
                pass
        return None
