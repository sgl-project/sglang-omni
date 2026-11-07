# SPDX-License-Identifier: Apache-2.0
"""Level-Zero ctypes bindings, IPC probe and the export/import context for XPU."""
from __future__ import annotations

import ctypes
import fcntl
import logging
import os
import struct
from dataclasses import dataclass
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)

LEVEL_ZERO_LOADER = "libze_loader.so.1"
ZE_RESULT_SUCCESS = 0
ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES = 0xC
ZE_STRUCTURE_TYPE_CONTEXT_DESC = 0xD
ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC = 0x15
ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC = 0x18
ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF = 0x2
ZE_MAX_IPC_HANDLE_SIZE = 64
DRIVER_DEFAULT_ALIGNMENT = 0


class LevelZeroError(RuntimeError):
    """A Level-Zero entry point reported a failure status."""


class ZeContextDescriptor(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
    ]


class ZeExternalMemoryExportDescriptor(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
    ]


class ZeDeviceMemAllocDescriptor(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("ordinal", ctypes.c_uint32),
    ]


class ZeDeviceExternalMemoryProperties(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("memoryAllocationImportTypes", ctypes.c_uint32),
        ("memoryAllocationExportTypes", ctypes.c_uint32),
        ("imageImportTypes", ctypes.c_uint32),
        ("imageExportTypes", ctypes.c_uint32),
    ]


class ZeIpcMemHandle(ctypes.Structure):
    _fields_ = [("data", ctypes.c_ubyte * ZE_MAX_IPC_HANDLE_SIZE)]


@dataclass(frozen=True, kw_only=True)
class LevelZeroDriver:
    """The GPU driver handle and its devices in Level-Zero enumeration order."""

    handle: int
    devices: tuple[int, ...]


@lru_cache(maxsize=1)
def load_level_zero() -> ctypes.CDLL:
    """Load and initialize the loader once torch's XPU runtime is up."""
    torch.xpu.init()
    library = ctypes.CDLL(LEVEL_ZERO_LOADER)
    void_pointer = ctypes.c_void_p
    count_pointer = ctypes.POINTER(ctypes.c_uint32)
    handle_pointer = ctypes.POINTER(ctypes.c_void_p)

    argument_types: dict[str, list[type]] = {
        "zeInit": [ctypes.c_uint32],
        "zeDriverGet": [count_pointer, handle_pointer],
        "zeDeviceGet": [void_pointer, count_pointer, handle_pointer],
        "zeContextCreate": [
            void_pointer,
            ctypes.POINTER(ZeContextDescriptor),
            handle_pointer,
        ],
        "zeContextDestroy": [void_pointer],
        "zeDeviceGetExternalMemoryProperties": [
            void_pointer,
            ctypes.POINTER(ZeDeviceExternalMemoryProperties),
        ],
        "zeMemAllocDevice": [
            void_pointer,
            ctypes.POINTER(ZeDeviceMemAllocDescriptor),
            ctypes.c_size_t,
            ctypes.c_size_t,
            void_pointer,
            handle_pointer,
        ],
        "zeMemFree": [void_pointer, void_pointer],
        "zeMemGetAddressRange": [
            void_pointer,
            void_pointer,
            handle_pointer,
            ctypes.POINTER(ctypes.c_size_t),
        ],
        "zeMemGetIpcHandle": [
            void_pointer,
            void_pointer,
            ctypes.POINTER(ZeIpcMemHandle),
        ],
        "zeMemPutIpcHandle": [void_pointer, ZeIpcMemHandle],
        "zeMemOpenIpcHandle": [
            void_pointer,
            void_pointer,
            ZeIpcMemHandle,
            ctypes.c_uint32,
            handle_pointer,
        ],
        "zeMemCloseIpcHandle": [void_pointer, void_pointer],
    }
    for name, types in argument_types.items():
        entry_point = getattr(library, name)
        entry_point.argtypes = types
        entry_point.restype = ctypes.c_int

    status = library.zeInit(0)
    if status != ZE_RESULT_SUCCESS:
        raise LevelZeroError(f"zeInit failed with status 0x{status & 0xFFFFFFFF:x}")
    else:
        pass
    return library


def check_status(entry_point_name: str, status: int) -> None:
    if status != ZE_RESULT_SUCCESS:
        raise LevelZeroError(
            f"{entry_point_name} failed with status 0x{status & 0xFFFFFFFF:x}"
        )
    else:
        pass


@lru_cache(maxsize=1)
def level_zero_driver() -> LevelZeroDriver | None:
    """The first Level-Zero driver and its devices, or None without a driver."""
    library = load_level_zero()
    driver_count = ctypes.c_uint32(0)
    check_status("zeDriverGet", library.zeDriverGet(ctypes.byref(driver_count), None))
    if driver_count.value == 0:
        return None
    else:
        pass
    drivers = (ctypes.c_void_p * driver_count.value)()
    check_status(
        "zeDriverGet", library.zeDriverGet(ctypes.byref(driver_count), drivers)
    )
    device_count = ctypes.c_uint32(0)
    check_status(
        "zeDeviceGet",
        library.zeDeviceGet(drivers[0], ctypes.byref(device_count), None),
    )
    devices = (ctypes.c_void_p * device_count.value)()
    check_status(
        "zeDeviceGet",
        library.zeDeviceGet(drivers[0], ctypes.byref(device_count), devices),
    )
    return LevelZeroDriver(
        handle=int(drivers[0]),
        devices=tuple(int(devices[index]) for index in range(device_count.value)),
    )


def exports_dma_buf(device: int) -> bool:
    library = load_level_zero()
    properties = ZeDeviceExternalMemoryProperties(
        ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES, None, 0, 0, 0, 0
    )
    check_status(
        "zeDeviceGetExternalMemoryProperties",
        library.zeDeviceGetExternalMemoryProperties(
            ctypes.c_void_p(device), ctypes.byref(properties)
        ),
    )
    return bool(
        properties.memoryAllocationExportTypes & ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF
    )


@lru_cache(maxsize=1)
def is_level_zero_ipc_available() -> bool:
    """Whether the driver sees torch's devices and each exports a dma_buf."""
    if torch.version.xpu is None:
        return False
    else:
        pass

    # Any failure falls back to shm; the driver goes first as it initializes XPU.
    try:
        driver = level_zero_driver()
        torch_device_count = torch.xpu.device_count()
        all_export = driver is not None and all(
            exports_dma_buf(device) for device in driver.devices
        )
    except (AttributeError, OSError, RuntimeError) as error:
        logger.info(f"Level-Zero IPC unavailable: {error}")
        return False

    driver_device_count = 0 if driver is None else len(driver.devices)
    if driver_device_count == 0:
        logger.info("Level-Zero IPC unavailable: the driver exposes no devices")
        return False
    elif driver_device_count != torch_device_count:
        logger.warning(
            f"Level-Zero IPC disabled: the driver exposes {driver_device_count} "
            f"devices but torch sees {torch_device_count}, so device indices "
            f"cannot be trusted to match"
        )
        return False
    elif not all_export:
        logger.info("Level-Zero IPC unavailable: a device cannot export a dma_buf")
        return False
    else:
        return True


@dataclass(frozen=True, kw_only=True)
class AddressRange:
    """The allocation a device pointer belongs to, as this process sees it."""

    base_pointer: int
    segment_bytes: int


@dataclass(frozen=True, kw_only=True)
class ExportedSegment:
    """One exported Level-Zero allocation and the descriptor that shares it."""

    base_pointer: int
    segment_bytes: int
    handle_blob: bytes
    file_descriptor: int


class LevelZeroContext:
    """A private Level-Zero context that owns the exported pool and peer mappings."""

    def __init__(self) -> None:
        self.library = load_level_zero()
        driver = level_zero_driver()
        if driver is None:
            raise LevelZeroError("no Level-Zero driver is available")
        else:
            pass
        self.devices = driver.devices
        descriptor = ZeContextDescriptor(ZE_STRUCTURE_TYPE_CONTEXT_DESC, None, 0)
        context = ctypes.c_void_p()
        check_status(
            "zeContextCreate",
            self.library.zeContextCreate(
                ctypes.c_void_p(driver.handle),
                ctypes.byref(descriptor),
                ctypes.byref(context),
            ),
        )
        self.context = context

    def device_handle(self, device_index: int) -> ctypes.c_void_p:
        if device_index >= len(self.devices):
            raise LevelZeroError(
                f"Level-Zero device {device_index} is outside the {len(self.devices)} "
                f"devices this process can see"
            )
        else:
            return ctypes.c_void_p(self.devices[device_index])

    def allocate_exportable(self, device_index: int, num_bytes: int) -> int:
        """Allocate exportable, uncompressed device memory a peer card can read."""
        export = ZeExternalMemoryExportDescriptor(
            ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC,
            None,
            ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF,
        )
        descriptor = ZeDeviceMemAllocDescriptor(
            ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            ctypes.cast(ctypes.pointer(export), ctypes.c_void_p),
            0,
            0,
        )
        pointer = ctypes.c_void_p()
        check_status(
            "zeMemAllocDevice",
            self.library.zeMemAllocDevice(
                self.context,
                ctypes.byref(descriptor),
                num_bytes,
                DRIVER_DEFAULT_ALIGNMENT,
                self.device_handle(device_index),
                ctypes.byref(pointer),
            ),
        )
        return int(pointer.value)

    def free(self, device_pointer: int) -> None:
        check_status(
            "zeMemFree",
            self.library.zeMemFree(self.context, ctypes.c_void_p(device_pointer)),
        )

    def address_range(self, device_pointer: int) -> AddressRange:
        """Find the allocation or peer mapping that a device pointer lies in."""
        base = ctypes.c_void_p()
        segment_bytes = ctypes.c_size_t(0)
        check_status(
            "zeMemGetAddressRange",
            self.library.zeMemGetAddressRange(
                self.context,
                ctypes.c_void_p(device_pointer),
                ctypes.byref(base),
                ctypes.byref(segment_bytes),
            ),
        )
        return AddressRange(
            base_pointer=int(base.value or 0), segment_bytes=int(segment_bytes.value)
        )

    def export_segment(self, device_pointer: int) -> ExportedSegment:
        """Export the allocation that a device pointer belongs to."""
        allocation = self.address_range(device_pointer)
        handle = ZeIpcMemHandle()
        check_status(
            "zeMemGetIpcHandle",
            self.library.zeMemGetIpcHandle(
                self.context,
                ctypes.c_void_p(allocation.base_pointer),
                ctypes.byref(handle),
            ),
        )
        handle_blob = bytes(bytearray(handle.data))
        return ExportedSegment(
            base_pointer=allocation.base_pointer,
            segment_bytes=allocation.segment_bytes,
            handle_blob=handle_blob,
            file_descriptor=struct.unpack_from("<i", handle_blob)[0],
        )

    def release_segment(self, segment: ExportedSegment) -> None:
        handle = ZeIpcMemHandle()
        handle.data = (ctypes.c_ubyte * ZE_MAX_IPC_HANDLE_SIZE)(*segment.handle_blob)
        check_status(
            "zeMemPutIpcHandle",
            self.library.zeMemPutIpcHandle(self.context, handle),
        )

    def open_peer_segment(
        self,
        handle_blob: bytes,
        file_descriptor: int,
        device_index: int,
    ) -> int:
        """Map a peer allocation through its received descriptor, avoiding pidfd."""
        device = self.device_handle(device_index)
        local_descriptor = importable_descriptor(
            file_descriptor, struct.unpack_from("<i", handle_blob)[0]
        )
        try:
            local_blob = bytearray(handle_blob)
            struct.pack_into("<i", local_blob, 0, local_descriptor)
            handle = ZeIpcMemHandle()
            handle.data = (ctypes.c_ubyte * ZE_MAX_IPC_HANDLE_SIZE)(*local_blob)
            peer_pointer = ctypes.c_void_p()
            check_status(
                "zeMemOpenIpcHandle",
                self.library.zeMemOpenIpcHandle(
                    self.context, device, handle, 0, ctypes.byref(peer_pointer)
                ),
            )
        finally:
            if local_descriptor != file_descriptor:
                os.close(local_descriptor)
            else:
                pass
        return int(peer_pointer.value)

    def close_peer_segment(self, peer_pointer: int) -> None:
        check_status(
            "zeMemCloseIpcHandle",
            self.library.zeMemCloseIpcHandle(
                self.context, ctypes.c_void_p(peer_pointer)
            ),
        )

    def destroy(self) -> None:
        if self.context is None:
            return
        else:
            pass
        check_status("zeContextDestroy", self.library.zeContextDestroy(self.context))
        self.context = None


def importable_descriptor(received_descriptor: int, exporter_descriptor: int) -> int:
    """The received descriptor, or a duplicate if it reuses the exporter's number."""
    if received_descriptor != exporter_descriptor:
        return received_descriptor
    else:
        return fcntl.fcntl(received_descriptor, fcntl.F_DUPFD_CLOEXEC, 0)
