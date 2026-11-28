# SPDX-License-Identifier: Apache-2.0
"""Level-Zero ctypes bindings and the export/import context for XPU IPC."""
from __future__ import annotations

import ctypes
import fcntl
import logging
import os
import struct
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)

LEVEL_ZERO_LOADER = "libze_loader.so.1"
ZE_RESULT_SUCCESS = 0
ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES = 0x17
ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES = 0xC
ZE_STRUCTURE_TYPE_CONTEXT_DESC = 0xD
ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC = 0x15
ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC = 0x18
ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF = 0x2
ZE_MAX_IPC_HANDLE_SIZE = 64
ZE_MEMORY_TYPE_DEVICE = 2
DRIVER_DEFAULT_ALIGNMENT = 0
# /proc/<pid>/fd entries for a dma_buf descriptor link to this target.
DMA_BUF_LINK_TARGET_PREFIX = "/dmabuf"


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


class ZeMemoryAllocationProperties(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("memory_type", ctypes.c_int),
        ("identifier", ctypes.c_uint64),
        ("page_size", ctypes.c_uint64),
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
        "zeMemGetAllocProperties": [
            void_pointer,
            void_pointer,
            ctypes.POINTER(ZeMemoryAllocationProperties),
            handle_pointer,
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
        """Allocate exportable, uncompressed device memory a peer card can read.

        A torch.empty pool does not work on B60: under expandable segments its
        handle covers one 20 MiB chunk, and in a sender that sees only its own
        card the driver compresses it, so a peer card's first read fails with
        UR error 39.
        """
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

    def allocation_identifier(self, device_pointer: int) -> int | None:
        """The driver's id for a pointer, or None if this context cannot resolve it.

        A pointer torch allocated belongs to SYCL's context, so resolving one
        here is what shows the driver keeps one allocation map per driver rather
        than per context.
        """
        properties = ZeMemoryAllocationProperties(
            ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES, None, 0, 0, 0
        )
        device = ctypes.c_void_p()
        status = self.library.zeMemGetAllocProperties(
            self.context,
            ctypes.c_void_p(device_pointer),
            ctypes.byref(properties),
            ctypes.byref(device),
        )
        if status != ZE_RESULT_SUCCESS:
            return None
        elif properties.memory_type != ZE_MEMORY_TYPE_DEVICE:
            return None
        else:
            return int(properties.identifier)

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
        try:
            descriptor = dma_buf_descriptor(handle_blob, allocation.segment_bytes)
        except BaseException:
            # The export succeeded, so it has to go back here: the caller has no
            # segment to release once this raises.
            with suppress(LevelZeroError):
                check_status(
                    "zeMemPutIpcHandle",
                    self.library.zeMemPutIpcHandle(self.context, handle),
                )
            raise
        return ExportedSegment(
            base_pointer=allocation.base_pointer,
            segment_bytes=allocation.segment_bytes,
            handle_blob=handle_blob,
            file_descriptor=descriptor,
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


def dma_buf_descriptor(handle_blob: bytes, segment_bytes: int) -> int:
    """The dma_buf descriptor an IPC handle carries in its first four bytes.

    The handle is opaque per spec, so /proc must confirm a dma_buf of the exported size.
    """
    descriptor = struct.unpack_from("<i", handle_blob)[0]
    if descriptor < 0:
        raise LevelZeroError(
            f"Level-Zero IPC handle carries {descriptor} where a dma_buf "
            f"descriptor was expected"
        )
    else:
        pass
    try:
        link_target = os.readlink(f"/proc/self/fd/{descriptor}")
        descriptor_bytes = os.fstat(descriptor).st_size
    except OSError as error:
        raise LevelZeroError(
            f"Level-Zero IPC handle descriptor {descriptor} cannot be inspected: "
            f"{error}"
        ) from error
    if not link_target.startswith(DMA_BUF_LINK_TARGET_PREFIX):
        raise LevelZeroError(
            f"Level-Zero IPC handle descriptor {descriptor} is {link_target!r}, "
            f"not a dma_buf, so the handle layout is not the one this reads"
        )
    elif descriptor_bytes < segment_bytes:
        raise LevelZeroError(
            f"Level-Zero IPC dma_buf descriptor {descriptor} covers "
            f"{descriptor_bytes} bytes, short of the {segment_bytes} exported"
        )
    else:
        return descriptor
