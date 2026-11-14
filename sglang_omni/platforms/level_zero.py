# SPDX-License-Identifier: Apache-2.0
"""Level-Zero ctypes bindings, IPC probe and the export/import context for XPU.

Two driver behaviours this module depends on are checked at probe time rather
than assumed, because neither is promised by the Level-Zero spec:

- The pool is allocated in this module's own context, while torch reads and
  writes it on SYCL's context. The spec scopes a device pointer to the context
  that allocated it; the Intel driver instead resolves one through a
  driver-wide allocation map. ``torch_reads_private_context_memory`` proves
  both halves of that on the live driver before any edge uses the relay.
- An IPC handle is opaque per spec, so taking the dma_buf descriptor from its
  first four bytes reads a driver layout. ``dma_buf_descriptor`` accepts the
  descriptor only once the kernel agrees it is a dma_buf of at least the
  exported size.

Either check failing leaves the relay unavailable, so the platform keeps every
stage edge on shm. Both go away if torch gains XPU storage sharing
(pytorch/pytorch#191725), which exports through SYCL's own context and handle.
"""
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

from sglang_omni.platforms.xpu_dlpack import wrap_device_bytes

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
# Enough bytes for the probe to cross a cache line; it is freed immediately.
CONTEXT_PROBE_BYTES = 4096
CONTEXT_PROBE_PATTERN = 0xA7


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
    """Whether this driver supports every behaviour the relay needs.

    Beyond seeing torch's devices and exporting a dma_buf, the driver must let
    torch use memory allocated in this module's context, and must still carry a
    dma_buf descriptor in an IPC handle. Both are spec-silent, so both are
    checked here and any failure keeps every stage edge on shm.
    """
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
        pass

    try:
        return torch_reads_private_context_memory() and exports_dma_buf_handles()
    except (AttributeError, OSError, RuntimeError) as error:
        logger.warning(f"Level-Zero IPC disabled: {error}")
        return False


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

    The handle is opaque per spec, so the descriptor is only trusted once the
    kernel confirms it: /proc/self/fd links a dma_buf to "/dmabuf" and sizes it
    at the exported allocation. Anything else means the driver stopped putting a
    dma_buf there, which the caller turns into a fall back to shm rather than a
    descriptor sent to a peer that would map the wrong memory.
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


def torch_reads_private_context_memory() -> bool:
    """Whether torch can use memory this module allocates in its own context.

    The relay allocates the pool here and lets torch copy through it, which the
    spec does not promise across contexts. Both halves are checked on the live
    driver: that this context resolves a pointer torch allocated, and that a
    pattern torch writes through a wrapped pointer reads back. A driver that
    stops supporting either must leave the relay unavailable rather than move
    payloads that silently land in the wrong memory.
    """
    device_index = torch.xpu.current_device()
    context = LevelZeroContext()
    device_pointer = None
    try:
        witness = torch.ones(
            CONTEXT_PROBE_BYTES, dtype=torch.uint8, device=f"xpu:{device_index}"
        )
        if context.allocation_identifier(witness.data_ptr()) is None:
            logger.warning(
                "Level-Zero IPC disabled: this driver does not resolve torch's "
                "allocations from another context, so the pool torch copies "
                "through would not be addressable"
            )
            return False
        else:
            pass
        device_pointer = context.allocate_exportable(device_index, CONTEXT_PROBE_BYTES)
        probe = wrap_device_bytes(device_pointer, CONTEXT_PROBE_BYTES, device_index)
        probe.fill_(CONTEXT_PROBE_PATTERN)
        torch.xpu.synchronize(device_index)
        readback = probe.to("cpu")
        del probe
        if not bool((readback == CONTEXT_PROBE_PATTERN).all()):
            logger.warning(
                "Level-Zero IPC disabled: torch could not read back what it wrote "
                "through memory allocated in this module's context"
            )
            return False
        else:
            return True
    finally:
        try:
            if device_pointer is not None:
                context.free(device_pointer)
            else:
                pass
        finally:
            # The probe owns this context; nothing else would retry its teardown.
            context.destroy()


def exports_dma_buf_handles() -> bool:
    """Whether an IPC handle from this driver still carries a dma_buf descriptor."""
    device_index = torch.xpu.current_device()
    context = LevelZeroContext()
    device_pointer = None
    segment = None
    try:
        device_pointer = context.allocate_exportable(device_index, CONTEXT_PROBE_BYTES)
        segment = context.export_segment(device_pointer)
        return True
    except LevelZeroError as error:
        logger.warning(f"Level-Zero IPC disabled: {error}")
        return False
    finally:
        try:
            if segment is not None:
                context.release_segment(segment)
            else:
                pass
        finally:
            try:
                if device_pointer is not None:
                    context.free(device_pointer)
                else:
                    pass
            finally:
                # The probe owns all three; a caller that falls back to shm has
                # nothing left to retry them with.
                context.destroy()
