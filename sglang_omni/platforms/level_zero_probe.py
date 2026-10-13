# SPDX-License-Identifier: Apache-2.0
"""Whether this driver supports the Level-Zero IPC relay; any failure keeps shm.

Both behaviours checked here are spec-silent, so the probe proves them on the live
driver. XPU storage sharing in torch (pytorch/pytorch#191725) would retire them.
"""
from __future__ import annotations

import ctypes
import logging
from functools import lru_cache

import torch

from sglang_omni.platforms import level_zero
from sglang_omni.platforms.level_zero import (
    ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF,
    ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES,
    LevelZeroContext,
    LevelZeroError,
    ZeDeviceExternalMemoryProperties,
    check_status,
    load_level_zero,
)
from sglang_omni.platforms.xpu_dlpack import wrap_device_bytes

logger = logging.getLogger(__name__)

# Enough bytes for the probe to cross a cache line; it is freed immediately.
CONTEXT_PROBE_BYTES = 4096
CONTEXT_PROBE_PATTERN = 0xA7


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
    """Whether this driver supports every behaviour the relay needs."""
    if torch.version.xpu is None:
        return False
    else:
        pass

    # Any failure falls back to shm; the driver goes first as it initializes XPU.
    try:
        driver = level_zero.level_zero_driver()
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


def torch_reads_private_context_memory() -> bool:
    """Whether torch can use memory allocated in a private Level-Zero context.

    Checks that the context resolves a torch pointer and that torch reads back what it wrote.
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
