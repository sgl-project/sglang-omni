# SPDX-License-Identifier: Apache-2.0
"""CUDA IPC handle helpers for weight sharing."""

# Note (guozhihao): use UntypedStorage._share_cuda_ / _new_shared_cuda;
# do not revive raw OpenMemHandle + _new_with_weak_ptr (segfaults).

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import torch

_libcuda: ctypes.CDLL | None = None
_cu_mem_get_address_range = None


def _ensure_driver() -> ctypes.CDLL:
    global _libcuda, _cu_mem_get_address_range
    if _libcuda is not None:
        return _libcuda
    lib = ctypes.CDLL("libcuda.so.1")
    lib.cuInit.argtypes = [ctypes.c_uint]
    lib.cuInit.restype = ctypes.c_int
    err = lib.cuInit(0)
    if err != 0:
        raise RuntimeError(f"cuInit failed with error {err}")
    lib.cuMemGetAddressRange_v2.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
    ]
    lib.cuMemGetAddressRange_v2.restype = ctypes.c_int
    _libcuda = lib
    _cu_mem_get_address_range = lib.cuMemGetAddressRange_v2
    return lib


def get_allocation_range(ptr: int) -> tuple[int, int]:
    """Return ``(base, size_bytes)`` for the CUDA allocation containing ``ptr``."""
    _ensure_driver()
    assert _cu_mem_get_address_range is not None
    base = ctypes.c_void_p()
    size = ctypes.c_size_t()
    err = _cu_mem_get_address_range(
        ctypes.byref(base), ctypes.byref(size), ctypes.c_void_p(ptr)
    )
    if err != 0:
        raise RuntimeError(f"cuMemGetAddressRange_v2 failed with error {err}")
    if base.value is None:
        raise RuntimeError("cuMemGetAddressRange_v2 returned null base")
    return int(base.value), int(size.value)


def allocation_offset_bytes(tensor: torch.Tensor) -> int:
    """Byte offset of ``tensor.data_ptr()`` within its CUDA allocation block."""
    if not tensor.is_cuda:
        raise ValueError("allocation_offset_bytes requires a CUDA tensor")
    ptr = tensor.data_ptr()
    base, _ = get_allocation_range(ptr)
    return ptr - base


@dataclass(frozen=True)
class SharedCudaStorage:
    device_index: int
    handle: bytes
    storage_size_bytes: int
    storage_offset_bytes: int
    ref_counter_handle: bytes
    ref_counter_offset: int
    event_handle: bytes
    event_sync_required: bool


def export_storage(tensor: torch.Tensor) -> SharedCudaStorage:
    """Export the underlying CUDA storage of ``tensor`` for IPC."""
    if not tensor.is_cuda:
        raise ValueError("export_storage requires a CUDA tensor")
    storage = tensor.untyped_storage()
    (
        device_index,
        handle,
        storage_size_bytes,
        storage_offset_bytes,
        ref_counter_handle,
        ref_counter_offset,
        event_handle,
        event_sync_required,
    ) = storage._share_cuda_()
    return SharedCudaStorage(
        device_index=int(device_index),
        handle=handle,
        storage_size_bytes=int(storage_size_bytes),
        storage_offset_bytes=int(storage_offset_bytes),
        ref_counter_handle=ref_counter_handle,
        ref_counter_offset=int(ref_counter_offset),
        event_handle=event_handle,
        event_sync_required=bool(event_sync_required),
    )


def open_storage(shared: SharedCudaStorage) -> torch.UntypedStorage:
    """Open an exported CUDA storage in the importing process."""
    torch.cuda._lazy_init()
    return torch.UntypedStorage._new_shared_cuda(
        shared.device_index,
        shared.handle,
        shared.storage_size_bytes,
        shared.storage_offset_bytes,
        shared.ref_counter_handle,
        shared.ref_counter_offset,
        shared.event_handle,
        shared.event_sync_required,
    )


def rebuild_tensor(
    *,
    storage: torch.UntypedStorage,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    tensor_offset: int,
    device_index: int,
) -> torch.Tensor:
    """Rebuild a tensor view over an opened shared storage."""
    typed = torch.storage.TypedStorage(
        wrap_storage=storage, dtype=dtype, _internal=True
    )
    out = torch.empty(0, dtype=dtype, device=torch.device("cuda", device_index))
    out.set_(typed, int(tensor_offset), tuple(shape), tuple(stride))
    return out


def dtype_from_name(name: str) -> torch.dtype:
    if not name.startswith("torch."):
        raise ValueError(f"unsupported dtype name: {name}")
    dtype = getattr(torch, name.removeprefix("torch."), None)
    if dtype is None or not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported dtype name: {name}")
    return dtype


def dtype_to_name(dtype: torch.dtype) -> str:
    return str(dtype)
