# SPDX-License-Identifier: Apache-2.0
"""Expose raw Intel GPU device memory to torch as a DLPack tensor."""
from __future__ import annotations

import ctypes

import torch

DLPACK_DEVICE_ONEAPI = 14
DLPACK_TYPE_CODE_UINT = 1
DLPACK_UINT8_BITS = 8


class DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


DescriptorParts = tuple[DLManagedTensor, ctypes.Array, ctypes.Array]

# Torch reads a descriptor when its storage dies, so each lives until that deleter.
LIVE_DLPACK_DESCRIPTORS: dict[int, DescriptorParts] = {}

DLPACK_DELETER = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


def forget_dlpack_descriptor(
    descriptor_address: int | None,
    live_descriptors: dict[int, DescriptorParts] = LIVE_DLPACK_DESCRIPTORS,
) -> None:
    # Bound as a default, since module globals may be gone at interpreter shutdown.
    live_descriptors.pop(descriptor_address or 0, None)


FORGET_DLPACK_DESCRIPTOR = DLPACK_DELETER(forget_dlpack_descriptor)
# Never released: torch may free a wrapped storage after this module is torn down.
ctypes.pythonapi.Py_IncRef(ctypes.py_object(FORGET_DLPACK_DESCRIPTOR))


def wrap_device_bytes(
    device_pointer: int,
    num_bytes: int,
    device_index: int,
) -> torch.Tensor:
    """Wrap a Level-Zero device pointer as a uint8 tensor on the pointer's device."""
    shape = (ctypes.c_int64 * 1)(num_bytes)
    strides = (ctypes.c_int64 * 1)(1)
    descriptor = DLManagedTensor()
    descriptor.dl_tensor.data = ctypes.c_void_p(device_pointer)
    descriptor.dl_tensor.device = DLDevice(DLPACK_DEVICE_ONEAPI, device_index)
    descriptor.dl_tensor.ndim = 1
    descriptor.dl_tensor.dtype = DLDataType(DLPACK_TYPE_CODE_UINT, DLPACK_UINT8_BITS, 1)
    descriptor.dl_tensor.shape = shape
    descriptor.dl_tensor.strides = strides
    descriptor.dl_tensor.byte_offset = 0
    descriptor.manager_ctx = None
    descriptor.deleter = ctypes.cast(FORGET_DLPACK_DESCRIPTOR, ctypes.c_void_p).value

    capsule_new = ctypes.pythonapi.PyCapsule_New
    capsule_new.restype = ctypes.py_object
    capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule = capsule_new(ctypes.byref(descriptor), b"dltensor", None)
    tensor = torch.from_dlpack(capsule)
    # A failed conversion never calls the deleter, so register only on success.
    LIVE_DLPACK_DESCRIPTORS[ctypes.addressof(descriptor)] = (descriptor, shape, strides)
    return tensor
