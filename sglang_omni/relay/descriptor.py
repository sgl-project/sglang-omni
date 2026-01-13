# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Descriptor classes for NIXL-based RDMA operations.

This module contains Descriptor, Device, DeviceKind, and SerializedDescriptor classes
extracted from nixl_connect.py for better code organization.

NOTE: This code is copied from:
https://github.com/ai-dynamo/dynamo/blob/c94d097a1d08f6c064e213f1647327cae69937d2/lib/bindings/python/src/dynamo/nixl_connect/__init__.py

The original code has been adapted and reorganized for use in this project.
"""

from __future__ import annotations

import ctypes
import logging
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, ConfigDict, field_validator

try:
    import torch
except ImportError as e:
    raise ImportError(
        "PyTorch must be installed to use this module. Please install PyTorch, ex: 'pip install torch'."
    ) from e

try:
    import nixl._bindings as nixl_bindings

    NIXL_AVAILABLE = True
except ImportError:
    nixl_bindings = None
    NIXL_AVAILABLE = False
    logger.warning(
        "NIXL not available - Descriptor will work with limited functionality"
    )

# Handle forward reference to Connection
if TYPE_CHECKING:
    from sglang_omni.relay.nixl.connector import Connection

try:
    import cupy as array_module
except ImportError:
    try:
        import numpy as array_module
    except ImportError as e:
        raise ImportError("Numpy or CuPy must be installed to use this module.") from e

logger = logging.getLogger(__name__)


class DeviceKind(IntEnum):
    """
    Type of memory a descriptor has been allocated to.
    """

    UNSPECIFIED = 0

    HOST = 1
    """
    System (CPU) memory.
    """

    CUDA = 2
    """
    CUDA addressable device (GPU) memory.
    """

    def __str__(self) -> str:
        if self == DeviceKind.HOST:
            return "cpu"
        elif self == DeviceKind.CUDA:
            return "cuda"
        else:
            return "<invalid>"


class Device:
    """
    Represents a device in the system.
    """

    def __init__(
        self,
        metadata: str | tuple[DeviceKind, int],
    ) -> None:
        if metadata is None:
            raise ValueError("Argument `metadata` cannot be `None`.")
        if (
            isinstance(metadata, tuple)
            and len(metadata) == 2
            and isinstance(metadata[0], DeviceKind)
            and isinstance(metadata[1], int)
        ):
            kind, device_id = metadata
        elif isinstance(metadata, str):
            metadata = metadata.strip().lower()
            if metadata.startswith("cuda") or metadata.startswith("gpu"):
                kind = DeviceKind.CUDA
                device_id = (
                    0 if metadata.find(":") == -1 else int(metadata.split(":")[1])
                )
            elif metadata.startswith("cpu") or metadata.startswith("host"):
                kind = DeviceKind.HOST
                device_id = 0
            else:
                raise ValueError(
                    "Argument `metadata` must be in the format 'cuda:<device_id>' or 'cpu'."
                )
        else:
            raise TypeError(
                "Argument `metadata` must be a `tuple[MemoryKind, int]` or a `str`."
            )

        self._device_id = device_id
        self._kind = kind

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(kind={self._kind}, id={self._device_id})"

    def __str__(self) -> str:
        return (
            f"{self._kind}:{self._device_id}"
            if self._kind is DeviceKind.CUDA
            else f"{self._kind}"
        )

    @property
    def id(self) -> int:
        """
        Gets the device ID of the device.
        """
        return self._device_id

    @property
    def kind(self) -> DeviceKind:
        """
        Gets the memory kind of the device.
        """
        return self._kind


class SerializedDescriptor(BaseModel):
    """
    Pydantic serialization type for memory descriptors.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    device: str = "cpu"
    ptr: int = 0
    size: int = 0

    def to_descriptor(self) -> "Descriptor":
        """
        Deserialize the serialized descriptor into a `Descriptor` object.
        """
        return Descriptor(data=(self.ptr, self.size, self.device, None))

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        if not isinstance(v, str):
            raise TypeError("Argument `device` must be `str`.")
        v = v.strip().lower()
        if not (v.startswith("cuda") or v == "cpu"):
            raise ValueError(
                "Argument `device` must be one of 'cpu' or 'cuda:<device_id>'."
            )
        return v

    @field_validator("ptr")
    @classmethod
    def validate_ptr(cls, v: int) -> int:
        if v == 0:
            raise ValueError("Argument `ptr` cannot be zero (aka `null` or `None`).")
        return v

    @field_validator("size")
    @classmethod
    def validate_size(cls, v: int) -> int:
        if v < 0:
            raise ValueError(
                "Argument `size` must be an integer greater than or equal to zero."
            )
        return v


class Descriptor:
    """
    Memory descriptor that ensures memory is registered w/ NIXL, used for transferring data between workers.
    """

    def __init__(
        self,
        data: (
            torch.Tensor
            | tuple[array_module.ndarray, Device | str]
            | bytes
            | tuple[int, int, Device | str, Any]
        ),
    ) -> None:
        """
        Memory descriptor for transferring data between workers.

        Parameters
        ----------
        data : torch.Tensor | tuple[ndarray, Device|str] | bytes | tuple[int, int, Device|str, Any]
            The data to be transferred.

            When `torch.Tensor` is provided, the attributes of the tensor will be used to create the descriptor.

            When `tuple[ndarray, Device]` is provided, the tuple must contain:
            - `ndarray`: The CuPy or NumPy array to be transferred.
            - `Device`: Either a `Device` or a string representing the device type (e.g., "cuda" or "cpu").

            When `bytes` is provided, the pointer and size derived from the bytes object and memory type will be assumed to be CPU.

            When `tuple[int, int, Device|str, Any]` is provided, the tuple must contain the following elements:
            - `int`: Pointer to the data in memory.
            - `int`: Size of the data in bytes.
            - `Device`: Either a `Device` or a string representing the device type (e.g., "cuda" or "cpu").
            - `Any`: Optional reference to the data (e.g., the original tensor or bytes object).
                     This is useful for keeping a reference to the data in memory, but it is not required.

        Raises
        ------
        ValueError
            When `data` is `None`.
        TypeError
            When `data` is not a valid type (i.e., not `torch.Tensor`, `bytes`, or a valid tuple).
        TypeError
            When `data` is a tuple but the elements are not of the expected types (i.e., [`ndarray`, `Device|str`] OR [`int`, `int`, `Device|str`, `Any`]).
        """
        TYPE_ERROR_MESSAGE = "Argument `data` must be `torch.Tensor`, `tuple[ndarray, Device|str]`, `bytes`, or `tuple[int, int, Device|str, Any]`."
        if data is None:
            raise ValueError("Argument `data` cannot be `None`.")
        if not (
            isinstance(data, torch.Tensor)
            or isinstance(data, (bytes, bytearray))
            or isinstance(data, tuple)
        ):
            raise TypeError(TYPE_ERROR_MESSAGE)

        self._data_device: Device = Device("cpu")
        self._data_ptr: int = 0
        self._data_ref: Optional[Any] = None
        self._data_size: int = 0

        # Member fields for managing NIXL memory registration.
        # Note: ONLY local descriptors should be registered with NIXL,
        #      remote descriptors do not have a valid memory address and registration will fault.

        self._connection: Optional["Connection"] = None
        self._nixl_hndl: Optional[nixl_bindings.nixlRegDList] = None

        # Initially `None` cached serialized descriptor reference, populated when `get_metadata()` is called.
        self._serialized: Optional[SerializedDescriptor] = None

        # Data is `torch.Tensor`.
        if isinstance(data, torch.Tensor):
            self._data_ptr = data.data_ptr()
            self._data_size = data.numel() * data.element_size()
            if data.is_cuda:
                self._data_device = Device((DeviceKind.CUDA, data.get_device()))
            self._data_ref = data

            logger.debug(
                f"sglang_omni.relay.descriptor.{self.__class__.__name__}: Created {self.__repr__()} from `torch.Tensor`."
            )

        # Data is `tuple[ndarray, Device]`.
        elif (
            isinstance(data, tuple)
            and len(data) == 2
            and isinstance(data[0], array_module.ndarray)
            and (isinstance(data[1], Device) or isinstance(data[1], str))
        ):
            if hasattr(data[0], "__array_interface__"):
                self._data_ptr = data[0].__array_interface__["data"][0]
            elif hasattr(data[0], "__cuda_array_interface__"):
                self._data_ptr = data[0].__cuda_array_interface__["data"][0]
            else:
                raise TypeError(
                    "Argument `data[0]` must be a `ndarray` with a valid array interface."
                )
            self._data_size = data[0].nbytes
            self._data_device = (
                data[1] if isinstance(data[1], Device) else Device(data[1])
            )
            self._data_ref = data[0]

            logger.debug(
                f"sglang_omni.relay.descriptor.{self.__class__.__name__}: Created {self.__repr__()} from `tuple[ndarray, Device|str]`."
            )

        # Data is `bytes`.
        elif isinstance(data, (bytes, bytearray)):
            (self._data_ptr, self._data_size) = self._buffer_to_ptr_size(data)
            self._data_ref = data

            logger.debug(
                f"sglang_omni.relay.descriptor.{self.__class__.__name__}: Created {self.__repr__()} from `bytes`."
            )

        # Data is `tuple[int, int, Device, dtype, tuple, Any]`.
        elif (
            isinstance(data, tuple)
            and len(data) >= 2
            and isinstance(data[0], int)
            and isinstance(data[1], int)
        ):
            if len(data) >= 3 and not (
                isinstance(data[2], Device) or isinstance(data[2], str)
            ):
                raise TypeError(
                    "Argument `data` must be a `tuple[int, int, Device|str, Any]`."
                )

            self._data_ptr = data[0]
            self._data_size = data[1]
            if len(data) >= 3:
                self._data_device = (
                    data[2] if isinstance(data[2], Device) else Device(data[2])
                )
            self._data_ref = data[3] if len(data) >= 4 else None

            logger.debug(
                f"sglang_omni.relay.descriptor.{self.__class__.__name__}: Created {self.__repr__()} from `tuple[int, int, Device|str, Any]`."
            )
        else:
            raise TypeError(TYPE_ERROR_MESSAGE)

    def __del__(self) -> None:
        if not (self._nixl_hndl is None or self._connection is None):
            # Deregister the memory with NIXL.
            self._connection._nixl.deregister_memory(self._nixl_hndl)
            self._nixl_hndl = None
            self._connection = None

        if self._data_ref is not None:
            # Release the reference to the data.
            del self._data_ref

        logger.debug(
            f"sglang_omni.relay.descriptor.{self.__class__.__name__}: Deleted {self.__repr__()}."
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self})"

    def __str__(self) -> str:
        return f"ptr={hex(self._data_ptr)}, size={self._data_size}, device={self._data_device}"

    @property
    def device(self) -> Device:
        """
        Gets the device the of the descriptor.
        """
        return self._data_device

    @property
    def is_registered(self) -> bool:
        """
        Gets whether the descriptor is registered with NIXL.
        """
        return self._connection is not None and self._nixl_hndl is not None

    @property
    def ptr(self) -> int:
        """
        Gets the pointer of the descriptor.
        """
        return self._data_ptr

    @property
    def size(self) -> int:
        """
        Gets the size of the descriptor.
        """
        return self._data_size

    @staticmethod
    def from_serialized(
        serialized: SerializedDescriptor,
    ) -> "Descriptor":
        """
        Deserializes a `SerializedDescriptor` into a `Descriptor` object.

        Parameters
        ----------
        serialized : SerializedDescriptor
            The serialized descriptor to deserialize.

        Returns
        -------
        Descriptor
            The deserialized descriptor.
        """
        if not isinstance(serialized, SerializedDescriptor):
            raise TypeError("Argument `serialized` must be `SerializedDescriptor`.")

        return serialized.to_descriptor()

    @property
    def metadata(self) -> SerializedDescriptor:
        """
        Serializes the descriptor into a `SerializedDescriptor` object.
        """
        if self._serialized is None:
            self._serialized = SerializedDescriptor(
                device=f"{self._data_device}",
                ptr=self._data_ptr,
                size=self._data_size,
            )  # type: ignore[operator]

        return self._serialized

    def deregister_with_connector(self, connection: "Connection") -> None:
        """
        Deregisters the memory of the descriptor with NIXL.
        """
        from sglang_omni.relay.nixl.connector import Connection

        if not isinstance(connection, Connection):
            raise TypeError(
                "Argument `connection` must be `sglang_omni.relay.nixl.connector.Connection`."
            )
        if connection != self._connection:
            raise RuntimeError(
                "Descriptor can only be deregistered from the connection it was registered with. "
                f"Existing connection: {self._connection.name if self._connection is not None else None}, requested connection: {connection.name}."
            )
            return

        if self._nixl_hndl is None:
            logger.warning(
                f"sglang_omni.relay.descriptor.{self.__class__.__name__}: Request to deregister Descriptor {self.__repr__()} cannot be completed because the Descriptor is not registered."
            )
            return

        connection._nixl.deregister_memory(self._nixl_hndl)
        self._nixl_hndl = None
        self._connection = None
        logger.debug(
            f"sglang_omni.relay.descriptor.{self.__class__.__name__}: Deregistered {self.__repr__()} with NIXL."
        )

    def register_with_connector(
        self,
        connection: "Connection",
    ) -> None:
        """
        Registers the memory of the descriptor with NIXL.
        """
        from sglang_omni.relay.nixl.connector import Connection

        if not isinstance(connection, Connection):
            raise TypeError(
                "Argument `connection` must be `sglang_omni.relay.nixl.connector.Connection`."
            )
        if self._data_ptr == 0:
            raise ValueError("Cannot register memory with a null pointer.")
        if self._connection is not None:
            if self._connection != connection:
                raise RuntimeError(
                    "Descriptor cannot be registered with more than one connection. "
                    f"Existing connection: {self._connection.name}, new connection: {connection.name}."
                )
            # Descriptor is already registered with this connection.
            return

        # When the descriptor is already registered with NIXL, just return.
        if self._nixl_hndl is not None:
            return

        # Register the memory with NIXL.
        self._connection = connection

        if isinstance(self._data_ref, torch.Tensor):
            self._nixl_hndl = connection._nixl.register_memory(self._data_ref)
        else:
            mem_type = str(self._data_device.kind)
            reg_list = [
                (self._data_ptr, self._data_size, self._data_device.id, mem_type)
            ]
            self._nixl_hndl = connection._nixl.register_memory(reg_list, mem_type)

        logger.debug(
            f"sglang_omni.relay.descriptor.{self.__class__.__name__}: Registered {self.__repr__()} with NIXL."
        )

    def _buffer_to_ptr_size(
        self,
        data: bytes | bytearray,
    ) -> tuple[int, int]:
        """
        Returns the memory address of the underlying data of a bytes or bytearray object as well as its size.

        Parameters
        ----------
        data : bytes | bytearray
            The bytes or bytearray object to get the data pointer and size from.

        Returns
        -------
        tuple[int, int]
            A tuple containing the memory address of the underlying data and its size.
        """
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("Argument `data` must be `bytes` or `bytearray`.")

        cptr = ctypes.c_char_p()
        size = ctypes.c_size_t()

        # Request the pointer to the underlying data and the buffer's size from ctypes.
        try:
            ret = ctypes.pythonapi.PyObject_AsCharBuffer(
                ctypes.py_object(data),
                ctypes.byref(cptr),
                ctypes.byref(size),
            )
            # Ensure the call was successful and the pointer is valid.
            if ret != 0 or cptr.value is None:
                raise RuntimeError(
                    f"ctypes.pythonapi.PyObject_AsCharBuffer failed (error: {ret})."
                )
            # The resulting pointer is a `char*`, cast it to a `void*` to that ctypes will provide the address.
            # `c_char_p.value` returns a `bytes`` object instead of the pointer address;
            # whereas `c_void_p.value` returns the actual pointer address as an `int`.
            vptr = ctypes.cast(cptr, ctypes.c_void_p)

            return (0, 0) if vptr.value is None else (vptr.value, size.value)
        except Exception as e:
            raise RuntimeError(
                f"Failed to get memory address of the underlying data and size of `{type(data).__name__}` object via ctypes inspection."
            ) from e
