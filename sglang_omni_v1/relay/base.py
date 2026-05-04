# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

import torch

logger = logging.getLogger(__name__)

# Registry mapping relay type names to Relay classes
RELAY_REGISTRY: Dict[str, Type[Relay]] = {}


def register_relay(name: str):
    """
    Decorator to register a Relay subclass in the global registry.

    Usage:
        @register_relay("nccl")
        class NcclRelay(Relay): ...
    """

    def decorator(cls):
        if name in RELAY_REGISTRY:
            logger.warning(f"Relay type '{name}' is already registered. Overwriting!")
        RELAY_REGISTRY[name] = cls
        return cls

    return decorator


def create_relay(relay_type: str, **kwargs) -> Relay:
    """
    Factory function to create a Relay instance based on relay_type.
    Automatically filters kwargs to match the target class __init__ parameters.
    """
    import inspect

    if relay_type not in RELAY_REGISTRY:
        # Try dynamic import to trigger registration
        try:
            if relay_type == "nccl":
                from .nccl import NcclRelay  # noqa
            elif relay_type == "shm":
                from .shm import ShmRelay  # noqa
            elif relay_type == "nixl":
                from .nixl import NixlRelay  # noqa
            elif relay_type == "mooncake":
                from .mooncake import MooncakeRelay  # noqa
        except ImportError:
            pass

    if relay_type not in RELAY_REGISTRY:
        available = list(RELAY_REGISTRY.keys())
        raise ValueError(
            f"Unknown relay type: '{relay_type}'. Available types: {available}"
        )

    relay_cls = RELAY_REGISTRY[relay_type]

    # Filter kwargs to match the target class constructor signature
    sig = inspect.signature(relay_cls.__init__)
    valid_kwargs = {}
    for param_name in sig.parameters:
        if param_name in kwargs:
            valid_kwargs[param_name] = kwargs[param_name]
        elif param_name == "kwargs":
            # If class accepts **kwargs, pass all remaining parameters
            valid_kwargs.update(kwargs)
            break

    return relay_cls(**valid_kwargs)


class RelayOperation(ABC):
    """Abstract handle for asynchronous operations (Put/Get)."""

    @property
    @abstractmethod
    def metadata(self) -> Any:
        """Returns metadata required by the receiver (e.g., SHM name, memory pointer)."""

    @abstractmethod
    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        """Asynchronously waits for the transfer or copy to complete."""


class Relay(ABC):
    """Abstract interface for data transfer backends (SHM, RDMA, etc.)."""

    @abstractmethod
    async def put_async(
        self, tensor: torch.Tensor, request_id: str = None, dst_rank: int = None
    ) -> RelayOperation:
        """
        Asynchronously sends a tensor.
        Returns an operation handle containing metadata for the receiver.
        """

    @abstractmethod
    async def get_async(
        self, metadata: Any, dest_tensor: torch.Tensor, request_id: str = None
    ) -> RelayOperation:
        """
        Asynchronously retrieves data into dest_tensor using provided metadata.
        Returns an operation handle to await completion.
        """

    @abstractmethod
    def cleanup(self, request_id: str) -> None:
        """Cleans up resources for a specific request (e.g., on abort)."""

    @abstractmethod
    def close(self) -> None:
        """Shuts down the relay and releases global resources."""


class CreditAllocator:
    """
    Manages credits for flow control.
    Supports two modes:
    - Simple mode: Returns credit IDs (0, 1, 2, ...) for flow control only
    - Memory mode: Returns memory offsets (0, slot_size, 2*slot_size, ...) for memory pool management
    """

    def __init__(
        self,
        credits: int,
        slot_size: Optional[int] = None,
        base_ptr: Optional[int] = None,
    ):
        self.credits = credits
        self.slot_size = slot_size
        self.base_ptr = base_ptr
        self._free_credits = asyncio.Queue(maxsize=credits)

        # Initialize credits
        for i in range(credits):
            if slot_size is not None:
                # Memory mode: return offsets
                self._free_credits.put_nowait(i * slot_size)
            else:
                # Simple mode: return credit IDs
                self._free_credits.put_nowait(i)

    async def acquire_async(self) -> int:
        """Acquire a credit (blocks if none available)."""
        return await self._free_credits.get()

    def release(self, credit_id: int):
        """Release a credit back to the pool."""
        try:
            self._free_credits.put_nowait(credit_id)
        except asyncio.QueueFull:
            logger.error("Attempted to release credit to a full pool!")
