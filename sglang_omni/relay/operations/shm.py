# SPDX-License-Identifier: Apache-2.0
"""Shared memory operation classes."""

from typing import Any

from sglang_omni.proto import SHMMetadata
from sglang_omni.relay.operations.base import BaseReadableOperation, BaseReadOperation


class SHMReadableOperation(BaseReadableOperation):
    """Operation object returned by SHMRelay.put(), compatible with NIXLRelay interface.

    Provides:
    - metadata(): Returns the SHMMetadata for the operation
    - wait_for_completion(): No-op for SHM (write is synchronous)
    """

    def __init__(self, shm_metadata: SHMMetadata):
        self._metadata = shm_metadata

    def metadata(self) -> SHMMetadata:
        """Return the SHM metadata for this operation."""
        return self._metadata

    async def wait_for_completion(self) -> None:
        """Wait for the operation to complete. No-op for SHM (synchronous)."""


class SHMReadOperation(BaseReadOperation):
    """Operation object returned by SHMRelay.get(), compatible with NIXLRelay interface.

    Provides:
    - wait_for_completion(): No-op for SHM (read is synchronous)
    - data: The deserialized data (available immediately)
    """

    def __init__(self, data: Any, size: int):
        self.data = data
        self._size = size

    @property
    def size(self) -> int:
        """Return the size of the data in bytes."""
        return self._size

    async def wait_for_completion(self) -> None:
        """Wait for the operation to complete. No-op for SHM (synchronous)."""
