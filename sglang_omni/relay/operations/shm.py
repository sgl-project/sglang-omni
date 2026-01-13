# SPDX-License-Identifier: Apache-2.0
"""Shared memory operation classes."""

from sglang_omni.relay.nixl import SHMMetadata
from sglang_omni.relay.operations.base import BaseReadableOperation, BaseReadOperation


class SHMReadableOperation(BaseReadableOperation):
    """Operation object returned by SHMRelay.put(), compatible with NIXLRelay interface.

    Provides:
    - metadata(): Returns the SHMMetadata for the operation
    - wait_for_completion(): No-op for SHM (write is synchronous)
    """

    def __init__(self, metadata: SHMMetadata):
        self._metadata = metadata

    def metadata(self) -> SHMMetadata:
        """Return the SHM metadata for this operation."""
        return self._metadata

    async def wait_for_completion(self) -> None:
        """Wait for the operation to complete. No-op for SHM (synchronous)."""


class SHMReadOperation(BaseReadOperation):
    """Operation object returned by SHMRelay.get(), compatible with NIXLRelay interface.

    Provides:
    - wait_for_completion(): No-op for SHM (read is synchronous)
    """

    def __init__(self, size: int = 0):
        """Initialize SHMReadOperation.

        Args:
            size: Total size of the data in bytes
        """
        self._size = size

    @property
    def size(self) -> int:
        """Return the size of the data in bytes."""
        return self._size

    async def wait_for_completion(self) -> None:
        """Wait for the operation to complete. No-op for SHM (synchronous)."""
