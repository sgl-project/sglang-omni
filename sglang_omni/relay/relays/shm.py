# SPDX-License-Identifier: Apache-2.0
"""Shared memory relay for inter-stage data transfer."""

import logging
import pickle
from multiprocessing import shared_memory as _shm
from typing import Any

from sglang_omni.relay.descriptor import Descriptor
from sglang_omni.relay.nixl import SHMMetadata
from sglang_omni.relay.operations.shm import SHMReadableOperation, SHMReadOperation
from sglang_omni.relay.relays.base import Relay

logger = logging.getLogger(__name__)


def shm_write_buffer(buffer: bytes | memoryview, size: int) -> SHMMetadata:
    """Write buffer (bytes or memoryview) into SharedMemory and return metadata.

    This is a unified function that can handle both bytes and numpy arrays efficiently.

    Args:
        buffer: Buffer to write (bytes, bytearray, or memoryview of numpy array)
        size: Size of data to write in bytes

    Returns:
        SHMMetadata: Metadata for the created shared memory segment
    """
    shm = _shm.SharedMemory(create=True, size=size)
    try:
        shm_mv = memoryview(shm.buf)
        # Direct copy from buffer to shared memory (efficient)
        shm_mv[:size] = (
            buffer[:size]
            if isinstance(buffer, memoryview)
            else memoryview(buffer)[:size]
        )
        del shm_mv
        meta = SHMMetadata(name=shm.name, size=size)
        try:
            shm.close()
        except Exception as e:
            logger.debug("Failed to close shared memory: %s", e)
        return meta
    except Exception:
        # Cleanup on error
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass
        raise


def shm_read_to_descriptors(meta: SHMMetadata, descriptors: list[Descriptor]) -> None:
    """Read data from shared memory directly into descriptor buffers (zero-copy).

    Supports both single segment (legacy) and multiple segments (new format).
    Each descriptor receives data from a corresponding SHM segment.

    Args:
        meta: SHM metadata with segment info. If shm_segments is provided, uses
              multi-segment mode; otherwise uses legacy single-segment mode.
        descriptors: Pre-allocated buffers to receive data. Must match number
                     of segments in multi-segment mode.

    Raises:
        ValueError: If descriptor count/size doesn't match segments.
    """
    # Read from each segment into corresponding descriptor
    for desc, segment_info in zip(descriptors, meta.shm_segments):
        segment_name = segment_info["name"]
        segment_size = segment_info["size"]

        shm = _shm.SharedMemory(name=segment_name)
        try:
            shm_mv = memoryview(shm.buf)
            copy_size = min(desc.size, segment_size)

            # Copy data into descriptor buffer
            if hasattr(desc, "_data_ref") and desc._data_ref is not None:
                buffer = desc._data_ref
                try:
                    import numpy as np

                    if isinstance(buffer, np.ndarray):
                        # numpy array - use memoryview for efficient copy
                        buffer_mv = memoryview(buffer)
                        buffer_mv[:copy_size] = shm_mv[:copy_size]
                    else:
                        # bytes-like object
                        buffer_bytes = shm_mv[:copy_size].tobytes()
                        if isinstance(buffer, bytearray):
                            buffer[:copy_size] = buffer_bytes
                        else:
                            import ctypes

                            ctypes.memmove(
                                ctypes.addressof(ctypes.c_char.from_buffer(buffer, 0)),
                                buffer_bytes,
                                copy_size,
                            )
                except ImportError:
                    import ctypes

                    buffer_bytes = shm_mv[:copy_size].tobytes()
                    if isinstance(buffer, bytearray):
                        buffer[:copy_size] = buffer_bytes
                    else:
                        ctypes.memmove(
                            ctypes.addressof(ctypes.c_char.from_buffer(buffer, 0)),
                            buffer_bytes,
                            copy_size,
                        )
            else:
                # Use memory pointer directly
                import ctypes

                buffer_bytes = shm_mv[:copy_size].tobytes()
                ctypes.memmove(desc.ptr, buffer_bytes, copy_size)

            del shm_mv
        finally:
            # Cleanup shared memory segment
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except Exception:
                pass


class SHMRelay(Relay):
    """Shared memory relay for inter-stage data transfer.

    This relay uses Python's multiprocessing.shared_memory for
    transferring pickle-serialized Python objects between stages.

    Interface is designed to be compatible with NIXLRelay:
    - put(descriptors) -> SHMReadableOperation (with .metadata() method)
    - get(metadata, descriptors) -> SHMReadOperation (with .wait_for_completion() method)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize SHM relay.

        Args:
            config: Optional configuration dict.
                - threshold_bytes: Size threshold for using SHM vs inline.
                                   0 means always use SHM (default).
        """
        config = config or {}
        self.threshold = config.get("threshold_bytes", 0)
        self._pending_segments: dict[str, list[SHMMetadata]] = (
            {}
        )  # request_id -> [metadata]
        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "errors": 0,
        }

    def put(self, descriptors: list[Descriptor]) -> SHMReadableOperation:
        """Put descriptors into shared memory.

        Serializes data from each descriptor and writes to separate shared memory segments.
        Supports multiple descriptors, with metadata structure compatible with RdmaMetadata.

        Parameters
        ----------
        descriptors : list[Descriptor]
            List of Descriptor objects containing tensor/data to transfer.
            Each descriptor will be written to a separate shared memory segment.

        Returns
        -------
        SHMReadableOperation
            Operation object with metadata() and wait_for_completion() methods.
            Metadata contains descriptors list (compatible with RdmaMetadata format).
        """
        if not descriptors:
            raise ValueError("descriptors cannot be empty")

        try:
            # Serialize each descriptor and write to separate SHM segments
            serialized_descriptors = []
            shm_segments = []
            total_size = 0

            for desc in descriptors:
                # Get serialized descriptor metadata
                serialized_desc = desc.metadata  # Returns SerializedDescriptor
                serialized_descriptors.append(serialized_desc)

                # Extract payload from descriptor buffer and write to shared memory
                # Use unified shm_write_buffer for both numpy arrays and bytes (efficient)
                import numpy as np

                if isinstance(desc._data_ref, np.ndarray):
                    # Direct write: numpy array → shared memory (only 1 copy via memoryview)
                    buffer_mv = memoryview(desc._data_ref[: desc.size])
                    segment_meta = shm_write_buffer(buffer_mv, desc.size)
                else:
                    # Fallback: convert to bytes first
                    payload_bytes = (
                        desc._data_ref.tobytes()[: desc.size]
                        if hasattr(desc._data_ref, "tobytes")
                        else pickle.dumps(desc._data_ref)
                    )
                    segment_meta = shm_write_buffer(payload_bytes, len(payload_bytes))
                shm_segments.append(
                    {"name": segment_meta.name, "size": segment_meta.size}
                )
                total_size += segment_meta.size

            # Create metadata with descriptors list (compatible with RdmaMetadata)
            meta = SHMMetadata(
                descriptors=serialized_descriptors,
                shm_segments=shm_segments,
            )

            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += total_size

            logger.debug(
                "SHMRelay put: %d descriptors, total_size=%d, segments=%d",
                len(descriptors),
                total_size,
                len(shm_segments),
            )

            return SHMReadableOperation(meta)

        except Exception as e:
            self._metrics["errors"] += 1
            logger.error("SHMRelay put failed: %s", e)
            raise

    def get(
        self, metadata: SHMMetadata, descriptors: list[Descriptor]
    ) -> SHMReadOperation:
        """Get data from shared memory.

        Reads from shared memory into descriptor buffers (zero-copy).

        Parameters
        ----------
        metadata : SHMMetadata
            Metadata from the put operation (via readable_op.metadata())
        descriptors : list[Descriptor]
            List of Descriptor objects with pre-allocated buffers.
            Must not be empty.

        Returns
        -------
        SHMReadOperation
            Operation object with wait_for_completion() method.
        """
        if not descriptors:
            raise ValueError(
                "descriptors cannot be empty. Descriptors are required for zero-copy data transfer."
            )

        try:
            # Read directly into descriptor buffers
            shm_read_to_descriptors(metadata, descriptors)

            # Calculate total size from segments or legacy size
            if metadata.shm_segments:
                total_size = sum(seg["size"] for seg in metadata.shm_segments)
            else:
                total_size = metadata.size

            self._metrics["gets"] += 1

            logger.debug(
                "SHMRelay get: total_size=%d, descriptors=%d, segments=%d",
                total_size,
                len(descriptors),
                len(metadata.shm_segments) if metadata.shm_segments else 1,
            )

            # Return operation
            return SHMReadOperation(size=total_size)

        except Exception as e:
            self._metrics["errors"] += 1
            logger.error("SHMRelay get failed: %s", e)
            raise

    async def put_async(self, descriptors: list[Descriptor]) -> SHMReadableOperation:
        """Async version of put (SHM operations are fast, so just call sync)."""
        return self.put(descriptors)

    async def get_async(
        self, metadata: SHMMetadata, descriptors: list[Descriptor]
    ) -> SHMReadOperation:
        """Async version of get (SHM operations are fast, so just call sync).

        Returns SHMReadOperation with unified interface supporting both
        legacy mode (pre-deserialized) and descriptor mode (zero-copy).
        """
        return self.get(metadata, descriptors)

    def cleanup(self, request_id: str) -> None:
        """Force cleanup of all SHM segments for a request.

        Used when a request is aborted before data is consumed.
        """
        if request_id not in self._pending_segments:
            return

        for meta in self._pending_segments[request_id]:
            try:
                shm = _shm.SharedMemory(name=meta.name)
                shm.close()
                shm.unlink()
                logger.debug("SHM cleanup: req=%s, shm=%s", request_id, meta.name)
            except FileNotFoundError:
                # Already cleaned up
                pass
            except Exception as e:
                logger.warning(
                    "SHM cleanup failed for req %s, shm %s: %s",
                    request_id,
                    meta.name,
                    e,
                )

        del self._pending_segments[request_id]

    def health(self) -> dict[str, Any]:
        """Return health status and metrics."""
        return {
            "status": "healthy",
            "pending_requests": len(self._pending_segments),
            **self._metrics,
        }

    def close(self) -> None:
        """Clean shutdown - cleanup all pending segments."""
        for request_id in list(self._pending_segments.keys()):
            self.cleanup(request_id)
        logger.info("SHMRelay closed")
