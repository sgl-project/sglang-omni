# SPDX-License-Identifier: Apache-2.0
"""Shared memory relay for inter-stage data transfer."""

import logging
import pickle
from multiprocessing import shared_memory as _shm
from typing import Any

from sglang_omni.proto import SHMMetadata
from sglang_omni.relay.descriptor import Descriptor
from sglang_omni.relay.operations.shm import SHMReadableOperation, SHMReadOperation
from sglang_omni.relay.relays.base import Relay

logger = logging.getLogger(__name__)


def shm_write_bytes(payload: bytes) -> SHMMetadata:
    """Write bytes into SharedMemory and return metadata.

    Caller should close the segment; the receiver should unlink.
    """
    shm = _shm.SharedMemory(create=True, size=len(payload))
    mv = memoryview(shm.buf)
    mv[: len(payload)] = payload
    del mv
    meta = SHMMetadata(name=shm.name, size=len(payload))
    try:
        shm.close()
    except Exception as e:
        logger.debug("Failed to close shared memory: %s", e)
    return meta


def shm_read_bytes(meta: SHMMetadata) -> bytes:
    """Read bytes from SharedMemory by metadata and cleanup."""
    shm = _shm.SharedMemory(name=meta.name)
    mv = memoryview(shm.buf)
    data = bytes(mv[: meta.size])
    del mv
    try:
        shm.close()
    except Exception:
        pass
    try:
        shm.unlink()
    except Exception:
        pass
    return data


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

        Serializes the data from descriptors and writes to shared memory.

        Parameters
        ----------
        descriptors : list[Descriptor]
            List of Descriptor objects containing tensor/data to transfer.
            For SHM, only the first descriptor is used and data is pickle-serialized.

        Returns
        -------
        SHMReadableOperation
            Operation object with metadata() and wait_for_completion() methods.
        """
        if not descriptors:
            raise ValueError("descriptors cannot be empty")

        try:
            # For SHM relay, we serialize descriptor data via pickle
            # If descriptor has _data_ref, use that; otherwise serialize descriptor info
            desc = descriptors[0]
            if hasattr(desc, "_data_ref") and desc._data_ref is not None:
                payload = pickle.dumps(desc._data_ref)
            else:
                # Fallback: serialize the raw data pointer info (less useful but safe)
                payload = pickle.dumps({"ptr": desc.ptr, "size": desc.size})

            meta = shm_write_bytes(payload)

            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += meta.size

            logger.debug(
                "SHMRelay put: %d descriptors, size=%d, shm=%s",
                len(descriptors),
                meta.size,
                meta.name,
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

        Reads from shared memory and deserializes the data.
        For SHM relay, descriptors are optional (data is returned in operation).

        Parameters
        ----------
        metadata : SHMMetadata
            Metadata from the put operation (via readable_op.metadata())
        descriptors : list[Descriptor]
            List of Descriptor objects. For SHM relay, these are ignored since
            data is deserialized and returned in the operation object.

        Returns
        -------
        SHMReadOperation
            Operation object with wait_for_completion() method and .data attribute.
        """
        if not isinstance(metadata, SHMMetadata):
            raise TypeError(f"metadata must be SHMMetadata, got {type(metadata)}")

        try:
            data_bytes = shm_read_bytes(metadata)
            obj = pickle.loads(data_bytes)

            self._metrics["gets"] += 1

            logger.debug(
                "SHMRelay get: size=%d, shm=%s",
                metadata.size,
                metadata.name,
            )

            return SHMReadOperation(obj, metadata.size)

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
        """Async version of get (SHM operations are fast, so just call sync)."""
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

    # ========== Convenience Methods for Pipeline Usage ==========

    def put_object(
        self,
        request_id: str,
        data: Any,
        from_stage: str = "",
        to_stage: str = "",
    ) -> tuple[bool, SHMMetadata | None]:
        """Store a Python object in shared memory (convenience method).

        This method provides a simpler interface for pipeline stages that
        don't need the full Descriptor-based API.

        Args:
            request_id: Request identifier (for cleanup tracking)
            data: Python object to store
            from_stage: Source stage name (for logging)
            to_stage: Destination stage name (for logging)

        Returns:
            (success, metadata) tuple
        """
        try:
            payload = pickle.dumps(data)
            meta = shm_write_bytes(payload)

            # Track for potential cleanup
            if request_id:
                if request_id not in self._pending_segments:
                    self._pending_segments[request_id] = []
                self._pending_segments[request_id].append(meta)

            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += meta.size

            logger.debug(
                "SHM put: %s -> %s, req=%s, size=%d, shm=%s",
                from_stage,
                to_stage,
                request_id,
                meta.size,
                meta.name,
            )
            return True, meta

        except Exception as e:
            self._metrics["errors"] += 1
            logger.error("SHM put failed for req %s: %s", request_id, e)
            return False, None

    def get_object(
        self,
        request_id: str,
        metadata: SHMMetadata,
        from_stage: str = "",
        to_stage: str = "",
    ) -> tuple[Any, int] | None:
        """Retrieve a Python object from shared memory (convenience method).

        This method provides a simpler interface for pipeline stages.

        Args:
            request_id: Request identifier
            metadata: SHM metadata from put_object operation
            from_stage: Source stage name (for logging)
            to_stage: Destination stage name (for logging)

        Returns:
            (object, size) tuple or None on failure
        """
        try:
            data_bytes = shm_read_bytes(metadata)
            obj = pickle.loads(data_bytes)

            # Remove from pending (it's been consumed)
            if request_id and request_id in self._pending_segments:
                self._pending_segments[request_id] = [
                    m
                    for m in self._pending_segments[request_id]
                    if m.name != metadata.name
                ]
                if not self._pending_segments[request_id]:
                    del self._pending_segments[request_id]

            self._metrics["gets"] += 1

            logger.debug(
                "SHM get: %s -> %s, req=%s, size=%d, shm=%s",
                from_stage,
                to_stage,
                request_id,
                metadata.size,
                metadata.name,
            )
            return obj, metadata.size

        except Exception as e:
            self._metrics["errors"] += 1
            logger.error("SHM get failed for req %s: %s", request_id, e)
            return None
