# SPDX-License-Identifier: Apache-2.0
"""Worker that runs the processing loop."""

from __future__ import annotations

import asyncio
import logging
import pickle
from typing import TYPE_CHECKING, Any

import numpy as np

from sglang_omni.engines.base import Engine
from sglang_omni.proto import CompleteMessage, DataReadyMessage
from sglang_omni.relay.descriptor import Descriptor

if TYPE_CHECKING:
    from sglang_omni.pipeline.stage import Stage

logger = logging.getLogger(__name__)


class Worker:
    """Worker that runs the processing loop.

    Loop: get request -> engine.add_request -> engine.get_result -> route -> send
    """

    def __init__(self, engine: Engine):
        self.engine = engine
        self.stage: Stage | None = None
        self._running = False

    def bind(self, stage: Stage) -> None:
        """Bind this worker to a stage."""
        self.stage = stage

    async def run(self) -> None:
        """Main processing loop."""
        if self.stage is None:
            raise RuntimeError("Worker not bound to a stage")

        self._running = True
        logger.info("Worker started for stage %s", self.stage.name)

        try:
            while self._running:
                # Get request from queue
                item = await self.stage.request_queue.get()
                if item is None:  # Shutdown signal
                    break

                request_id, data = item
                await self._process_request(request_id, data)

        except asyncio.CancelledError:
            logger.info("Worker cancelled for stage %s", self.stage.name)
        finally:
            self._running = False

    async def _process_request(self, request_id: str, data: Any) -> None:
        """Process a single request."""
        try:
            # Process through engine
            await self.engine.add_request(request_id, data)
            output = await self.engine.get_result(request_id)

            # Determine next stage
            next_stage = self.stage.get_next(request_id, output)

            # Transform output
            transformed = self.engine.transform_output(request_id, output, next_stage)

            # Route
            if next_stage is None:
                # END - send completion
                await self._send_complete(request_id, transformed)
            else:
                # Send to next stage
                await self._send_to_next(request_id, next_stage, transformed)

        except asyncio.CancelledError:
            logger.debug("Worker: request %s cancelled", request_id)
        except Exception as e:
            logger.error("Worker: request %s failed: %s", request_id, e)
            await self._send_failure(request_id, str(e))

    async def _send_complete(self, request_id: str, result: Any) -> None:
        """Send completion to coordinator."""
        logger.debug("Worker: %s completed (END)", request_id)
        await self.stage.control_plane.send_complete(
            CompleteMessage(
                request_id=request_id,
                from_stage=self.stage.name,
                success=True,
                result=result,
            )
        )

    async def _send_to_next(self, request_id: str, next_stage: str, data: Any) -> None:
        """Send data to next stage."""
        logger.debug("Worker: routing %s to %s", request_id, next_stage)

        # Write using unified relay interface
        try:
            # Unified approach: serialize data first for both relay types
            # This allows both SHMRelay and NIXLRelay to use the same descriptor format
            serialized_data = pickle.dumps(data)
            data_size = len(serialized_data)

            # Create a numpy buffer to hold the serialized data
            # Use np.frombuffer to create a view, then copy to make it writable
            buffer = np.frombuffer(serialized_data, dtype=np.uint8).copy()

            # Create descriptor with serialized data buffer
            # Both SHMRelay and NIXLRelay can use this format
            descriptor = Descriptor((buffer.ctypes.data, data_size, "cpu", buffer))

            # Put data and get metadata
            readable_op = await self.stage.relay.put_async([descriptor])
            metadata = readable_op.metadata()

            logger.debug(
                "Worker: data written from %s to %s for req=%s",
                self.stage.name,
                next_stage,
                request_id,
            )

            # Get endpoint
            endpoint = self.stage.endpoints.get(next_stage)
            if endpoint is None:
                await self._send_failure(request_id, f"Unknown stage: {next_stage}")
                return

            # Send notification
            await self.stage.control_plane.send_to_stage(
                next_stage,
                endpoint,
                DataReadyMessage(
                    request_id=request_id,
                    from_stage=self.stage.name,
                    to_stage=next_stage,
                    shm_metadata=metadata,  # Can be SHMMetadata or RdmaMetadata
                ),
            )

            await readable_op.wait_for_completion()

        except Exception as e:
            logger.error("Worker: failed to write data for req=%s: %s", request_id, e)
            await self._send_failure(request_id, f"Failed to write data: {e}")
            return

    async def _send_failure(self, request_id: str, error: str) -> None:
        """Send failure to coordinator."""
        await self.stage.control_plane.send_complete(
            CompleteMessage(
                request_id=request_id,
                from_stage=self.stage.name,
                success=False,
                error=error,
            )
        )

    def stop(self) -> None:
        """Stop the worker."""
        self._running = False
