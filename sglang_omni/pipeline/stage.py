# SPDX-License-Identifier: Apache-2.0
"""Stage abstraction for pipeline processing."""

from __future__ import annotations

import asyncio
import logging
import pickle
from typing import Any, Callable

import numpy as np

from sglang_omni.pipeline.control_plane import StageControlPlane
from sglang_omni.pipeline.input_handler import DirectInput, InputHandler
from sglang_omni.pipeline.worker import Worker
from sglang_omni.proto import (
    DataReadyMessage,
    ShutdownMessage,
    StageInfo,
    SubmitMessage,
)
from sglang_omni.relay.descriptor import Descriptor
from sglang_omni.relay.relays.base import Relay
from sglang_omni.relay.relays.nixl import NIXLRelay
from sglang_omni.relay.relays.shm import SHMRelay

logger = logging.getLogger(__name__)


# Type alias for get_next function
# Returns: next_stage_name or None for END
GetNextFn = Callable[[str, Any], str | None]


class Stage:
    """A processing stage in the pipeline.

    Responsibilities:
    - Receive work (via control plane)
    - Handle input aggregation (via input_handler)
    - Queue work for workers
    - Workers process and route output
    """

    def __init__(
        self,
        name: str,
        get_next: GetNextFn,
        recv_endpoint: str,
        coordinator_endpoint: str,
        abort_endpoint: str,
        endpoints: dict[str, str],
        input_handler: InputHandler | None = None,
        relay: Relay | None = None,
        relay_config: dict[str, Any] | None = None,
    ):
        """Initialize a stage.

        Args:
            name: Stage name (unique identifier)
            get_next: Function to determine next stage
                      (request_id, output) -> stage_name or None
            recv_endpoint: ZMQ endpoint to receive work
            coordinator_endpoint: ZMQ endpoint to send completions
            abort_endpoint: ZMQ endpoint for abort broadcasts
            endpoints: Dict of stage_name -> endpoint for routing
            input_handler: Input handler for aggregation (default: DirectInput)
            relay: Relay instance for data transfer (default: NIXLRelay if config provided, else SHMRelay)
            relay_config: Configuration dict for NIXLRelay (if relay is None)
        """
        self.name = name
        self.get_next = get_next
        self.endpoints = endpoints
        self.input_handler = input_handler or DirectInput()

        # Components
        # Initialize relay: use provided relay, or create NIXLRelay if config provided, else SHMRelay
        if relay_config is not None:
            self.relay = NIXLRelay(relay_config)
        else:
            self.relay = SHMRelay()

        self.control_plane = StageControlPlane(
            stage_name=name,
            recv_endpoint=recv_endpoint,
            coordinator_endpoint=coordinator_endpoint,
            abort_endpoint=abort_endpoint,
        )

        # Request queue for workers
        self.request_queue: asyncio.Queue[tuple[str, Any] | None] = asyncio.Queue()

        # Workers
        self.workers: list[Worker] = []

        # State
        self._running = False
        self._aborted_requests: set[str] = set()

    def add_worker(self, worker: Worker) -> None:
        """Add a worker to this stage."""
        worker.bind(self)
        self.workers.append(worker)

    async def start(self) -> None:
        """Start the stage."""
        await self.control_plane.start()
        self._running = True
        logger.info("Stage %s started", self.name)

    async def stop(self) -> None:
        """Stop the stage."""
        self._running = False

        # Signal workers to stop
        for _ in self.workers:
            await self.request_queue.put(None)

        self.control_plane.close()
        logger.info("Stage %s stopped", self.name)

    async def run(self) -> None:
        """Main loop: receive work, handle input, queue for workers."""
        await self.start()

        # Start workers
        worker_tasks = [asyncio.create_task(w.run()) for w in self.workers]

        # Start abort listener
        abort_task = asyncio.create_task(self._abort_listener())

        try:
            while self._running:
                # Receive work
                msg = await self.control_plane.recv()

                if isinstance(msg, ShutdownMessage):
                    logger.info("Stage %s received shutdown", self.name)
                    break

                await self._handle_message(msg)

        except asyncio.CancelledError:
            logger.info("Stage %s cancelled", self.name)
        except Exception as e:
            logger.error("Stage %s error: %s", self.name, e)
            raise
        finally:
            # Stop
            await self.stop()

            # Cancel abort listener
            abort_task.cancel()
            try:
                await abort_task
            except asyncio.CancelledError:
                pass

            # Wait for workers
            for task in worker_tasks:
                task.cancel()
            await asyncio.gather(*worker_tasks, return_exceptions=True)

    async def _abort_listener(self) -> None:
        """Background task to listen for abort broadcasts."""
        try:
            while self._running:
                abort_msg = await self.control_plane.recv_abort()
                self._on_abort(abort_msg.request_id)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Stage %s abort listener error: %s", self.name, e)

    async def _handle_message(self, msg: DataReadyMessage | SubmitMessage) -> None:
        """Handle an incoming message."""
        if isinstance(msg, SubmitMessage):
            await self._process_submit(msg)
        elif isinstance(msg, DataReadyMessage):
            await self._process_data_ready(msg)
        else:
            logger.warning(
                "Stage %s received unexpected message: %s", self.name, type(msg)
            )

    async def _process_submit(self, msg: SubmitMessage) -> None:
        """Process initial submission from coordinator."""
        request_id = msg.request_id
        logger.debug("Stage %s received submit: req=%s", self.name, request_id)

        if request_id in self._aborted_requests:
            logger.debug("Stage %s skipping aborted req=%s", self.name, request_id)
            return

        # Handle input (for DirectInput, just returns data)
        data = self.input_handler.receive(request_id, "coordinator", msg.data)
        if data is not None:
            await self.request_queue.put((request_id, data))

    async def _process_data_ready(self, msg: DataReadyMessage) -> None:
        """Process data ready notification from previous stage."""
        request_id = msg.request_id
        logger.debug(
            "Stage %s received data_ready: req=%s from %s",
            self.name,
            request_id,
            msg.from_stage,
        )

        if request_id in self._aborted_requests:
            logger.debug("Stage %s skipping aborted req=%s", self.name, request_id)
            self.relay.cleanup(request_id)
            return

        # Read data using relay interface
        try:
            # Determine relay type and prepare descriptors accordingly
            if isinstance(self.relay, SHMRelay):
                # SHMRelay: descriptors can be empty, data is in read_op.data
                read_op = await self.relay.get_async(
                    metadata=msg.shm_metadata, descriptors=[]
                )
                await read_op.wait_for_completion()
                data = read_op.data
            else:
                # NIXLRelay: need to create descriptors from metadata
                # Extract remote descriptors from metadata
                remote_descriptors = msg.shm_metadata.to_descriptors()

                # Create local descriptors (buffers) to receive data
                # Handle both single Descriptor and list[Descriptor] cases
                if isinstance(remote_descriptors, list):
                    # Multiple descriptors - create buffers for each
                    local_descriptors = []
                    for remote_desc in remote_descriptors:
                        # Create a buffer of the same size
                        buffer = np.empty(remote_desc.size, dtype=np.uint8)
                        local_desc = Descriptor(
                            (buffer.ctypes.data, remote_desc.size, "cpu", buffer)
                        )
                        local_descriptors.append(local_desc)
                else:
                    # Single descriptor
                    buffer = np.empty(remote_descriptors.size, dtype=np.uint8)
                    local_desc = Descriptor(
                        (buffer.ctypes.data, remote_descriptors.size, "cpu", buffer)
                    )
                    local_descriptors = [local_desc]

                read_op = await self.relay.get_async(
                    metadata=msg.shm_metadata, descriptors=local_descriptors
                )
                await read_op.wait_for_completion()

                # Extract data from buffer(s)
                # For simple Python objects, data should be in the first buffer
                if len(local_descriptors) > 0:
                    buffer = local_descriptors[0]._data_ref
                    # Deserialize the data (assuming it was pickled)
                    data = pickle.loads(buffer.tobytes())
                else:
                    logger.error(
                        "Stage %s: no descriptors to extract data from for req=%s",
                        self.name,
                        request_id,
                    )
                    return
        except Exception as e:
            logger.error(
                "Stage %s failed to get data for req=%s: %s", self.name, request_id, e
            )
            return

        # Handle input aggregation
        merged = self.input_handler.receive(request_id, msg.from_stage, data)
        if merged is not None:
            await self.request_queue.put((request_id, merged))

    def _on_abort(self, request_id: str) -> None:
        """Handle abort for a request."""
        logger.debug("Stage %s: aborting req=%s", self.name, request_id)
        self._aborted_requests.add(request_id)
        self.input_handler.cancel(request_id)
        self.relay.cleanup(request_id)

        # Notify workers' engines
        for worker in self.workers:
            asyncio.create_task(worker.engine.abort(request_id))

    def info(self) -> StageInfo:
        """Return stage info."""
        return StageInfo(
            name=self.name,
            control_endpoint=self.control_plane.recv_endpoint,
        )

    def health(self) -> dict[str, Any]:
        """Return health status."""
        return {
            "name": self.name,
            "running": self._running,
            "queue_size": self.request_queue.qsize(),
            "num_workers": len(self.workers),
            "relay": self.relay.health(),
        }
