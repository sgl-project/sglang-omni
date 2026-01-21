# SPDX-License-Identifier: Apache-2.0
"""Stage abstraction for pipeline processing."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from sglang_omni.pipeline.control_plane import StageControlPlane
from sglang_omni.pipeline.stage.input import DirectInput, InputHandler
from sglang_omni.pipeline.stage.router import WorkerRouter
from sglang_omni.pipeline.stage.work import InputRef
from sglang_omni.pipeline.worker.runtime import Worker
from sglang_omni.proto import (
    DataReadyMessage,
    ShutdownMessage,
    StageInfo,
    SubmitMessage,
)
from sglang_omni.relay.base import Relay
from sglang_omni.relay.nixl import NixlRelay

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
            relay: Relay instance for data transfer (default: NixlRelay if config provided)
            relay_config: Configuration dict for NixlRelay (if relay is None)
        """
        self.name = name
        self.get_next = get_next
        self.endpoints = endpoints
        self.input_handler = input_handler or DirectInput()

        # Components
        # Initialize relay: use provided relay, or create NixlRelay if config provided
        if relay is not None:
            self.relay = relay
        elif relay_config is not None:
            # Extract engine_id and device from config, with defaults
            engine_id = relay_config.get("worker_id", f"{name}_relay")
            device = "cuda" if relay_config.get("gpu_id") is not None else "cpu"
            self.relay = NixlRelay(engine_id=engine_id, device=device)
        else:
            # Default: create NixlRelay with default config
            self.relay = NixlRelay(engine_id=f"{name}_relay")

        self.router = WorkerRouter()

        self.control_plane = StageControlPlane(
            stage_name=name,
            recv_endpoint=recv_endpoint,
            coordinator_endpoint=coordinator_endpoint,
            abort_endpoint=abort_endpoint,
        )

        # Workers
        self.workers: list[Worker] = []

        # State
        self._running = False
        self._aborted_requests: set[str] = set()

    def add_worker(self, worker: Worker) -> None:
        """Add a worker to this stage."""
        queue = self.router.add_worker()
        worker.bind(self, queue)
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
        for worker in self.workers:
            await worker.queue.put(None)

        self.control_plane.close()
        self.relay.close()
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

        input_ref = InputRef.from_payload("coordinator", msg.data)
        work = self.input_handler.receive(request_id, "coordinator", input_ref)
        if work is not None:
            self.router.enqueue(work)

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

        input_ref = InputRef.from_metadata(msg.from_stage, msg.shm_metadata)
        work = self.input_handler.receive(request_id, msg.from_stage, input_ref)
        if work is not None:
            self.router.enqueue(work)

    def _on_abort(self, request_id: str) -> None:
        """Handle abort for a request."""
        logger.debug("Stage %s: aborting req=%s", self.name, request_id)
        self._aborted_requests.add(request_id)
        self.router.clear_request(request_id)
        self.input_handler.cancel(request_id)
        self.relay.cleanup(request_id)

        # Notify workers' engines
        for worker in self.workers:
            asyncio.create_task(worker.executor.abort(request_id))

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
            "queue_size": self.router.queue_size(),
            "num_workers": self.router.num_workers(),
            "relay": self.relay.health(),
        }
