# SPDX-License-Identifier: Apache-2.0
"""Worker that runs the processing loop."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from sglang_omni.executors.interface import Executor
from sglang_omni.pipeline.stage.work import WorkDescriptor
from sglang_omni.pipeline.worker.data_plane import DataPlaneAdapter
from sglang_omni.proto import (
    CompleteMessage,
    DataReadyMessage,
    StagePayload,
    StreamMessage,
)

if TYPE_CHECKING:
    from sglang_omni.pipeline.stage.runtime import Stage

logger = logging.getLogger(__name__)


class Worker:
    """Worker that runs the processing loop.

    Loop: get work -> executor.add_request -> executor.get_result -> route -> send
    """

    def __init__(self, executor: Executor, role: str | None = None):
        self.executor = executor
        self.engine = executor  # Backward-compatible alias.
        self.role = role
        self.stage: Stage | None = None
        self.data_plane: DataPlaneAdapter | None = None
        self.queue: asyncio.Queue[WorkDescriptor | None] | None = None
        self._running = False

    def bind(self, stage: Stage, queue: asyncio.Queue[WorkDescriptor | None]) -> None:
        """Bind this worker to a stage."""
        self.stage = stage
        self.data_plane = DataPlaneAdapter(stage.relay)
        self.queue = queue

    async def run(self) -> None:
        """Main processing loop."""
        if self.stage is None or self.queue is None or self.data_plane is None:
            raise RuntimeError("Worker not bound to a stage")

        self._running = True
        logger.info("Worker started for stage %s", self.stage.name)

        try:
            while self._running:
                work = await self.queue.get()
                if work is None:  # Shutdown signal
                    break

                await self._process_request(work)

        except asyncio.CancelledError:
            logger.info("Worker cancelled for stage %s", self.stage.name)
        finally:
            self._running = False

    async def _process_request(self, work: WorkDescriptor) -> None:
        """Process a single request."""
        request_id = work.request_id
        try:
            if self.data_plane is None:
                raise RuntimeError("Worker not bound to a data plane")
            payloads = await self._load_inputs(work)
            merged = self._merge_payloads(work, payloads)
            if not isinstance(merged, StagePayload):
                raise TypeError(f"Expected StagePayload, got {type(merged)}")
            if merged.request_id != request_id:
                raise ValueError(
                    "Merged payload request_id mismatch "
                    f"(expected={request_id} got={merged.request_id})"
                )

            await self.executor.add_request(merged)

            stream_task: asyncio.Task[None] | None = None
            stream_fn = getattr(self.executor, "stream", None)
            if callable(stream_fn):
                stream_iter = stream_fn(request_id)
                if stream_iter is not None:
                    stream_task = asyncio.create_task(
                        self._forward_stream(request_id, stream_iter)
                    )

            output_payload = await self.executor.get_result()
            if not isinstance(output_payload, StagePayload):
                raise TypeError(
                    "Executor must return StagePayload, " f"got {type(output_payload)}"
                )
            if output_payload.request_id != request_id:
                raise ValueError(
                    "Output payload request_id mismatch "
                    f"(expected={request_id} got={output_payload.request_id})"
                )

            # Determine next stage
            next_stage = self.stage.get_next(request_id, output_payload)

            # Route
            if next_stage is None:
                if stream_task is not None:
                    await self._finish_stream_task(stream_task)
                # END - send completion
                await self._send_complete(request_id, output_payload.data)
            else:
                if stream_task is not None:
                    await self._finish_stream_task(stream_task)
                # Send to next stage
                await self._send_to_next(request_id, next_stage, output_payload)

        except asyncio.CancelledError:
            logger.debug("Worker: request %s cancelled", request_id)
        except Exception as e:
            logger.error("Worker: request %s failed: %s", request_id, e)
            await self._send_failure(request_id, str(e))
        finally:
            if self.stage is not None:
                self.stage.router.clear_request(request_id)

    async def _load_inputs(self, work: WorkDescriptor) -> dict[str, StagePayload]:
        payloads: dict[str, StagePayload] = {}
        for ref in work.inputs:
            if ref.payload is not None:
                payloads[ref.source] = ref.payload
                continue
            if ref.metadata is None:
                raise ValueError(f"Missing metadata for source={ref.source}")
            payloads[ref.source] = await self.data_plane.read_payload(
                work.request_id, ref.metadata
            )
        return payloads

    @staticmethod
    def _merge_payloads(
        work: WorkDescriptor,
        payloads: dict[str, StagePayload],
    ) -> StagePayload:
        if work.merge is not None:
            return work.merge(payloads)
        if len(payloads) != 1:
            raise ValueError("Multiple inputs require a merge function")
        return next(iter(payloads.values()))

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

    async def _send_to_next(
        self, request_id: str, next_stage: str, payload: StagePayload
    ) -> None:
        """Send data to next stage."""
        logger.debug("Worker: routing %s to %s", request_id, next_stage)

        # Write using unified relay interface
        try:
            endpoint = self.stage.endpoints.get(next_stage)
            if endpoint is None:
                await self._send_failure(request_id, f"Unknown stage: {next_stage}")
                return
            metadata, op = await self.data_plane.write_payload(request_id, payload)

            await self.stage.control_plane.send_to_stage(
                next_stage,
                endpoint,
                DataReadyMessage(
                    request_id=request_id,
                    from_stage=self.stage.name,
                    to_stage=next_stage,
                    shm_metadata=metadata,
                ),
            )

            await op.wait_for_completion()
            self.data_plane.cleanup(request_id)

        except Exception as e:
            logger.exception("Worker: failed to write data for req=%s", request_id)
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

    async def _forward_stream(self, request_id: str, stream_iter: Any) -> None:
        """Forward streaming chunks to the coordinator."""
        if self.stage is None:
            return
        try:
            async for chunk in stream_iter:
                if chunk is None:
                    continue
                await self.stage.control_plane.send_stream(
                    StreamMessage(
                        request_id=request_id,
                        from_stage=self.stage.name,
                        chunk=chunk,
                        stage_name=self.stage.name,
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Worker stream error for %s: %s", request_id, exc)

    async def _finish_stream_task(self, task: asyncio.Task[None]) -> None:
        """Wait for stream task to finish and cancel if it stalls."""
        try:
            await asyncio.wait_for(task, timeout=0.5)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def stop(self) -> None:
        """Stop the worker."""
        self._running = False
