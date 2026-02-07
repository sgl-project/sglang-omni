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

    Requests are processed concurrently: each dequeued work item becomes an
    independent task.  A dedicated dispatcher task consumes results from the
    executor (in completion order) and routes them to the corresponding request
    task via per-request :class:`asyncio.Future` objects.

    Back-pressure is provided naturally by the work queue (upstream) and by
    the executor's ``add_request`` (downstream).
    """

    def __init__(self, executor: Executor, role: str | None = None):
        self.executor = executor
        self.engine = executor  # Backward-compatible alias.
        self.role = role
        self.stage: Stage | None = None
        self.data_plane: DataPlaneAdapter | None = None
        self.queue: asyncio.Queue[WorkDescriptor | None] | None = None
        self._running = False
        self._result_waiters: dict[str, asyncio.Future[StagePayload]] = {}

    def bind(self, stage: Stage, queue: asyncio.Queue[WorkDescriptor | None]) -> None:
        """Bind this worker to a stage."""
        self.stage = stage
        self.data_plane = DataPlaneAdapter(stage.relay)
        self.queue = queue

    async def run(self) -> None:
        """Main processing loop."""
        if self.stage is None or self.queue is None or self.data_plane is None:
            raise RuntimeError("Worker not bound to a stage")

        try:
            await self.executor.start()
            self._running = True
            logger.info("Worker started for stage %s", self.stage.name)

            inflight: set[asyncio.Task[None]] = set()
            dispatcher = asyncio.create_task(self._dispatch_results())

            try:
                while self._running:
                    work = await self.queue.get()
                    if work is None:
                        break

                    task = asyncio.create_task(self._process_request(work))
                    inflight.add(task)
                    task.add_done_callback(inflight.discard)

                if inflight:
                    await asyncio.gather(*inflight, return_exceptions=True)
            finally:
                dispatcher.cancel()
                try:
                    await dispatcher
                except asyncio.CancelledError:
                    pass

        except asyncio.CancelledError:
            logger.info("Worker cancelled for stage %s", self.stage.name)
        finally:
            self._running = False
            await self.executor.stop()

    async def _dispatch_results(self) -> None:
        """Single consumer that routes executor results to per-request futures."""
        while True:
            try:
                result = await self.executor.get_result()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Worker dispatcher: get_result error")
                continue

            fut = self._result_waiters.pop(result.request_id, None)
            if fut is not None and not fut.done():
                fut.set_result(result)
            else:
                logger.warning(
                    "Worker dispatcher: no waiter for request %s",
                    result.request_id,
                )

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

            # Register future BEFORE add_request so the dispatcher can
            # route the result even if the executor completes synchronously.
            loop = asyncio.get_running_loop()
            fut: asyncio.Future[StagePayload] = loop.create_future()
            self._result_waiters[request_id] = fut

            await self.executor.add_request(merged)

            stream_task: asyncio.Task[None] | None = None
            stream_fn = getattr(self.executor, "stream", None)
            if callable(stream_fn):
                stream_iter = stream_fn(request_id)
                if stream_iter is not None:
                    stream_task = asyncio.create_task(
                        self._forward_stream(request_id, stream_iter)
                    )

            output_payload = await fut

            if not isinstance(output_payload, StagePayload):
                raise TypeError(
                    "Executor must return StagePayload, " f"got {type(output_payload)}"
                )
            if output_payload.request_id != request_id:
                raise ValueError(
                    "Output payload request_id mismatch "
                    f"(expected={request_id} got={output_payload.request_id})"
                )

            # Route
            next_stage = self.stage.get_next(request_id, output_payload)

            if next_stage is None:
                if stream_task is not None:
                    await self._finish_stream_task(stream_task)
                await self._send_complete(request_id, output_payload.data)
            else:
                if stream_task is not None:
                    await self._finish_stream_task(stream_task)
                for stage_name in self._normalize_next_stages(next_stage):
                    await self._send_to_next(request_id, stage_name, output_payload)

        except asyncio.CancelledError:
            logger.debug("Worker: request %s cancelled", request_id)
        except Exception:
            logger.exception("Worker: request %s failed", request_id)
            await self._send_failure(request_id, str(request_id))
        finally:
            self._result_waiters.pop(request_id, None)
            if self.stage is not None:
                self.stage.router.clear_request(request_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    @staticmethod
    def _normalize_next_stages(next_stage: str | list[str]) -> list[str]:
        match next_stage:
            case str() as stage:
                return [stage]
            case list() as stages if stages:
                return stages
            case list():
                raise ValueError("get_next returned an empty stage list")
            case _:
                raise TypeError(
                    "get_next must return a stage name, list of stage names, or None"
                )

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
