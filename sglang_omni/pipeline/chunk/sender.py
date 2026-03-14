# SPDX-License-Identifier: Apache-2.0
"""Sender-side chunk transfer via relay."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import torch

from sglang_omni.pipeline.worker.data_plane import _extract_tensors
from sglang_omni.proto.messages import ChunkReadyMessage

logger = logging.getLogger(__name__)


@dataclass
class _PendingChunk:
    request_id: str
    tensor: torch.Tensor | None  # None for done/error
    is_chunks_done: bool = False
    error: str | None = None
    metadata: dict | None = None


class ChunkTransferAdapter:
    """Enqueues chunks synchronously; sends them via relay in a background task.

    Usage:
        sender = ChunkTransferAdapter(data_plane, control_plane, ...)
        await sender.start()
        sender.enqueue("req-1", tensor)  # sync, non-blocking
        sender.enqueue_chunks_done("req-1")
        await sender.stop()
    """

    def __init__(
        self,
        data_plane,
        control_plane,
        to_stage: str,
        to_stage_endpoint: str,
        from_stage: str,
    ):
        self._data_plane = data_plane
        self._control_plane = control_plane
        self._to_stage = to_stage
        self._to_stage_endpoint = to_stage_endpoint
        self._from_stage = from_stage
        self._queue: asyncio.Queue[_PendingChunk] = asyncio.Queue(maxsize=64)
        self._task: asyncio.Task | None = None
        self._running = False
        self._chunk_counters: dict[str, int] = {}

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._send_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def enqueue(
        self, request_id: str, tensor: torch.Tensor, metadata: dict | None = None
    ) -> None:
        """Enqueue a chunk for async transfer. Safe to call from sync context."""
        try:
            self._queue.put_nowait(
                _PendingChunk(request_id=request_id, tensor=tensor, metadata=metadata)
            )
        except asyncio.QueueFull:
            raise RuntimeError(f"chunk sender queue full for {request_id}")

    def enqueue_chunks_done(self, request_id: str) -> None:
        """Signal that no more chunks will be sent for this request."""
        try:
            self._queue.put_nowait(
                _PendingChunk(request_id=request_id, tensor=None, is_chunks_done=True)
            )
        except asyncio.QueueFull:
            raise RuntimeError(f"chunk sender queue full for {request_id}")

    def enqueue_error(self, request_id: str, error: str) -> None:
        """Signal error for a request."""
        try:
            self._queue.put_nowait(
                _PendingChunk(request_id=request_id, tensor=None, error=error)
            )
        except asyncio.QueueFull:
            raise RuntimeError(f"chunk sender queue full for {request_id}")

    async def _send_loop(self) -> None:
        while self._running or not self._queue.empty():
            try:
                pending = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            await self._send_one(pending)

    async def _send_one(self, pending: _PendingChunk) -> None:
        request_id = pending.request_id
        chunk_id = self._chunk_counters.get(request_id, 0)

        if pending.error:
            msg = ChunkReadyMessage(
                request_id=request_id,
                from_stage=self._from_stage,
                to_stage=self._to_stage,
                chunk_id=chunk_id,
                relay_metadata={},
                error=pending.error,
            )
            await self._control_plane.send_to_stage(
                self._to_stage, self._to_stage_endpoint, msg
            )
            self._chunk_counters.pop(request_id, None)
            return

        if pending.is_chunks_done:
            msg = ChunkReadyMessage(
                request_id=request_id,
                from_stage=self._from_stage,
                to_stage=self._to_stage,
                chunk_id=chunk_id,
                relay_metadata={},
                is_chunks_done=True,
            )
            await self._control_plane.send_to_stage(
                self._to_stage, self._to_stage_endpoint, msg
            )
            self._chunk_counters.pop(request_id, None)
            return

        try:
            blob_key = f"{request_id}:chunk:{chunk_id}"
            metadata, op = await self._data_plane.write_blob(blob_key, pending.tensor)
            await op.wait_for_completion()
            relay_metadata = metadata
            if pending.metadata:
                safe_metadata, metadata_tensors = _extract_tensors(pending.metadata)
                relay_metadata["chunk_metadata"] = safe_metadata
                if metadata_tensors:
                    tensor_blobs: dict[str, dict[str, Any]] = {}
                    for meta_idx, (path, tensor) in enumerate(metadata_tensors.items()):
                        meta_blob_key = f"{blob_key}:meta:{meta_idx}"
                        meta_metadata, meta_op = await self._data_plane.write_blob(
                            meta_blob_key, tensor
                        )
                        await meta_op.wait_for_completion()
                        tensor_blobs[path] = {
                            "blob_key": meta_blob_key,
                            "relay_metadata": meta_metadata,
                        }
                    relay_metadata["chunk_metadata_tensors"] = tensor_blobs
            msg = ChunkReadyMessage(
                request_id=request_id,
                from_stage=self._from_stage,
                to_stage=self._to_stage,
                chunk_id=chunk_id,
                relay_metadata=relay_metadata,
            )
            await self._control_plane.send_to_stage(
                self._to_stage, self._to_stage_endpoint, msg
            )
            self._chunk_counters[request_id] = chunk_id + 1
        except Exception as exc:
            logger.error("ChunkTransfer send failed for %s: %s", request_id, exc)
            err_msg = ChunkReadyMessage(
                request_id=request_id,
                from_stage=self._from_stage,
                to_stage=self._to_stage,
                chunk_id=chunk_id,
                relay_metadata={},
                error=str(exc),
            )
            await self._control_plane.send_to_stage(
                self._to_stage, self._to_stage_endpoint, err_msg
            )
