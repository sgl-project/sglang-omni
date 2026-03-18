# SPDX-License-Identifier: Apache-2.0
"""DirectModelExecutor — run a torch model directly without engine infrastructure."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from sglang_omni.executors.interface import Executor
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)


class DirectModelExecutor(Executor):
    """Run a torch model directly: input → model.forward(**kwargs) → output.

    For simple models (Code2Wav, CodePredictor wrapper) that don't need
    batching, scheduling, or the encoder pipeline.

    Supports two modes:
    - **Batch** (default): prefetch all chunks, forward once, emit one result.
    - **Streaming** (streaming=True): consume chunks one-by-one from mailbox,
      forward each, enqueue result to downstream via stream_fn,
      then emit a summary result when EOS is received.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device,
        request_builder: Callable[[StagePayload], dict[str, Any]],
        result_builder: Callable[[StagePayload, Any], StagePayload],
        *,
        streaming: bool = False,
    ):
        self._model = model
        self._device = torch.device(device)
        self._request_builder = request_builder
        self._result_builder = result_builder
        self._streaming = streaming
        self._results: asyncio.Queue[StagePayload] = asyncio.Queue()
        self._aborted: set[str] = set()
        # Set externally by compiler for stream-receiving stages
        self._stream_queue: Any | None = None
        # Set externally by compiler for stream-sending stages
        self._stream_fn: Callable | None = None
        self._target_stage: str | None = None

    def set_stream_fn(self, fn: Callable) -> None:
        """Set the streaming output callback. Sync, non-blocking."""
        self._stream_fn = fn

    def set_stream_target(self, target_stage: str) -> None:
        """Configure the downstream stage for streaming outputs."""
        self._target_stage = target_stage

    async def add_request(self, payload: StagePayload) -> None:
        request_id = payload.request_id
        if request_id in self._aborted:
            return

        if self._streaming and self._stream_queue is not None:
            await self._process_streaming(payload)
        else:
            await self._process_batch(payload)

    async def _process_batch(self, payload: StagePayload) -> None:
        """Original batch mode: prefetch all chunks, forward once."""
        request_id = payload.request_id

        if self._stream_queue is not None:
            chunks = []
            while True:
                item = await self._stream_queue.get(request_id)
                if item is None:  # EOS
                    break
                chunks.append(item)
            payload.prefetched_chunks = chunks

        model_inputs = self._request_builder(payload)

        loop = asyncio.get_running_loop()
        output = await loop.run_in_executor(None, self._run_model, model_inputs)

        result_payload = self._result_builder(payload, output)
        if not isinstance(result_payload, StagePayload):
            result_payload = StagePayload(
                request_id=request_id,
                request=payload.request,
                data=result_payload,
            )

        await self._results.put(result_payload)

    async def _process_streaming(self, payload: StagePayload) -> None:
        """Streaming mode: consume chunks one-by-one, forward each."""
        request_id = payload.request_id
        chunk_count = 0
        loop = asyncio.get_running_loop()

        while True:
            if request_id in self._aborted:
                break

            item = await self._stream_queue.get(request_id)
            if item is None:  # EOS
                break

            # Build input from single chunk
            payload.prefetched_chunks = [item]
            model_inputs = self._request_builder(payload)

            # Forward
            output = await loop.run_in_executor(None, self._run_model, model_inputs)

            # Enqueue result as chunk to downstream via stream_fn
            if self._stream_fn is not None:
                if self._target_stage is None:
                    raise RuntimeError(
                        "DirectModelExecutor streaming requires a configured target stage"
                    )
                chunk_tensor, chunk_metadata = self._extract_chunk_output(output)
                self._stream_fn(
                    request_id,
                    chunk_tensor,
                    self._target_stage,
                    metadata=chunk_metadata,
                )

            chunk_count += 1

        # EOS is signaled by the Worker after executor completes

        # Emit summary result for the pipeline
        result_payload = self._result_builder(payload, {"chunk_count": chunk_count})
        if not isinstance(result_payload, StagePayload):
            result_payload = StagePayload(
                request_id=request_id,
                request=payload.request,
                data=result_payload,
            )
        await self._results.put(result_payload)

    @staticmethod
    def _extract_chunk_output(output: Any) -> tuple[torch.Tensor, dict | None]:
        """Extract tensor and metadata from model output for chunk transfer."""
        if isinstance(output, torch.Tensor):
            return output, None
        if isinstance(output, dict):
            tensor = output.get("tensor", output.get("codes"))
            metadata = output.get("metadata")
            if tensor is None:
                # Use first tensor value found
                for v in output.values():
                    if isinstance(v, torch.Tensor):
                        tensor = v
                        break
            # Pass summed_embeddings in metadata for feedback channel
            summed = output.get("summed_embeddings")
            if summed is not None:
                if metadata is None:
                    metadata = {}
                metadata["summed_embeddings"] = summed
            return tensor, metadata
        return output, None

    @torch.no_grad()
    def _run_model(self, inputs: dict[str, Any]) -> Any:
        if self._device.type == "cuda":
            torch.cuda.set_device(self._device)
        return self._model(**inputs)

    async def get_result(self) -> StagePayload:
        while True:
            result = await self._results.get()
            if result.request_id in self._aborted:
                continue
            return result

    async def abort(self, request_id: str) -> None:
        self._aborted.add(request_id)
