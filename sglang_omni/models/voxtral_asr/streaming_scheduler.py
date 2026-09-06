# SPDX-License-Identifier: Apache-2.0
"""Prototype streaming scheduler for Voxtral realtime ASR.

This scheduler implements chunk-by-chunk realtime transcription with token
feedback: audio chunks arrive as ``stream_chunk`` messages, and generated text
tokens are appended to the next chunk's prompt.  It is intentionally separate
from the offline/batched ``OmniScheduler`` path and is meant to be wired into a
future ``/v1/realtime`` adapter for Voxtral.

Notes / TODO:
- The encoder currently runs a full forward pass over the current audio window
  on every chunk.  Incremental encoder KV-cache reuse is left for future work.
- The scheduler assumes one active stream per request_id.
- It does not yet integrate with SGLang's CUDA graph / continuous batching
  paths for the text decoder; each chunk is processed as a single-request
  forward.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class _StreamState:
    """Mutable state for one realtime stream."""

    request_id: str
    audio_buffer: deque[np.ndarray] = field(default_factory=deque)
    token_buffer: deque[int] = field(default_factory=deque)
    # Prompt prefix tokens (start + streaming tokens) from mistral-common.
    prefix_tokens: list[int] = field(default_factory=list)
    output_text: list[str] = field(default_factory=list)
    finished: bool = False


class VoxtralRealtimeStreamingScheduler:
    """Minimal streaming scheduler for Voxtral realtime ASR.

    Implements the sglang-omni scheduler surface:
      - inbox: Queue[IncomingMessage]
      - outbox: Queue[OutgoingMessage]
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        audio_config: Any,
        max_new_tokens_per_chunk: int = 16,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.audio_config = audio_config
        self.max_new_tokens_per_chunk = max_new_tokens_per_chunk

        self.inbox: asyncio.Queue[Any] = asyncio.Queue()
        self.outbox: asyncio.Queue[Any] = asyncio.Queue()
        self._streams: dict[str, _StreamState] = {}
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def abort(self, request_id: str) -> None:
        self._streams.pop(request_id, None)

    async def _loop(self) -> None:
        while True:
            message = await self.inbox.get()
            try:
                await self._handle_message(message)
            except Exception as exc:
                logger.exception("Error handling streaming message: %s", exc)
                await self.outbox.put(
                    self._make_message(
                        message.request_id,
                        "error",
                        {"error": str(exc)},
                    )
                )

    async def _handle_message(self, message: Any) -> None:
        msg_type = message.type
        request_id = message.request_id

        if msg_type == "new_request":
            audio_encoder = self.tokenizer.instruct.audio_encoder
            prefix_tokens = (
                self.tokenizer.instruct.start()
                + audio_encoder.encode_streaming_tokens()
            )
            state = _StreamState(
                request_id=request_id,
                prefix_tokens=list(prefix_tokens),
            )
            self._streams[request_id] = state
            await self._maybe_process_chunk(state)

        elif msg_type == "stream_chunk":
            state = self._streams.get(request_id)
            if state is None:
                raise ValueError(f"Unknown stream {request_id}")
            chunk = message.data.get("audio")
            if chunk is not None:
                if isinstance(chunk, np.ndarray):
                    state.audio_buffer.append(chunk)
                else:
                    state.audio_buffer.append(np.array(chunk))
            await self._maybe_process_chunk(state)

        elif msg_type == "stream_done":
            state = self._streams.pop(request_id, None)
            if state is not None:
                state.finished = True
                await self._maybe_process_chunk(state, flush=True)
                await self.outbox.put(
                    self._make_message(
                        request_id, "result", {"text": "".join(state.output_text)}
                    )
                )

    async def _maybe_process_chunk(
        self, state: _StreamState, *, flush: bool = False
    ) -> None:
        """Process one audio chunk and emit any generated text deltas."""
        # Placeholder: real implementation needs access to the model worker and
        # ForwardBatch construction.  Here we just log the chunk shape and token
        # buffer length so the integration point is visible.
        if not state.audio_buffer:
            return

        audio_chunk = np.concatenate(list(state.audio_buffer))
        state.audio_buffer.clear()

        logger.debug(
            "[voxtral-realtime] request=%s chunk_samples=%d tokens_so_far=%d flush=%s",
            state.request_id,
            len(audio_chunk),
            len(state.token_buffer),
            flush,
        )

        # TODO: construct ForwardBatch from state.prefix_tokens + list(token_buffer),
        # run model forward, sample next tokens, append to token_buffer, and emit
        # text deltas to outbox.

    def _make_message(self, request_id: str, msg_type: str, data: Any) -> Any:
        from sglang_omni.scheduling.messages import OutgoingMessage

        return OutgoingMessage(
            request_id=request_id,
            type=msg_type,
            data=data,
            target=None,
            metadata=None,
        )
