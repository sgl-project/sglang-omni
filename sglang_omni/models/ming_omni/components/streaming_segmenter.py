# SPDX-License-Identifier: Apache-2.0
"""Streaming text segmenter scheduler for Ming-Omni V1.

Bridges incremental text deltas from the thinker (via the stream channel)
into speakable segments routed to the streaming talker. Uses the pure-logic
``SegmenterState`` from ``streaming_text``.

Wire-up:
- Thinker emits per-token uint8-text-deltas to this stage via ``stream_to``.
- Thinker also fans out its final payload to this stage via ``next`` (so we
  know when generation is done and to attach the per-request handle).
- This stage emits per-segment uint8 tensors to ``talker_stream`` via
  ``stream_to``, then emits a result payload when both the upstream stream
  is done AND the main payload has arrived.
"""

from __future__ import annotations

import logging
import queue as _queue_mod
import time
from dataclasses import dataclass, field
from typing import Any

from sglang_omni.models.ming_omni.components.streaming_text import (
    SegmenterConfig,
    SegmenterState,
    TextSegment,
    TokenCountFn,
    text_to_uint8_tensor,
    uint8_tensor_to_text,
)
from sglang_omni.models.ming_omni.pipeline.next_stage import TALKER_STREAM_STAGE
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)


def _default_token_count(text: str) -> int:
    # Whitespace tokens for English; codepoint length for CJK fallback.
    return len(text.split()) or len(text)


@dataclass
class _RequestState:
    segmenter: SegmenterState
    payload: StagePayload | None = None
    payload_arrived: bool = False
    stream_done: bool = False
    finalized: bool = False
    aborted: bool = False
    segment_count: int = 0
    first_text_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MingStreamingSegmenterScheduler:
    """Stream-aware scheduler for the Ming streaming TTS segmenter stage.

    Same inbox/outbox contract as SimpleScheduler so the stage runtime can
    drive it without branching. Single-threaded: ``start()`` blocks on the
    inbox loop until ``stop()``.
    """

    def __init__(
        self,
        *,
        config: SegmenterConfig | None = None,
        token_count_fn: TokenCountFn | None = None,
        target_stage: str = TALKER_STREAM_STAGE,
    ) -> None:
        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()

        self._config = config or SegmenterConfig()
        self._token_count_fn = token_count_fn or _default_token_count
        self._target_stage = target_stage
        self._running = False
        self._states: dict[str, _RequestState] = {}

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._running = True
        while self._running:
            try:
                msg = self.inbox.get(timeout=0.1)
            except _queue_mod.Empty:
                self._tick_first_segment_timeouts()
                continue
            try:
                self._handle_message(msg)
            except Exception as exc:
                logger.exception(
                    "MingStreamingSegmenterScheduler: failed handling %s for %s",
                    msg.type,
                    msg.request_id,
                )
                self._emit_error(msg.request_id, exc)
                self._states.pop(msg.request_id, None)

    def stop(self) -> None:
        self._running = False

    def abort(self, request_id: str) -> None:
        state = self._states.get(request_id)
        if state is None:
            return
        state.aborted = True
        # Drop without finalizing — the runtime will close the stream queue.
        self._states.pop(request_id, None)

    # ------------------------------------------------------------------ dispatch
    def _handle_message(self, msg: IncomingMessage) -> None:
        if msg.type == "new_request":
            self._on_new_request(msg)
        elif msg.type == "stream_chunk":
            self._on_stream_chunk(msg)
        elif msg.type == "stream_done":
            self._on_stream_done(msg)
        else:
            logger.debug("MingStreamingSegmenter: ignored message type=%s", msg.type)

    # ------------------------------------------------------------------ payload arrival
    def _on_new_request(self, msg: IncomingMessage) -> None:
        request_id = msg.request_id
        payload = msg.data
        state = self._states.get(request_id)
        if state is None:
            state = _RequestState(
                segmenter=SegmenterState(self._config, self._token_count_fn),
            )
            self._states[request_id] = state
        state.payload = payload
        state.payload_arrived = True
        # If the upstream stream already signalled done before the payload
        # arrived, finalize now.
        if state.stream_done and not state.finalized:
            self._finalize(request_id, state)

    # ------------------------------------------------------------------ stream chunk
    def _on_stream_chunk(self, msg: IncomingMessage) -> None:
        request_id = msg.request_id
        item = msg.data
        state = self._states.get(request_id)
        if state is None:
            # First sight of this request: create state so we can buffer
            # incoming text even before the thinker's main payload arrives.
            state = _RequestState(
                segmenter=SegmenterState(self._config, self._token_count_fn),
            )
            self._states[request_id] = state
        if state.aborted or state.finalized:
            return
        if not isinstance(item, StreamItem):
            return

        text = uint8_tensor_to_text(item.data)
        if not text:
            return
        now_ms = self._now_ms()
        if state.first_text_ms is None:
            state.first_text_ms = now_ms

        for segment in state.segmenter.push(text, now_ms=now_ms):
            self._emit_segment(request_id, segment)
            state.segment_count += 1
            state.first_text_ms = None

    # ------------------------------------------------------------------ stream done
    def _on_stream_done(self, msg: IncomingMessage) -> None:
        request_id = msg.request_id
        state = self._states.get(request_id)
        if state is None:
            return
        state.stream_done = True
        if state.payload_arrived and not state.finalized:
            self._finalize(request_id, state)

    # ------------------------------------------------------------------ first-seg timer
    def _tick_first_segment_timeouts(self) -> None:
        if not self._states:
            return
        now_ms = self._now_ms()
        wait = self._config.first_segment_max_wait_ms
        for request_id, state in list(self._states.items()):
            if state.aborted or state.finalized or state.segment_count != 0:
                continue
            if state.first_text_ms is None:
                continue
            if now_ms - state.first_text_ms < wait:
                continue
            if (
                state.segmenter.buffer_token_count()
                < self._config.first_segment_min_tokens
            ):
                continue
            for segment in state.segmenter.push("", now_ms=now_ms):
                self._emit_segment(request_id, segment)
                state.segment_count += 1
                state.first_text_ms = None

    # ------------------------------------------------------------------ finalize
    def _finalize(self, request_id: str, state: _RequestState) -> None:
        if state.finalized:
            return
        state.finalized = True

        final_segments = state.segmenter.flush()
        if not final_segments:
            final_segments = [
                TextSegment(
                    segment_id=state.segment_count,
                    text="",
                    is_final_segment=True,
                )
            ]
        for segment in final_segments:
            self._emit_segment(request_id, segment)
            state.segment_count += 1

        payload = state.payload
        # If we received streams but the payload never arrived (defensive),
        # synthesize an empty payload so the result channel is closed.
        if payload is None:
            payload = StagePayload(request_id=request_id, request=None, data={})
        # Strip the upstream tensor-laden state dict; only forward the
        # segmenter's own summary stats. The talker_stream stage doesn't
        # need the thinker_out / prompt / encoder_outs fields, and they
        # carry torch.Tensor instances that downstream msgpack can't
        # serialize when emitting the terminal result.
        payload.data = {
            "segment_count": state.segment_count,
            "aborted": False,
        }
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=payload,
            )
        )
        self._states.pop(request_id, None)

    # ------------------------------------------------------------------ emit helpers
    def _emit_segment(self, request_id: str, segment: TextSegment) -> None:
        data = text_to_uint8_tensor(segment.text)
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=data,
                target=self._target_stage,
                metadata={
                    "segment_id": segment.segment_id,
                    "is_final_segment": bool(segment.is_final_segment),
                    "text_len": int(data.numel()),
                },
            )
        )

    def _emit_error(self, request_id: str, exc: BaseException) -> None:
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="error",
                data=exc,
            )
        )

    @staticmethod
    def _now_ms() -> int:
        return int(time.monotonic() * 1000)
