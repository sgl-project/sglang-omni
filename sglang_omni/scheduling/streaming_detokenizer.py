# SPDX-License-Identifier: Apache-2.0
"""Streaming text detokenizer for a dedicated decode stage.

Consumes per-token stream_chunk messages, emits UTF-8-boundary-safe text
deltas, and asks the model for the terminal result dict via build_result.
"""

from __future__ import annotations

import logging
import queue
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

DONE_SEEN_MAX = 10000
DONE_SEEN_EVICT_TO = 5000

BuildResultFn = Callable[[StagePayload, bool], dict]


class Tokenizer(Protocol):
    def decode(
        self, token_ids: list[int], skip_special_tokens: bool = False
    ) -> str: ...


class HasItem(Protocol):
    def item(self) -> int: ...


class StreamChunk(Protocol):
    data: int | HasItem


@dataclass(kw_only=True)
class RequestState:
    pending_tokens: list[int] = field(default_factory=list)
    payload: StagePayload | None = None
    done: bool = False


def stream_token_id(data: int | HasItem) -> int:
    if isinstance(data, int):
        return data
    return int(data.item())


class StreamingDetokenizeScheduler:
    """Stream-aware decode stage. build_result owns the model-specific final dict."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        eos_token_id: int | None,
        *,
        build_result: BuildResultFn,
        stage_name: str = "decode",
    ) -> None:
        self.inbox: queue.Queue[IncomingMessage] = queue.Queue()
        self.outbox: queue.Queue[OutgoingMessage] = queue.Queue()
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.build_result = build_result
        self.stage_name = stage_name
        self.is_running = False
        self.request_states: dict[str, RequestState] = {}
        self.done_seen: OrderedDict[str, None] = OrderedDict()

    def start(self) -> None:
        self.is_running = True
        while self.is_running:
            try:
                msg = self.inbox.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if msg.type == "new_request":
                    self.on_new_request(msg.request_id, msg.data)
                elif msg.type == "stream_chunk":
                    self.on_stream_chunk(msg.request_id, msg.data)
                elif msg.type == "stream_done":
                    self.on_stream_done(msg.request_id)
            except Exception as exc:
                # note (Chenyang): isolate one request; escaping start() crashes the stage.
                logger.exception(
                    f"StreamingDetokenizeScheduler failed request {msg.request_id}"
                )
                self.abort(msg.request_id)
                self.outbox.put(
                    OutgoingMessage(
                        request_id=msg.request_id,
                        type="error",
                        data=exc,
                    )
                )

    def stop(self) -> None:
        self.is_running = False

    def abort(self, request_id: str) -> None:
        self.request_states.pop(request_id, None)
        self.done_seen.pop(request_id, None)

    def ensure_request_state(self, request_id: str) -> RequestState:
        request_state = self.request_states.get(request_id)
        if request_state is None:
            request_state = RequestState()
            self.request_states[request_id] = request_state
        return request_state

    def emit_text_delta(self, request_id: str, text: str) -> None:
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                target=None,
                data={
                    "text": text,
                    "modality": "text",
                    "stage_name": self.stage_name,
                },
                metadata={"modality": "text"},
            )
        )

    def on_stream_chunk(self, request_id: str, chunk: StreamChunk) -> None:
        request_state = self.ensure_request_state(request_id)
        request_state.pending_tokens.append(stream_token_id(chunk.data))
        candidate = self.tokenizer.decode(
            request_state.pending_tokens, skip_special_tokens=True
        )
        if "\ufffd" in candidate:
            return
        request_state.pending_tokens.clear()
        if candidate:
            self.emit_text_delta(request_id, candidate)

    def on_stream_done(self, request_id: str) -> None:
        request_state = self.request_states.get(request_id)
        if request_state is None:
            self.done_seen[request_id] = None
            overflow = len(self.done_seen) - DONE_SEEN_MAX
            if overflow > 0:
                for _ in range(len(self.done_seen) - DONE_SEEN_EVICT_TO):
                    self.done_seen.popitem(last=False)
            return
        request_state.done = True
        if request_state.payload is not None:
            self.finalize(request_id)

    def on_new_request(self, request_id: str, payload: StagePayload) -> None:
        request_state = self.ensure_request_state(request_id)
        request_state.payload = payload
        if request_id in self.done_seen:
            request_state.done = True
            self.done_seen.pop(request_id, None)
        is_streaming = bool((payload.request.params or {}).get("stream", False))
        if request_state.done or not is_streaming:
            self.finalize(request_id)

    def finalize(self, request_id: str) -> None:
        request_state = self.request_states.pop(request_id, None)
        self.done_seen.pop(request_id, None)
        if request_state is None or request_state.payload is None:
            return
        if request_state.pending_tokens:
            leftover = self.tokenizer.decode(
                request_state.pending_tokens, skip_special_tokens=True
            )
            if leftover:
                self.emit_text_delta(request_id, leftover)
        is_streaming = bool(
            (request_state.payload.request.params or {}).get("stream", False)
        )
        request_state.payload.data = self.build_result(
            request_state.payload, is_streaming
        )
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=request_state.payload,
            )
        )
