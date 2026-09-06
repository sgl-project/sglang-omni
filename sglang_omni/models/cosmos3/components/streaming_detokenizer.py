# SPDX-License-Identifier: Apache-2.0
"""Streaming text detokenizer for the Cosmos3 AR pipeline."""

from __future__ import annotations

import logging
import queue as queue_mod
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, cast

from sglang_omni.models.cosmos3.components.text_preprocessor import (
    load_cosmos3_tokenizer,
)
from sglang_omni.models.cosmos3.payload_types import (
    Cosmos3PipelineState,
    PromptInputs,
    TextOutput,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

_DONE_SEEN_MAX = 10000
_DONE_SEEN_EVICT_TO = 5000
_STATE_MAX = 10000
_STATE_ORPHAN_IDLE_S = 300.0


def _trim_matched_stop_ids(
    output_ids: list[int], matched_stop: int | str | None
) -> list[int]:
    """Drop a matched stop token id, mirroring SGLang's detokenizer trim."""
    if isinstance(matched_stop, int) and output_ids and output_ids[-1] == matched_stop:
        return output_ids[:-1]
    return output_ids


def _trim_matched_stop_text(text: str, matched_stop: int | str | None) -> str:
    """Truncate at a matched stop string, mirroring SGLang's detokenizer trim."""
    if isinstance(matched_stop, str):
        pos = text.find(matched_stop)
        if pos != -1:
            return text[:pos]
    return text


@dataclass
class _RequestState:
    pending_tokens: list[int] = field(default_factory=list)
    payload: StagePayload | None = None
    done: bool = False
    last_seen: float = 0.0


class Cosmos3StreamingDetokenizer:
    """Decode per-token stream messages and the terminal AR payload."""

    def __init__(self, tokenizer: Any, *, stage_name: str = "decode") -> None:
        self.inbox: queue_mod.Queue[IncomingMessage] = queue_mod.Queue()
        self.outbox: queue_mod.Queue[OutgoingMessage] = queue_mod.Queue()
        self.stage_name = stage_name
        self._tokenizer = tokenizer
        self._running = False
        self._state: dict[str, _RequestState] = {}
        self._done_seen: OrderedDict[str, None] = OrderedDict()
        self._aborted: OrderedDict[str, None] = OrderedDict()
        self._last_evict_s = 0.0
        # Stage aborts run on the event-loop thread while handlers run here.
        self._state_lock = threading.RLock()

    def start(self) -> None:
        self._running = True
        while self._running:
            try:
                message = self.inbox.get(timeout=0.1)
            except queue_mod.Empty:
                continue
            try:
                if message.type == "new_request":
                    self._on_new_request(message.request_id, message.data)
                elif message.type == "stream_chunk":
                    self._on_stream_chunk(message.request_id, message.data)
                elif message.type == "stream_done":
                    self._on_stream_done(message.request_id)
            except Exception as exc:
                logger.exception(
                    "Cosmos3 detokenizer failed request %s", message.request_id
                )
                self.abort(message.request_id)
                self.outbox.put(
                    OutgoingMessage(
                        request_id=message.request_id,
                        type="error",
                        data=exc,
                    )
                )

    def stop(self) -> None:
        self._running = False

    def abort(self, request_id: str) -> None:
        with self._state_lock:
            self._state.pop(request_id, None)
            self._done_seen.pop(request_id, None)
            self._aborted[request_id] = None
            if len(self._aborted) > _DONE_SEEN_MAX:
                for _ in range(len(self._aborted) - _DONE_SEEN_EVICT_TO):
                    self._aborted.popitem(last=False)

    def _ensure_state(self, request_id: str) -> _RequestState:
        with self._state_lock:
            now = time.monotonic()
            state = self._state.get(request_id)
            if state is None:
                state = _RequestState(last_seen=now)
                self._state[request_id] = state
            state.last_seen = now
            if len(self._state) > _STATE_MAX and now - self._last_evict_s >= 1.0:
                self._last_evict_s = now
                self._evict_idle_orphans(now)
            return state

    def _evict_idle_orphans(self, now: float) -> None:
        cutoff = now - _STATE_ORPHAN_IDLE_S
        with self._state_lock:
            stale = [
                request_id
                for request_id, state in self._state.items()
                if state.payload is None and not state.done and state.last_seen < cutoff
            ]
            for request_id in stale:
                self._state.pop(request_id, None)
        if stale:
            logger.warning(
                "Evicted %d idle orphan stream states (cap %d exceeded)",
                len(stale),
                _STATE_MAX,
            )

    def _on_stream_chunk(self, request_id: str, item: StreamItem) -> None:
        with self._state_lock:
            if request_id in self._aborted:
                return
            state = self._ensure_state(request_id)
            if item.metadata and item.metadata.get("terminal_flush"):
                # Tokens held upstream as possible stop-string prefixes arrive
                # in one finish-time chunk; buffer them for _finalize's
                # matched-stop trim instead of emitting them live.
                state.pending_tokens.extend(int(t) for t in item.data.tolist())
                return
            token_id = int(item.data.item())
            state.pending_tokens.append(token_id)
            candidate = self._tokenizer.decode(
                state.pending_tokens,
                skip_special_tokens=True,
            )
            if candidate.endswith("\ufffd"):
                return
            state.pending_tokens.clear()
            if not candidate:
                return
            self._emit_text(request_id, candidate)

    def _emit_text(self, request_id: str, text: str) -> None:
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

    def _on_stream_done(self, request_id: str) -> None:
        with self._state_lock:
            if request_id in self._aborted:
                return
            state = self._state.get(request_id)
            if state is None:
                self._done_seen[request_id] = None
                if len(self._done_seen) > _DONE_SEEN_MAX:
                    for _ in range(len(self._done_seen) - _DONE_SEEN_EVICT_TO):
                        self._done_seen.popitem(last=False)
                return
            state.done = True
            if state.payload is not None:
                self._finalize(request_id)

    def _on_new_request(self, request_id: str, payload: StagePayload) -> None:
        with self._state_lock:
            if request_id in self._aborted:
                return
            state = self._ensure_state(request_id)
            state.payload = payload
            if request_id in self._done_seen:
                state.done = True
                self._done_seen.pop(request_id, None)
            is_streaming = bool((payload.request.params or {}).get("stream", False))
            if state.done or not is_streaming:
                self._finalize(request_id)

    def _finalize(self, request_id: str) -> None:
        with self._state_lock:
            state = self._state.pop(request_id, None)
            self._done_seen.pop(request_id, None)
            if state is None or state.payload is None:
                return
            if state.pending_tokens:
                text_out = (state.payload.data or {}).get("text_out") or {}
                matched_stop = text_out.get("matched_stop")
                trailing_text = self._tokenizer.decode(
                    _trim_matched_stop_ids(state.pending_tokens, matched_stop),
                    skip_special_tokens=True,
                )
                trailing_text = _trim_matched_stop_text(trailing_text, matched_stop)
                if trailing_text:
                    self._emit_text(request_id, trailing_text)

            is_streaming = bool(
                (state.payload.request.params or {}).get("stream", False)
            )
            state.payload.data = self._build_result(
                state.payload,
                is_streaming=is_streaming,
            )
            self.outbox.put(
                OutgoingMessage(
                    request_id=request_id,
                    type="result",
                    data=state.payload,
                )
            )

    def _build_result(
        self,
        payload: StagePayload,
        *,
        is_streaming: bool,
    ) -> dict[str, Any]:
        state = Cosmos3PipelineState.from_dict(payload.data)
        text_out = cast(TextOutput, state.text_out)

        output_ids = text_out["output_ids"]
        matched_stop = text_out.get("matched_stop")
        result: dict[str, Any] = {"modality": "text"}
        if not is_streaming:
            text = self._tokenizer.decode(
                _trim_matched_stop_ids(output_ids, matched_stop),
                skip_special_tokens=True,
            )
            result["text"] = _trim_matched_stop_text(text, matched_stop)

        for key in ("finish_reason", "output_token_logprobs", "weight_version"):
            if text_out.get(key) is not None:
                result[key] = text_out[key]

        prompt = cast(PromptInputs, state.prompt)
        prompt_tokens = int(prompt["input_ids"].numel())
        result["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(output_ids),
            "total_tokens": prompt_tokens + len(output_ids),
        }
        return result


def create_streaming_detokenize_scheduler(
    model_path: str,
    *,
    revision: str | None = None,
    stage_name: str = "decode",
) -> Cosmos3StreamingDetokenizer:
    return Cosmos3StreamingDetokenizer(
        tokenizer=load_cosmos3_tokenizer(model_path, revision=revision),
        stage_name=stage_name,
    )


__all__ = [
    "Cosmos3StreamingDetokenizer",
    "create_streaming_detokenize_scheduler",
]
