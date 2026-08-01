# SPDX-License-Identifier: Apache-2.0
"""Streaming text detokenizer for the Cosmos3 AR pipeline."""

from __future__ import annotations

import logging
import queue as queue_mod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from sglang_omni.models.cosmos3.components.text_preprocessor import (
    load_cosmos3_tokenizer,
)
from sglang_omni.models.cosmos3.payload_types import Cosmos3PipelineState
from sglang_omni.models.cosmos3.request_builders import THINKER_STAGE
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

_DONE_SEEN_MAX = 10000
_DONE_SEEN_EVICT_TO = 5000


@dataclass
class _RequestState:
    pending_tokens: list[int] = field(default_factory=list)
    payload: StagePayload | None = None
    done: bool = False


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
        self._state.pop(request_id, None)
        self._done_seen.pop(request_id, None)

    def _ensure_state(self, request_id: str) -> _RequestState:
        state = self._state.get(request_id)
        if state is None:
            state = _RequestState()
            self._state[request_id] = state
        return state

    def _on_stream_chunk(self, request_id: str, item: Any) -> None:
        raw_token = item.data
        token_id = (
            int(raw_token.item()) if hasattr(raw_token, "item") else int(raw_token)
        )
        state = self._ensure_state(request_id)
        state.pending_tokens.append(token_id)
        candidate = self._tokenizer.decode(
            state.pending_tokens,
            skip_special_tokens=True,
        )
        if "\ufffd" in candidate:
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
        state = self._ensure_state(request_id)
        state.payload = payload
        if request_id in self._done_seen:
            state.done = True
            self._done_seen.pop(request_id, None)
        is_streaming = bool((payload.request.params or {}).get("stream", False))
        if state.done or not is_streaming:
            self._finalize(request_id)

    def _finalize(self, request_id: str) -> None:
        state = self._state.pop(request_id, None)
        self._done_seen.pop(request_id, None)
        if state is None or state.payload is None:
            return
        if state.pending_tokens:
            trailing_text = self._tokenizer.decode(
                state.pending_tokens,
                skip_special_tokens=True,
            )
            if trailing_text:
                self._emit_text(request_id, trailing_text)

        is_streaming = bool((state.payload.request.params or {}).get("stream", False))
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
        text_out = state.text_out or state.engine_outputs.get(THINKER_STAGE)
        if not isinstance(text_out, dict):
            text_out = {"output_ids": [], "is_final": True}

        output_ids = list(text_out.get("output_ids") or [])
        result: dict[str, Any] = {"modality": "text"}
        if not is_streaming:
            result["text"] = self._tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
            )

        for key in ("finish_reason", "output_token_logprobs", "weight_version"):
            if text_out.get(key) is not None:
                result[key] = text_out[key]

        input_ids = (
            state.prompt.get("input_ids") if isinstance(state.prompt, dict) else None
        )
        prompt_tokens = (
            int(input_ids.numel())
            if hasattr(input_ids, "numel")
            else len(input_ids or [])
        )
        result["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(output_ids),
            "total_tokens": prompt_tokens + len(output_ids),
        }
        return result


def create_streaming_detokenize_scheduler(
    model_path: str,
    *,
    stage_name: str = "decode",
) -> Cosmos3StreamingDetokenizer:
    return Cosmos3StreamingDetokenizer(
        tokenizer=load_cosmos3_tokenizer(model_path),
        stage_name=stage_name,
    )


__all__ = [
    "Cosmos3StreamingDetokenizer",
    "create_streaming_detokenize_scheduler",
]
