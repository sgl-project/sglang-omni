# SPDX-License-Identifier: Apache-2.0
"""Streaming detokenizer scheduler for the Qwen3-Omni decode stage.

Replaces the one-shot SimpleScheduler-based decode. Consumes per-token
``stream_chunk`` IncomingMessages from the thinker (each carrying a single
token id), incrementally detokenizes via HF tokenizer with UTF-8 boundary
safety, and emits text deltas as ``OutgoingMessage(type="stream", target=None)``
which the stage runtime forwards to the Coordinator. Final result is emitted
on ``new_request`` (the thinker's terminal payload via ``next``), preserving
the legacy result shape so non-streaming callers see no change.
"""
from __future__ import annotations

import logging
import queue as _queue_mod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from transformers import AutoTokenizer

from sglang_omni.models.qwen3_omni.merge import decode_events
from sglang_omni.models.qwen3_omni.payload_types import OmniEvent, PipelineState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

THINKER_STAGE = "thinker"

# Cap on orphan stream_done entries (zero-token race + late-done leak).
# When exceeded, evict oldest first down to _DONE_SEEN_EVICT_TO.
_DONE_SEEN_MAX = 10000
_DONE_SEEN_EVICT_TO = 5000


def _event_to_dict(event: OmniEvent) -> dict[str, Any]:
    return {
        "type": event.type,
        "modality": event.modality,
        "payload": dict(event.payload),
        "is_final": bool(event.is_final),
    }


@dataclass
class _RequestState:
    pending_tokens: list[int] = field(default_factory=list)
    payload: StagePayload | None = None
    done: bool = False


class StreamingDetokenizeScheduler:
    """Stream-aware decode stage."""

    def __init__(
        self,
        tokenizer: Any,
        eos_token_id: int | None,
        *,
        stage_name: str = "decode",
    ):
        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()
        self._tokenizer = tokenizer
        self._eos_token_id = eos_token_id
        self.stage_name = stage_name
        self._running = False
        self._state: dict[str, _RequestState] = {}
        self._done_seen: OrderedDict[str, None] = OrderedDict()

    def start(self) -> None:
        self._running = True
        while self._running:
            try:
                msg = self.inbox.get(timeout=0.1)
            except _queue_mod.Empty:
                continue

            # Per-request failure isolation: a malformed payload, tokenizer
            # edge case, or PipelineState/decode_events bug must fail only
            # the offending request — letting the exception escape `start()`
            # trips `Stage._handle_scheduler_crash`, which fails every
            # active request on the decode stage. Mirrors the
            # SimpleScheduler / FishScheduler / Code2WavScheduler contract.
            try:
                if msg.type == "new_request":
                    self._on_new_request(msg.request_id, msg.data)
                elif msg.type == "stream_chunk":
                    self._on_stream_chunk(msg.request_id, msg.data)
                elif msg.type == "stream_done":
                    self._on_stream_done(msg.request_id)
            except Exception as exc:
                logger.exception(
                    "StreamingDetokenizeScheduler failed request %s",
                    msg.request_id,
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
        self._running = False

    def abort(self, request_id: str) -> None:
        self._state.pop(request_id, None)
        self._done_seen.pop(request_id, None)

    def _ensure_state(self, request_id: str) -> _RequestState:
        s = self._state.get(request_id)
        if s is None:
            s = _RequestState()
            self._state[request_id] = s
        return s

    def _on_stream_chunk(self, request_id: str, item: Any) -> None:
        data = item.data
        token_id = int(data.item()) if hasattr(data, "item") else int(data)
        s = self._ensure_state(request_id)
        s.pending_tokens.append(token_id)

        candidate = self._tokenizer.decode(s.pending_tokens, skip_special_tokens=True)
        # Incomplete multi-byte UTF-8 surfaces as U+FFFD; hold pending
        # until the next token completes the byte sequence.
        if "�" in candidate:
            return

        s.pending_tokens.clear()
        if not candidate:
            return  # special tokens suppressed; nothing to emit

        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                target=None,  # terminal stream → Coordinator
                data={
                    "text": candidate,
                    "modality": "text",
                    "stage_name": self.stage_name,
                },
                metadata={"modality": "text"},
            )
        )

    def _on_stream_done(self, request_id: str) -> None:
        # No state row means either zero-token generation (no chunk created
        # state) or a late duplicate done after _finalize. Latch both;
        # _on_new_request consumes the zero-token case, the FIFO cap below
        # evicts duplicates.
        s = self._state.get(request_id)
        if s is None:
            self._done_seen[request_id] = None
            if len(self._done_seen) > _DONE_SEEN_MAX:
                for _ in range(len(self._done_seen) - _DONE_SEEN_EVICT_TO):
                    self._done_seen.popitem(last=False)
            return
        s.done = True
        if s.payload is not None:
            self._finalize(request_id)

    def _on_new_request(self, request_id: str, payload: StagePayload) -> None:
        s = self._ensure_state(request_id)
        s.payload = payload
        if request_id in self._done_seen:
            s.done = True
            self._done_seen.pop(request_id, None)
        is_streaming = bool((payload.request.params or {}).get("stream", False))
        if s.done or not is_streaming:
            self._finalize(request_id)

    def _finalize(self, request_id: str) -> None:
        s = self._state.pop(request_id, None)
        self._done_seen.pop(request_id, None)
        if s is None or s.payload is None:
            return
        # Flush leftover pending — UTF-8 may be truncated mid-char (e.g. on
        # max_tokens); without this the streaming client misses trailing
        # bytes that non-streaming clients still see in the final result.
        if s.pending_tokens:
            leftover = self._tokenizer.decode(
                s.pending_tokens, skip_special_tokens=True
            )
            if leftover:
                self.outbox.put(
                    OutgoingMessage(
                        request_id=request_id,
                        type="stream",
                        target=None,
                        data={
                            "text": leftover,
                            "modality": "text",
                            "stage_name": self.stage_name,
                        },
                        metadata={"modality": "text"},
                    )
                )
        is_streaming = bool((s.payload.request.params or {}).get("stream", False))
        result = self._build_result(s.payload, is_streaming=is_streaming)
        s.payload.data = result
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=s.payload,
            )
        )

    def _build_result(
        self, payload: StagePayload, *, is_streaming: bool = False
    ) -> dict[str, Any]:
        state = PipelineState.from_dict(payload.data)
        thinker_out = state.thinker_out or state.engine_outputs.get(THINKER_STAGE)
        if not isinstance(thinker_out, dict):
            thinker_out = {
                "output_ids": [],
                "step": 0,
                "is_final": True,
                "extra_model_outputs": {},
            }

        step = int(thinker_out.get("step") or len(thinker_out.get("output_ids", [])))
        events = list(
            decode_events(
                thinker_out=thinker_out,
                state=state,
                tokenizer=self._tokenizer,
                eos_token_id=self._eos_token_id,
                step=step,
            )
        )
        event_dicts = [_event_to_dict(event) for event in events]

        result: dict[str, Any] = {"events": event_dicts}
        final_event = next(
            (
                e
                for e in reversed(events)
                if e.is_final or e.type in {"text_final", "final"}
            ),
            None,
        )
        if final_event is not None:
            result.update(final_event.payload)
            result.setdefault("modality", final_event.modality)

        # Streaming clients already received the full output as per-token
        # text deltas via OutgoingMessage(type="stream"). The terminal
        # result must NOT carry the reconstructed full text — direct
        # consumers of Client.completion_stream() append every chunk's
        # "text" field and would otherwise emit the whole response twice.
        # Mirrors the code2wav slim-final contract for audio.
        if is_streaming:
            result.pop("text", None)
        elif "text" not in result:
            output_ids = thinker_out.get("output_ids")
            if isinstance(output_ids, list) and output_ids:
                result["text"] = self._tokenizer.decode(
                    output_ids, skip_special_tokens=True
                )
                result.setdefault("modality", "text")

        finish_reason = thinker_out.get("finish_reason")
        if finish_reason is not None:
            result.setdefault("finish_reason", finish_reason)

        input_ids = (
            state.prompt.get("input_ids") if isinstance(state.prompt, dict) else None
        )
        if input_ids is None:
            prompt_tokens = 0
        elif hasattr(input_ids, "numel"):
            prompt_tokens = int(input_ids.numel())
        else:
            prompt_tokens = len(input_ids)

        completion_ids = thinker_out.get("output_ids") or []
        completion_tokens = len(completion_ids)

        result.setdefault(
            "usage",
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )

        return result


def create_streaming_detokenize_scheduler(
    model_path: str,
    *,
    stage_name: str = "decode",
) -> StreamingDetokenizeScheduler:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return StreamingDetokenizeScheduler(
        tokenizer=tokenizer,
        eos_token_id=tokenizer.eos_token_id,
        stage_name=stage_name,
    )
