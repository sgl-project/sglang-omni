# SPDX-License-Identifier: Apache-2.0
"""Streaming detokenizer scheduler for the Qwen3-Omni decode stage.

Replaces the one-shot SimpleScheduler-based decode. Consumes per-token
``stream_chunk`` IncomingMessages from the thinker (each carrying a single
token id), incrementally detokenizes via HF tokenizer with UTF-8 boundary
safety, and emits text deltas as ``OutgoingMessage(type="stream", target=None)``
which the stage runtime forwards to the Coordinator. Final result is emitted
on ``new_request`` (the thinker's terminal payload via ``next``), preserving
the legacy result shape so non-streaming callers see no change.

Delta strategy: keep cumulative ``output_ids``, decode the whole prefix on
each step, and emit ``decode(prefix)[len(emitted):]``. This matches HF's
``TextStreamer`` and is robust to tokenizers whose per-token decoding depends
on neighbors (sentencepiece BOS, leading-space BPE artifacts, etc.).
"""
from __future__ import annotations

import logging
import queue as _queue_mod
from dataclasses import dataclass, field
from typing import Any

from transformers import AutoTokenizer

from sglang_omni.models.qwen3_omni.merge import decode_events
from sglang_omni.models.qwen3_omni.payload_types import OmniEvent, PipelineState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

THINKER_STAGE = "thinker"


def _event_to_dict(event: OmniEvent) -> dict[str, Any]:
    return {
        "type": event.type,
        "modality": event.modality,
        "payload": dict(event.payload),
        "is_final": bool(event.is_final),
    }


@dataclass
class _RequestState:
    output_ids: list[int] = field(default_factory=list)
    emitted_text: str = ""
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
        # Tracks request_ids where stream_done arrived without an active
        # state row (zero-token race). _on_new_request consumes the entry;
        # abort/_finalize clean up to bound the set.
        self._done_seen: set[str] = set()

    def start(self) -> None:
        self._running = True
        while self._running:
            try:
                msg = self.inbox.get(timeout=0.1)
            except _queue_mod.Empty:
                continue

            if msg.type == "new_request":
                self._on_new_request(msg.request_id, msg.data)
            elif msg.type == "stream_chunk":
                self._on_stream_chunk(msg.request_id, msg.data)
            elif msg.type == "stream_done":
                self._on_stream_done(msg.request_id)

    def stop(self) -> None:
        self._running = False

    def abort(self, request_id: str) -> None:
        self._state.pop(request_id, None)
        self._done_seen.discard(request_id)

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
        s.output_ids.append(token_id)

        # Decode the whole prefix; this is robust to tokenizers whose
        # per-token decoding depends on neighbors. Incomplete multi-byte
        # UTF-8 surfaces as U+FFFD; hold and wait for the next token.
        text = self._tokenizer.decode(s.output_ids, skip_special_tokens=True)
        if "�" in text:
            return

        delta = text[len(s.emitted_text) :]
        s.emitted_text = text
        if not delta:
            return  # special-token-only step

        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                target=None,  # terminal stream → Coordinator
                data={
                    "text": delta,
                    "modality": "text",
                    "stage_name": self.stage_name,
                },
                metadata={"modality": "text"},
            )
        )

    def _on_stream_done(self, request_id: str) -> None:
        # Two orderings reach this with no state row:
        #   1. Late stream_done after _finalize popped state. Drop silently.
        #   2. stream_done before any chunk and before new_request (e.g.,
        #      zero-token generation). Latch into _done_seen so
        #      _on_new_request finalizes when it arrives.
        # We can't tell them apart cheaply, so we always latch and rely on
        # _on_new_request / abort to consume. A bounded cap evicts stale
        # entries from duplicate-done bugs to prevent unbounded growth.
        s = self._state.get(request_id)
        if s is None:
            self._done_seen.add(request_id)
            if len(self._done_seen) > 10000:
                excess = len(self._done_seen) - 5000
                it = iter(self._done_seen)
                stale = [next(it) for _ in range(excess)]
                self._done_seen -= set(stale)
            return
        s.done = True
        if s.payload is not None:
            self._finalize(request_id)

    def _on_new_request(self, request_id: str, payload: StagePayload) -> None:
        s = self._ensure_state(request_id)
        s.payload = payload
        if request_id in self._done_seen:
            s.done = True
            self._done_seen.discard(request_id)
        is_streaming = bool((payload.request.params or {}).get("stream", False))
        if s.done or not is_streaming:
            self._finalize(request_id)

    def _finalize(self, request_id: str) -> None:
        s = self._state.pop(request_id, None)
        self._done_seen.discard(request_id)
        if s is None or s.payload is None:
            return
        # Flush leftover bytes that never resolved (e.g. max_tokens cut a
        # multi-byte char in half). Without this the streaming client misses
        # trailing bytes that non-streaming clients still see in the final
        # result.
        text = self._tokenizer.decode(s.output_ids, skip_special_tokens=True)
        if text and len(text) > len(s.emitted_text):
            leftover = text[len(s.emitted_text) :]
            if leftover and "�" not in leftover:
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
        result = self._build_result(s.payload)
        s.payload.data = result
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=s.payload,
            )
        )

    def _build_result(self, payload: StagePayload) -> dict[str, Any]:
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

        if "text" not in result:
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
