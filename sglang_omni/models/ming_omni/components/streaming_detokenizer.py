# SPDX-License-Identifier: Apache-2.0
"""Streaming detokenizer scheduler for the Ming-Omni decode stage.

Replaces the one-shot SimpleScheduler-based decode for text-only pipelines.
Consumes per-token ``stream_chunk`` IncomingMessages from the thinker (each
carrying a single token id as a torch.LongTensor), incrementally detokenizes
with UTF-8 boundary safety, and emits text deltas as
``OutgoingMessage(type="stream", target=None)`` which the stage runtime
forwards to the Coordinator.

Final result is emitted on ``new_request`` (the thinker's terminal payload),
preserving the existing non-streaming result shape. When streaming, ``text``
is stripped from the final result to avoid sending the full response twice.

Incremental decode runs on the held ``pending_tokens`` buffer only, which is
equality-safe for suffix-additive tokenizers (byte-level BPE, as Ming uses)
but would drop inter-word spaces with a sentencepiece/metaspace tokenizer.
"""
from __future__ import annotations

import logging
import queue as _queue_mod
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from sglang_omni.models.ming_omni.io import MingOmniPipelineState
from sglang_omni.models.ming_omni.pipeline.merge import decode_events
from sglang_omni.models.ming_omni.pipeline.next_stage import THINKER_STAGE
from sglang_omni.models.ming_omni.pipeline.state_io import load_state
from sglang_omni.models.ming_omni.pipeline.usage import build_text_usage
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

_DONE_SEEN_MAX = 10000
_DONE_SEEN_EVICT_TO = 5000

# The Stage runtime delivers abort(request_id) whenever a request fails or is
# cancelled, which calls our abort() and removes the entry from _state. This
# cap is a safety net against orphan entries if abort is ever lost (e.g. a
# stream_chunk arrived before new_request, and the request was aborted before
# new_request was delivered to the decode stage). Only entries idle for
# _STATE_ORPHAN_IDLE_S are evicted: live streaming requests receive chunks
# continuously, and done=True entries are awaiting an imminent new_request —
# evicting either would drop tokens or hang an active request.
_STATE_MAX = 10000
_STATE_ORPHAN_IDLE_S = 300.0


@dataclass
class RequestState:
    pending_tokens: list[int] = field(default_factory=list)
    payload: StagePayload | None = None
    done: bool = False
    last_seen: float = 0.0


class MingStreamingDetokenizeScheduler:
    """Stream-aware decode stage for Ming-Omni text-only pipelines.

    Public contract (used by Stage):
        ``inbox``, ``outbox``, ``start()``, ``stop()``, ``abort(request_id)``
    """

    def __init__(
        self,
        tokenizer: Any,
        eos_token_id: int | None,
        *,
        stage_name: str = "decode",
    ):
        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.stage_name = stage_name
        self.running = False
        self.state: dict[str, RequestState] = {}
        self.done_seen: OrderedDict[str, None] = OrderedDict()
        # abort() runs on the stage's event-loop thread while start() runs on
        # the scheduler thread; guards iteration/multi-op sections.
        self.state_lock = threading.Lock()

    def start(self) -> None:
        self.running = True
        while self.running:
            try:
                msg = self.inbox.get(timeout=0.1)
            except _queue_mod.Empty:
                continue
            try:
                if msg.type == "new_request":
                    self.on_new_request(msg.request_id, msg.data)
                elif msg.type == "stream_chunk":
                    self.on_stream_chunk(msg.request_id, msg.data)
                elif msg.type == "stream_done":
                    self.on_stream_done(msg.request_id)
                else:
                    pass
            except Exception as exc:
                logger.exception(
                    "MingStreamingDetokenizeScheduler failed request %s",
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
        self.running = False

    def abort(self, request_id: str) -> None:
        with self.state_lock:
            self.state.pop(request_id, None)
            self.done_seen.pop(request_id, None)

    def ensure_state(self, request_id: str) -> RequestState:
        s = self.state.get(request_id)
        if s is None:
            s = RequestState(last_seen=time.monotonic())
            self.state[request_id] = s
            if len(self.state) > _STATE_MAX:
                self.evict_idle_orphans()
            else:
                pass
        else:
            pass
        s.last_seen = time.monotonic()
        return s

    def evict_idle_orphans(self) -> None:
        cutoff = time.monotonic() - _STATE_ORPHAN_IDLE_S
        with self.state_lock:
            stale = [
                rid
                for rid, st in self.state.items()
                if st.payload is None and not st.done and st.last_seen < cutoff
            ]
            for rid in stale:
                self.state.pop(rid, None)
        if stale:
            logger.warning(
                "Evicted %d idle orphan stream states (cap %d exceeded)",
                len(stale),
                _STATE_MAX,
            )
        else:
            pass

    def on_stream_chunk(self, request_id: str, item: Any) -> None:
        # item is the StreamItem the runtime wraps around the thinker's
        # torch.tensor([token_id], dtype=torch.long)
        data = item.data
        token_id = int(data.item()) if hasattr(data, "item") else int(data)

        s = self.ensure_state(request_id)
        s.pending_tokens.append(token_id)

        candidate = self.tokenizer.decode(s.pending_tokens, skip_special_tokens=True)
        # A trailing U+FFFD means an incomplete multi-byte UTF-8 char; hold
        # until the next token. Interior U+FFFD (model emitting a literal
        # replacement char) flushes normally — holding would stall streaming
        # for the rest of the request.
        if candidate.endswith("�"):
            return
        else:
            pass

        s.pending_tokens.clear()
        if not candidate:
            return
        else:
            pass

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

    def on_stream_done(self, request_id: str) -> None:
        s = self.state.get(request_id)
        if s is None:
            # Zero-token generation or late duplicate done — latch for
            # _on_new_request to consume.
            self.done_seen[request_id] = None
            if len(self.done_seen) > _DONE_SEEN_MAX:
                with self.state_lock:
                    while len(self.done_seen) > _DONE_SEEN_EVICT_TO:
                        self.done_seen.popitem(last=False)
            else:
                pass
            return
        else:
            pass
        s.done = True
        if s.payload is not None:
            self.finalize(request_id)
        else:
            pass

    def on_new_request(self, request_id: str, payload: StagePayload) -> None:
        s = self.ensure_state(request_id)
        s.payload = payload
        if request_id in self.done_seen:
            s.done = True
            self.done_seen.pop(request_id, None)
        else:
            pass
        is_streaming = bool((payload.request.params or {}).get("stream", False))
        if s.done or not is_streaming:
            self.finalize(request_id)
        else:
            pass

    def finalize(self, request_id: str) -> None:
        s = self.state.pop(request_id, None)
        self.done_seen.pop(request_id, None)
        if s is None or s.payload is None:
            return
        else:
            pass

        # Flush any remaining pending tokens (e.g. truncated UTF-8 on max_tokens).
        if s.pending_tokens:
            leftover = self.tokenizer.decode(s.pending_tokens, skip_special_tokens=True)
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
            else:
                pass
        else:
            pass

        is_streaming = bool((s.payload.request.params or {}).get("stream", False))
        result = self.build_result(s.payload, is_streaming=is_streaming)
        s.payload.data = result
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=s.payload,
            )
        )

    def build_result(
        self, payload: StagePayload, *, is_streaming: bool = False
    ) -> dict[str, Any]:
        state = load_state(payload)
        thinker_out = state.thinker_out or state.engine_outputs.get(THINKER_STAGE)
        if not isinstance(thinker_out, dict):
            thinker_out = {
                "output_ids": [],
                "step": 0,
                "is_final": True,
                "extra_model_outputs": {},
            }
        else:
            pass

        step = int(thinker_out.get("step") or len(thinker_out.get("output_ids", [])))
        events = list(
            decode_events(
                thinker_out=thinker_out,
                state=state,
                tokenizer=self.tokenizer,
                eos_token_id=self.eos_token_id,
                step=step,
            )
        )

        result: dict[str, Any] = {"events": [event_to_dict(e) for e in events]}
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
        else:
            pass

        # Streaming clients already received the full output as per-token
        # deltas; strip text from the terminal result to prevent
        # double-sending. Must mirror the emission gate in
        # make_text_stream_output_builder: when text output was not requested
        # no deltas were ever emitted, so the final result keeps its text.
        if is_streaming and text_output_requested(payload.request):
            result.pop("text", None)
        elif "text" not in result:
            output_ids = thinker_out.get("output_ids")
            if isinstance(output_ids, list) and output_ids:
                result["text"] = self.tokenizer.decode(
                    output_ids, skip_special_tokens=True
                )
                result.setdefault("modality", "text")
            else:
                pass
        else:
            pass

        attach_decode_final_metadata(result, state, thinker_out)

        return result


def event_to_dict(event: Any) -> dict[str, Any]:
    return {
        "type": event.type,
        "modality": event.modality,
        "payload": dict(event.payload),
        "is_final": bool(event.is_final),
    }


def text_output_requested(request: OmniRequest) -> bool:
    """Return True if text is among the requested output modalities.

    Reads ``request.metadata["output_modalities"]``; defaults to True when
    the field is absent (text is always produced unless explicitly excluded).
    """
    metadata = request.metadata
    if not isinstance(metadata, dict):
        return True
    else:
        pass
    modalities = metadata.get("output_modalities")
    if modalities is None:
        return True
    else:
        pass
    if isinstance(modalities, str):
        return modalities.lower() == "text"
    else:
        pass
    if isinstance(modalities, (list, tuple, set)):
        return any(str(m).lower() == "text" for m in modalities)
    else:
        pass
    return True


def attach_decode_final_metadata(
    result: dict[str, Any],
    state: MingOmniPipelineState,
    thinker_out: dict[str, Any],
) -> None:
    finish_reason = thinker_out.get("finish_reason")
    if finish_reason is not None:
        result.setdefault("finish_reason", finish_reason)
    else:
        pass
    result.setdefault("usage", build_text_usage(state, thinker_out))


def create_ming_streaming_detokenize_scheduler(
    model_path: str,
    *,
    stage_name: str = "decode",
) -> MingStreamingDetokenizeScheduler:
    from sglang_omni.models.ming_omni.components.common import load_ming_tokenizer

    tokenizer = load_ming_tokenizer(model_path)
    return MingStreamingDetokenizeScheduler(
        tokenizer=tokenizer,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        stage_name=stage_name,
    )
