"""Realtime output delivery and completion state."""

import asyncio
from collections import deque
from dataclasses import dataclass, replace

from sglang_omni.serve.realtime.control import Closed, Failure
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    AudioFinished,
    ContextLimitError,
    OutputEvent,
    ResponseEvent,
    ResponseFinished,
    ResponseStarted,
    ResponseStatus,
    TextDelta,
    TextFinished,
)
from sglang_omni.serve.realtime.types import (
    PCM16_BYTES_PER_SAMPLE,
    Envelope,
    RuntimeLimits,
    Unit,
)


def envelope_size_bytes(envelope: Envelope) -> int:
    return len(repr(envelope).encode())


@dataclass(kw_only=True)
class ResponseState:
    output_modalities: tuple[str, ...]
    item_id: str = ""
    text: str = ""
    is_terminal: bool = False
    is_visible: bool = False
    is_terminal_sent: bool = False
    has_close_terminals_queued: bool = False
    is_text_done_sent: bool = False
    is_audio_done_sent: bool = False
    is_audio_visible: bool = False


class OutputBuffer:
    def __init__(self, limits: RuntimeLimits) -> None:
        self.limits = limits
        self.queued_envelopes: deque[tuple[Envelope, int]] = deque()
        self.queued_bytes = 0
        self.output_ready = asyncio.Event()
        self.next_chunk_index = 0
        self.responses: dict[str, ResponseState] = {}

    def append_unbounded(self, envelope: Envelope, size_bytes: int) -> None:
        self.queued_envelopes.append((envelope, size_bytes))
        self.queued_bytes += size_bytes
        self.output_ready.set()

    def enqueue(self, envelope: Envelope) -> None:
        size_bytes = envelope_size_bytes(envelope)
        if (
            len(self.queued_envelopes) >= self.limits.max_output_events
            or self.queued_bytes + size_bytes > self.limits.max_output_bytes
        ):
            raise RuntimeError("outbound event budget exhausted")
        else:
            pass
        self.append_unbounded(envelope, size_bytes)

    def dequeue(self) -> Envelope | None:
        if not self.queued_envelopes:
            return None
        else:
            envelope, size_bytes = self.queued_envelopes.popleft()
            self.queued_bytes -= size_bytes
            return envelope

    def clear(self) -> None:
        self.queued_envelopes.clear()
        self.queued_bytes = 0

    def emit(
        self,
        event: OutputEvent,
        unit: Unit | None,
        output_modalities: tuple[str, ...],
    ) -> None:
        response_id = event.response_id if isinstance(event, ResponseEvent) else None
        # Note (Junnan Li): Keep each response's negotiated modalities across hot updates.
        response_state = (
            self.responses.get(response_id) if response_id is not None else None
        )
        modalities = (
            response_state.output_modalities if response_state is not None else None
        ) or output_modalities
        if isinstance(event, (AudioDelta, AudioFinished)) and "audio" not in modalities:
            return
        elif isinstance(event, ResponseFinished) and "audio" not in modalities:
            event = replace(event, has_audio=False)
        else:
            pass
        if isinstance(event, ResponseStarted):
            self.start_response(event.response_id, modalities)
        elif isinstance(
            event,
            (TextDelta, TextFinished, AudioDelta, AudioFinished, ResponseFinished),
        ):
            response_state = self.responses.get(event.response_id)
            if response_state is None:
                raise RuntimeError("response output precedes creation")
            elif response_state.is_terminal:
                return
            elif response_state.item_id and event.item_id != response_state.item_id:
                raise RuntimeError("only one message item per response is supported")
            else:
                pass
            response_state.item_id = event.item_id
            if isinstance(event, (TextDelta, TextFinished)):
                response_text = (
                    response_state.text + event.text
                    if isinstance(event, TextDelta)
                    else event.text
                )
                if len(response_text) > self.limits.max_history_chars:
                    raise ContextLimitError("response text context limit")
                else:
                    pass
                response_state.text = response_text
            elif isinstance(event, AudioDelta) and len(event.pcm) % (
                PCM16_BYTES_PER_SAMPLE
            ):
                raise RuntimeError("producer emitted invalid PCM16")
            elif isinstance(event, ResponseFinished):
                response_state.is_terminal = True
            else:
                pass
        else:
            pass
        self.enqueue(
            Envelope(
                event=event,
                unit=unit,
                chunk_index=self.next_chunk_index,
                output_modalities=tuple(modalities),
            )
        )
        self.next_chunk_index += 1

    def start_response(self, response_id: str, modalities: tuple[str, ...]) -> None:
        if response_id in self.responses:
            raise RuntimeError("duplicate response creation")
        elif len(self.responses) >= self.limits.max_output_events:
            raise RuntimeError("unfinished response budget exhausted")
        else:
            self.responses[response_id] = ResponseState(output_modalities=modalities)

    def finish_responses(self, status: ResponseStatus, reason: str) -> None:
        """Reserve at most three closing events per bounded response slot."""
        for response_id, response_state in list(self.responses.items()):
            if (
                response_state.is_terminal_sent
                or response_state.has_close_terminals_queued
            ):
                continue
            elif not response_state.is_visible:
                self.responses.pop(response_id)
                continue
            else:
                pass
            response_state.is_terminal = True
            response_state.has_close_terminals_queued = True
            item_id = response_state.item_id or "item_" + response_id
            closing_events: list[OutputEvent] = []
            if (
                not response_state.is_text_done_sent
                and response_state.output_modalities
            ):
                closing_events.append(
                    TextFinished(response_id, item_id, response_state.text)
                )
            else:
                pass
            if (
                response_state.is_audio_visible
                and not response_state.is_audio_done_sent
            ):
                closing_events.append(AudioFinished(response_id, item_id))
            else:
                pass
            closing_events.append(
                ResponseFinished(
                    response_id,
                    item_id,
                    response_state.text,
                    response_state.is_audio_visible,
                    status,
                    reason,
                )
            )
            for closing_event in closing_events:
                envelope = Envelope(
                    event=closing_event,
                    is_control=True,
                    output_modalities=tuple(response_state.output_modalities),
                )
                self.append_unbounded(envelope, envelope_size_bytes(envelope))

    def before_send(self, envelope: Envelope) -> None:
        """Close must observe lifecycle visibility before the socket send yields."""
        event = envelope.event
        if isinstance(event, ResponseStarted):
            self.responses[event.response_id].is_visible = True
        elif isinstance(event, ResponseFinished):
            self.responses[event.response_id].is_terminal_sent = True
        elif isinstance(event, TextFinished):
            self.responses[event.response_id].is_text_done_sent = True
        elif isinstance(event, AudioFinished):
            self.responses[event.response_id].is_audio_done_sent = True
        elif isinstance(event, AudioDelta):
            self.responses[event.response_id].is_audio_visible = True
        else:
            pass

    def sent(self, envelope: Envelope) -> None:
        event = envelope.event
        if isinstance(event, ResponseFinished):
            self.responses.pop(event.response_id)
        elif isinstance(event, Closed):
            self.responses.clear()
        else:
            pass

    def enqueue_terminal(self, event: Failure | Closed) -> None:
        # Note (Junnan Li): Terminal notifications must remain deliverable after media overflow.
        self.queued_envelopes = deque(
            (envelope, size_bytes)
            for envelope, size_bytes in self.queued_envelopes
            if isinstance(envelope.event, Failure)
            or (
                envelope.is_control
                and isinstance(
                    envelope.event, (TextFinished, AudioFinished, ResponseFinished)
                )
            )
        )
        self.queued_bytes = sum(size_bytes for _, size_bytes in self.queued_envelopes)
        envelope = Envelope(event=event, is_control=True)
        self.append_unbounded(envelope, envelope_size_bytes(envelope))
