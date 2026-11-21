"""Realtime output delivery and completion state."""

import asyncio
import json
import math
from collections import deque
from dataclasses import dataclass, replace

from sglang_omni.serve.realtime.control import Closed, Failure
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    AudioFinished,
    InputTranscription,
    OutputEvent,
    ResponseEvent,
    ResponseFinished,
    ResponseProgress,
    ResponseStarted,
    ResponseStatus,
    TextDelta,
    TextFinished,
)
from sglang_omni.serve.realtime.types import (
    PCM16_BYTES_PER_SAMPLE,
    ContextLimitError,
    Envelope,
    OutputBudgetError,
    ResponseRequestMeasurementOrigin,
    ResponseRequestMetrics,
    RuntimeLimits,
    Unit,
)

# Note (Junnan Li): The JSON of one server event around its audio or text: type, event and item ids, indexes.
ENVELOPE_BYTES = 256
RESPONSE_REQUEST_MEASUREMENT_ORIGIN: ResponseRequestMeasurementOrigin = {
    "ttft": (
        "model-turn request start to first non-empty text output; pending requests "
        "keep the latest start before response creation"
    ),
    "ttfp": (
        "model-turn request start to first non-empty audio output; pending requests "
        "keep the latest start before response creation"
    ),
}


def envelope_size_bytes(envelope: Envelope) -> int:
    """The bytes the event puts on the websocket: its base64 audio or UTF-8 text plus the envelope."""
    event = envelope.event
    metrics_bytes = (
        len(json.dumps(envelope.response_request_metrics).encode())
        if envelope.response_request_metrics is not None
        else 0
    )
    if isinstance(event, AudioDelta):
        return math.ceil(len(event.pcm) / 3) * 4 + ENVELOPE_BYTES + metrics_bytes
    elif isinstance(
        event, (TextDelta, TextFinished, ResponseFinished, InputTranscription)
    ):
        return len(event.text.encode()) + ENVELOPE_BYTES + metrics_bytes
    else:
        return ENVELOPE_BYTES + metrics_bytes


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
    request_started_at_seconds: float | None = None
    time_to_first_text_milliseconds: float | None = None
    time_to_first_pcm_milliseconds: float | None = None

    def request_metrics(self) -> ResponseRequestMetrics | None:
        if (
            self.time_to_first_text_milliseconds is None
            and self.time_to_first_pcm_milliseconds is None
        ):
            return None
        else:
            metrics: ResponseRequestMetrics = {
                "source": "server_monotonic_request_start",
                "measurement_origin": dict(RESPONSE_REQUEST_MEASUREMENT_ORIGIN),
            }
        if self.time_to_first_text_milliseconds is not None:
            metrics["ttft_ms"] = self.time_to_first_text_milliseconds
        else:
            pass
        if self.time_to_first_pcm_milliseconds is not None:
            metrics["ttfp_ms"] = self.time_to_first_pcm_milliseconds
        else:
            pass
        return metrics


class OutputBuffer:
    def __init__(self, limits: RuntimeLimits) -> None:
        self.limits = limits
        self.queued_envelopes: deque[tuple[Envelope, int]] = deque()
        self.queued_bytes = 0
        self.output_ready = asyncio.Event()
        self.next_chunk_index = 0
        self.responses: dict[str, ResponseState] = {}
        self.pending_request_started_at_seconds: float | None = None

    def mark_model_turn_request_started(self, started_at_seconds: float) -> None:
        """Keep the latest native request start until a response claims it."""
        self.pending_request_started_at_seconds = started_at_seconds

    def append_unbounded(self, envelope: Envelope, size_bytes: int) -> None:
        self.queued_envelopes.append((envelope, size_bytes))
        self.queued_bytes += size_bytes
        self.output_ready.set()

    def enqueue(self, envelope: Envelope) -> None:
        size_bytes = envelope_size_bytes(envelope)
        if self.queued_bytes + size_bytes > self.limits.max_output_bytes:
            raise OutputBudgetError()
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
        observed_at_seconds: float | None = None,
    ) -> None:
        response_id = event.response_id if isinstance(event, ResponseEvent) else None
        # Note (Junnan Li): Keep each response's negotiated modalities across hot updates.
        response_state = (
            self.responses.get(response_id) if response_id is not None else None
        )
        modalities = (
            response_state.output_modalities if response_state is not None else None
        ) or output_modalities
        if (
            isinstance(event, (AudioDelta, ResponseProgress, AudioFinished))
            and "audio" not in modalities
        ):
            return
        elif isinstance(event, ResponseFinished) and "audio" not in modalities:
            event = replace(event, has_audio=False)
        else:
            pass
        if isinstance(event, ResponseStarted):
            self.start_response(event.response_id, modalities)
        elif isinstance(
            event,
            (
                TextDelta,
                TextFinished,
                AudioDelta,
                ResponseProgress,
                AudioFinished,
                ResponseFinished,
            ),
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
            newly_observed_metrics = False
            if isinstance(event, (TextDelta, TextFinished)):
                response_text = (
                    response_state.text + event.text
                    if isinstance(event, TextDelta)
                    else event.text
                )
                if len(response_text) > self.limits.max_history_chars:
                    raise ContextLimitError()
                else:
                    pass
                response_state.text = response_text
                if (
                    isinstance(event, TextDelta)
                    and bool(event.text)
                    and response_state.time_to_first_text_milliseconds is None
                    and response_state.request_started_at_seconds is not None
                    and observed_at_seconds is not None
                ):
                    response_state.time_to_first_text_milliseconds = max(
                        (
                            observed_at_seconds
                            - response_state.request_started_at_seconds
                        )
                        * 1000.0,
                        0.0,
                    )
                    newly_observed_metrics = True
                else:
                    pass
            elif isinstance(event, AudioDelta) and len(event.pcm) % (
                PCM16_BYTES_PER_SAMPLE
            ):
                raise RuntimeError("producer emitted invalid PCM16")
            elif (
                isinstance(event, AudioDelta)
                and bool(event.pcm)
                and response_state.time_to_first_pcm_milliseconds is None
                and response_state.request_started_at_seconds is not None
                and observed_at_seconds is not None
            ):
                response_state.time_to_first_pcm_milliseconds = max(
                    (observed_at_seconds - response_state.request_started_at_seconds)
                    * 1000.0,
                    0.0,
                )
                newly_observed_metrics = True
            elif isinstance(event, ResponseFinished):
                response_state.is_terminal = True
                if event.status == "cancelled":
                    # Note (ruoyu): The listener stopped this reply, so its audio still waiting for a slow socket must not leave; the socket can hold seconds of it.
                    self.queued_envelopes = deque(
                        (envelope, size_bytes)
                        for envelope, size_bytes in self.queued_envelopes
                        if not (
                            isinstance(envelope.event, AudioDelta)
                            and envelope.event.response_id == event.response_id
                        )
                    )
                    self.queued_bytes = sum(
                        size_bytes for _, size_bytes in self.queued_envelopes
                    )
                else:
                    pass
            else:
                pass
        else:
            newly_observed_metrics = False
        response_request_metrics = (
            response_state.request_metrics()
            if response_state is not None
            and (newly_observed_metrics or isinstance(event, ResponseFinished))
            else None
        )
        self.enqueue(
            Envelope(
                event=event,
                unit=unit,
                chunk_index=self.next_chunk_index,
                output_modalities=tuple(modalities),
                response_request_metrics=response_request_metrics,
            )
        )
        self.next_chunk_index += 1

    def start_response(self, response_id: str, modalities: tuple[str, ...]) -> None:
        if response_id in self.responses:
            raise RuntimeError("duplicate response creation")
        else:
            self.responses[response_id] = ResponseState(
                output_modalities=modalities,
                request_started_at_seconds=self.pending_request_started_at_seconds,
            )
            self.pending_request_started_at_seconds = None

    def finish_responses(self, status: ResponseStatus, reason: str) -> None:
        """Queue at most three closing events per visible response."""
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
                    response_request_metrics=(
                        response_state.request_metrics()
                        if isinstance(closing_event, ResponseFinished)
                        else None
                    ),
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
