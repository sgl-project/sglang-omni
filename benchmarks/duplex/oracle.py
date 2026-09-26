"""Validate native duplex traces using client observations and protocol receipts."""

import base64
import binascii
import math
from collections import Counter
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from benchmarks.duplex.profiles import DEFAULT_PROFILE, PROFILES, ProfileName

Identifier = Annotated[str, Field(min_length=1)]
Nonnegative = Annotated[float, Field(ge=0)]

INPUT_SAMPLE_RATE = 16000
INPUT_BYTES_PER_MS = INPUT_SAMPLE_RATE * 2 / 1000
INPUT_BYTES_PER_S = INPUT_SAMPLE_RATE * 2
MAX_ADMISSION_ATTEMPTS = 3
SESSION_TIMEOUT_S = 240

REQUIRED_CAPABILITIES = {
    "interaction": "native",
    "native_full_duplex": True,
    "input_audio_format": {"type": "audio/pcm", "rate": INPUT_SAMPLE_RATE},
    "tail_policy": "pad",
    "microturn_ms": None,
    "client_commit": False,
    "proactive_output": False,
    "pressure_policy": "reject",
    "strict_order": True,
    "supports_server_interrupt": False,
    "supports_resume": False,
    "supports_truncate": False,
}

TERMINAL_REASONS = {
    "stop": ("completed", "sglang.input_audio.end"),
    "client_closed": ("cancelled", "session.close"),
}

MEDIA_DELTAS = (
    "response.output_audio.delta",
    "response.output_audio_transcript.delta",
    "response.output_text.delta",
)


class TraceRecord(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)
    direction: Literal["send", "receive", "error", "admission"]
    time_s: Nonnegative
    event: dict[str, JsonValue]


class AdmissionEvent(BaseModel):
    """The only diagnostic record allowed before a session exists."""

    model_config = ConfigDict(strict=True, extra="forbid")
    type: Literal["connection_denied"]
    http_status: Literal[503]
    attempt: Annotated[int, Field(ge=1, le=MAX_ADMISSION_ATTEMPTS)]


class MediaTime(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")
    t_start_ms: Nonnegative
    duration_ms: Nonnegative


class Metadata(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")
    seq: Annotated[int, Field(ge=0)] | None = None
    t_start_ms: Nonnegative | None = None
    media_time: MediaTime | None = None
    fatal: bool | None = None


class StatusDetails(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")
    reason: Identifier


class Response(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")
    id: Identifier
    status: str
    status_details: StatusDetails | None = None


class WireEvent(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow", allow_inf_nan=False)
    type: Identifier
    event_id: Identifier
    sglang: Metadata = Field(default_factory=Metadata)
    session: dict[str, JsonValue] | None = None
    response: Response | None = None
    response_id: Identifier | None = None
    item_id: Identifier | None = None
    unit_id: Identifier | None = None
    content_index: int | None = None
    output_index: int | None = None
    audio: str | None = None
    delta: str | None = None
    tail_policy: str | None = None
    client_event_id: Identifier | None = None
    seq: int | None = None
    accepted_end_ms: Nonnegative | None = None
    consumed_ms: Nonnegative | None = None
    discarded_ms: Nonnegative | None = None
    padding_ms: Nonnegative | None = None
    reason: str | None = None
    error: dict[str, JsonValue] | None = None


@dataclass
class SentCommand:
    event: WireEvent
    time_s: float
    input_end_ms: float


@dataclass
class ResponseState:
    terminal: str | None = None
    reason: str | None = None
    audio_done: bool = False
    audio_seen: bool = False
    item_id: str | None = None


def pcm_bytes(value: str | None) -> bytes:
    if value is None:
        raise ValueError("missing PCM payload")
    else:
        data = base64.b64decode(value, validate=True)
        if not data or len(data) % 2:
            raise ValueError("PCM16 payload must contain whole nonempty samples")
        else:
            return data


def evaluate_trace(
    records: list[dict[str, JsonValue]],
    *,
    scenario: Literal["continuous"],
    profile: ProfileName = DEFAULT_PROFILE,
) -> dict[str, JsonValue]:
    """Report protocol failures separately from an unexercised duplex scenario."""
    if scenario != "continuous":
        raise ValueError(f"unsupported scenario: {scenario}")
    contract = PROFILES[profile]
    unit_ms = contract.native_unit_ms
    unit_input_bytes = INPUT_SAMPLE_RATE * unit_ms // 1000 * 2
    required_capabilities = {
        **REQUIRED_CAPABILITIES,
        "native_unit_ms": unit_ms,
        "output_audio_format": {
            "type": "audio/pcm",
            "rate": contract.output_sample_rate,
        },
        "output_modalities": list(contract.output_modalities),
    }
    violations: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            violations.append(message)

    counts: Counter[tuple[str, str]] = Counter()
    ids: dict[str, set[str]] = {"send": set(), "receive": set()}
    sent: dict[str, SentCommand] = {}
    acknowledged: set[tuple[str, str | None]] = set()
    responses: dict[str, ResponseState] = {}
    input_times: list[float] = []
    audio_times: list[float] = []
    units: list[tuple[int, float]] = []
    input_bytes = output_bytes = next_seq = admissions = 0
    session_id = None
    updated = ended = drained = closed = False
    previous_time = -math.inf
    drain_after_eos_s = close_ack_s = None
    acknowledgments = {
        "session.updated": "session.update",
        "sglang.input_audio.accepted": "input_audio_buffer.append",
        "sglang.input_audio.ended": "sglang.input_audio.end",
        "sglang.input_audio.drained": "sglang.input_audio.end",
        "session.closed": "session.close",
    }

    for index, raw in enumerate(records):
        try:
            record = TraceRecord.model_validate(raw)
            check(record.time_s >= previous_time, f"record {index}: clock regressed")
            previous_time = record.time_s
            if record.direction == "admission":
                admission = AdmissionEvent.model_validate(record.event)
                check(
                    not ids["send"] and not ids["receive"],
                    "admission diagnostic after the session started",
                )
                check(
                    admission.attempt == admissions + 1,
                    "admission attempts are not contiguous from one",
                )
                admissions += 1
                continue
            if record.direction == "error":
                violations.append(f"client error: {record.event.get('message')}")
                continue
            event = WireEvent.model_validate(record.event)
            direction, typ, now = record.direction, event.type, record.time_s
            check(
                event.event_id not in ids[direction], f"duplicate {direction} event_id"
            )
            ids[direction].add(event.event_id)
            counts[direction, typ] += 1
            check(not closed, f"{typ} after session.closed")
            if direction == "send":
                if typ == "session.update":
                    check(
                        session_id is not None, "session.update before session.created"
                    )
                elif typ == "input_audio_buffer.append":
                    check(updated, "audio input before session.updated")
                    check(
                        not counts["send", "sglang.input_audio.end"], "input after EOS"
                    )
                    check(
                        event.sglang.seq == next_seq, "input sequence is not contiguous"
                    )
                    check(
                        event.sglang.t_start_ms is None
                        or math.isclose(
                            event.sglang.t_start_ms,
                            input_bytes / INPUT_BYTES_PER_MS,
                            rel_tol=0,
                            abs_tol=1e-7,
                        ),
                        "input media time is not sample-contiguous",
                    )
                    input_bytes += len(pcm_bytes(event.audio))
                    next_seq += 1
                    input_times.append(now)
                elif typ == "sglang.input_audio.end":
                    check(updated, "EOS before session.updated")
                elif typ == "session.close":
                    check(drained, "session.close before input drained")
                else:
                    violations.append(f"client command outside this scenario: {typ}")
                sent[event.event_id] = SentCommand(
                    event, now, input_bytes / INPUT_BYTES_PER_MS
                )
                continue

            cause = sent.get(event.client_event_id)
            if typ in acknowledgments:
                check(
                    cause is not None and cause.event.type == acknowledgments[typ],
                    f"{typ}: missing matching prior client event",
                )
                check(
                    (typ, event.client_event_id) not in acknowledged, f"duplicate {typ}"
                )
                acknowledged.add((typ, event.client_event_id))
            if typ == "session.created":
                check(session_id is None, "duplicate session.created")
                session = event.session or {}
                session_id = session.get("id")
                check(
                    isinstance(session_id, str) and bool(session_id),
                    "missing session ID",
                )
            elif typ == "session.updated":
                session = event.session or {}
                check(
                    session_id is not None and session.get("id") == session_id,
                    "session ID changed",
                )
                extension = session.get("sglang")
                grant = (
                    extension.get("granted") if isinstance(extension, dict) else None
                )
                check(isinstance(grant, dict), "missing granted capabilities")
                if isinstance(grant, dict):
                    for key, expected in required_capabilities.items():
                        check(
                            type(grant.get(key)) is type(expected)
                            and grant.get(key) == expected,
                            f"unsupported capability: {key}",
                        )
                    limits = grant.get("limits")
                    check(
                        isinstance(limits, dict)
                        and limits.get("session_timeout_s") == SESSION_TIMEOUT_S,
                        "unsupported capability: limits.session_timeout_s",
                    )
                updated = True
            elif typ == "sglang.input_audio.accepted":
                if cause is not None:
                    check(
                        event.seq == cause.event.sglang.seq,
                        "accepted input sequence mismatch",
                    )
                    check(
                        event.accepted_end_ms == cause.input_end_ms,
                        "accepted input duration mismatch",
                    )
            elif typ == "sglang.input_audio.ended":
                check(
                    event.accepted_end_ms == input_bytes / INPUT_BYTES_PER_MS,
                    "ended input duration mismatch",
                )
                check(event.tail_policy == "pad", "ended tail policy mismatch")
                ended = True
            elif typ == "sglang.input_audio.drained":
                check(ended, "input drained before ended receipt")
                expected_ms = input_bytes / INPUT_BYTES_PER_MS
                check(
                    event.accepted_end_ms == expected_ms,
                    "drained input duration mismatch",
                )
                check(
                    event.consumed_ms == expected_ms,
                    "drained consumed duration mismatch",
                )
                check(event.discarded_ms == 0, "accepted input was discarded")
                expected_padding = (
                    math.ceil(expected_ms / unit_ms) * unit_ms - expected_ms
                )
                check(event.padding_ms == expected_padding, "drained padding mismatch")
                drained = True
                if cause is not None:
                    drain_after_eos_s = now - cause.time_s
            elif typ == "sglang.unit.done":
                prefix, _, number = (event.unit_id or "").partition("_")
                media = event.sglang.media_time
                if prefix != "unit" or not number.isdigit():
                    violations.append(f"invalid native unit ID: {event.unit_id}")
                elif media is None:
                    violations.append("unit receipt missing media time")
                else:
                    unit = int(number)
                    check(
                        not units or unit > units[-1][0],
                        "native unit receipts are not increasing",
                    )
                    check(
                        media.t_start_ms == unit * unit_ms,
                        "native unit media time is not unit-aligned",
                    )
                    units.append((unit, media.duration_ms))
            elif typ == "response.created":
                check(updated, "response before session.updated")
                if event.response is None:
                    violations.append("response.created missing response")
                else:
                    check(
                        event.response.id not in responses, "duplicate response.created"
                    )
                    check(
                        event.response.status == "in_progress",
                        "invalid created response status",
                    )
                    responses[event.response.id] = ResponseState()
            elif typ.startswith("response."):
                forced = counts["send", "session.close"]
                check(
                    not drained or (typ not in MEDIA_DELTAS and bool(forced)),
                    f"{typ} after drain",
                )
                response_id = event.response.id if event.response else event.response_id
                state = responses.get(response_id)
                check(state is not None, f"{typ}: unknown response")
                if state is not None:
                    check(
                        state.terminal is None, f"{typ}: output after response terminal"
                    )
                    if typ == "response.done":
                        status = event.response.status if event.response else None
                        details = (
                            event.response.status_details if event.response else None
                        )
                        reason = details.reason if details is not None else None
                        check(
                            reason in TERMINAL_REASONS,
                            f"unsupported response terminal reason: {reason}",
                        )
                        if reason in TERMINAL_REASONS:
                            expected_status, command = TERMINAL_REASONS[reason]
                            check(
                                status == expected_status,
                                f"terminal status {status} contradicts reason {reason}",
                            )
                            if reason != "stop" or contract.stop_requires_eos:
                                check(
                                    bool(counts["send", command]),
                                    f"terminal reason {reason} without its client command",
                                )
                        # Note (wenyao): The pump consumes EOS before enqueueing drain.
                        check(
                            reason != "stop" or not drained,
                            "native completed terminal after the drain receipt",
                        )
                        check(
                            not state.audio_seen or state.audio_done,
                            "missing output_audio.done",
                        )
                        state.terminal, state.reason = status, reason
                    elif typ == "response.output_audio.delta":
                        check(not state.audio_done, "audio after output_audio.done")
                        check(event.item_id is not None, "audio missing item ID")
                        check(
                            state.item_id is None or state.item_id == event.item_id,
                            "audio item ID changed",
                        )
                        check(
                            event.content_index == 0 and event.output_index == 0,
                            "invalid audio indexes",
                        )
                        output_bytes += len(pcm_bytes(event.delta))
                        audio_times.append(now)
                        state.audio_seen = True
                        state.item_id = event.item_id
                    elif typ == "response.output_audio.done":
                        check(not state.audio_done, "duplicate output_audio.done")
                        state.audio_done = True
                    elif typ in (
                        "response.output_audio_transcript.delta",
                        "response.output_text.delta",
                    ):
                        check(event.delta is not None, "text delta missing content")
                    elif typ not in (
                        "response.output_audio_transcript.done",
                        "response.output_text.done",
                    ):
                        violations.append(f"unsupported server event: {typ}")
            elif typ == "session.closed":
                check(drained, "session.closed before input drained")
                check(
                    event.reason == "client_closed", "unexpected session close reason"
                )
                closed = True
                if cause is not None:
                    close_ack_s = now - cause.time_s
            elif typ == "error":
                error = event.error or {}
                violations.append(
                    f"server error: code={error.get('code')} "
                    f"fatal={event.sglang.fatal}: {error.get('message')}"
                )
            else:
                violations.append(f"unsupported server event: {typ}")
        except (ValidationError, ValueError, binascii.Error) as exc:
            violations.append(f"record {index}: {exc}")

    for direction, typ in (
        ("receive", "session.created"),
        ("send", "session.update"),
        ("receive", "session.updated"),
        ("send", "sglang.input_audio.end"),
        ("receive", "sglang.input_audio.ended"),
        ("receive", "sglang.input_audio.drained"),
        ("send", "session.close"),
        ("receive", "session.closed"),
    ):
        check(counts[direction, typ] == 1, f"expected exactly one {direction} {typ}")
    check(bool(input_times), "missing audio input")
    input_ms = input_bytes / INPUT_BYTES_PER_MS
    expected_units = math.ceil(input_bytes / unit_input_bytes)
    # Note (wenyao): Only unit-aligned input produces an extra empty EOS unit.
    nonempty = [duration for _, duration in units if duration]
    check(
        [unit for unit, _ in units] == list(range(len(units))),
        "native unit receipts are not contiguous from zero",
    )
    check(
        len(nonempty) == expected_units,
        f"expected {expected_units} nonempty native units, "
        f"observed {len(nonempty)}",
    )
    check(
        len(units) == len(nonempty)
        or (
            len(units) == len(nonempty) + 1
            and not units[-1][1]
            and not input_bytes % unit_input_bytes
        ),
        "an empty EOS unit requires exactly unit-aligned input",
    )
    check(
        sum(duration for _, duration in units) == input_ms,
        "native unit durations do not account for the accepted input",
    )
    if contract.continuous_output:
        expected_output = (
            expected_units * contract.output_sample_rate * unit_ms // 1000 * 2
        )
        check(
            output_bytes == expected_output,
            f"output audio is not conserved: {output_bytes} of "
            f"{expected_output} bytes for {expected_units} units",
        )
    for event_id, sent_command in sent.items():
        if sent_command.event.type == "input_audio_buffer.append":
            check(
                ("sglang.input_audio.accepted", event_id) in acknowledged,
                "missing input acceptance receipt",
            )
    for response_id, state in responses.items():
        check(
            state.terminal == "completed" and state.reason == "stop",
            f"response {response_id} ended {state.terminal}/{state.reason}, "
            "expected completed/stop",
        )
    coverage = {
        "input_output_overlap": bool(audio_times)
        and any(audio_times[0] < now < audio_times[-1] for now in input_times)
    }
    metrics = {
        "input_audio_s": input_bytes / INPUT_BYTES_PER_S,
        "output_audio_s": output_bytes / (contract.output_sample_rate * 2),
        "admission_denials": admissions,
        "first_audio_packet_s": (
            audio_times[0] - input_times[0] if audio_times and input_times else None
        ),
        "audio_packet_gap_max_s": max(
            (right - left for left, right in zip(audio_times, audio_times[1:])),
            default=None,
        ),
        "input_output_overlap": coverage["input_output_overlap"],
        "drain_after_eos_s": drain_after_eos_s,
        "close_ack_s": close_ack_s,
    }
    return {
        "status": (
            "fail"
            if violations
            else (
                "pass"
                if not contract.continuous_output or all(coverage.values())
                else "not_exercised"
            )
        ),
        "violations": violations,
        "coverage": coverage,
        "metrics": metrics,
    }
