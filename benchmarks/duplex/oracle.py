"""Validate duplex traces using client observations and protocol receipts."""

import base64
import binascii
import math
from collections import Counter

from pydantic import JsonValue, ValidationError

from benchmarks.duplex.oracle_models import (
    AdmissionEvent,
    Identifier,
    MediaTime,
    Metadata,
    Nonnegative,
    Response,
    ResponseState,
    SentCommand,
    StatusDetails,
    TraceRecord,
    WireEvent,
)
from benchmarks.duplex.profiles import (
    DEFAULT_PROFILE,
    PROFILES,
    DuplexProfile,
    ProfileName,
)

__all__ = [
    "AdmissionEvent",
    "Identifier",
    "MediaTime",
    "Metadata",
    "Nonnegative",
    "Response",
    "ResponseState",
    "SentCommand",
    "StatusDetails",
    "TraceRecord",
    "WireEvent",
    "evaluate_trace",
    "pcm_bytes",
]

INPUT_SAMPLE_RATE = 16000
INPUT_BYTES_PER_MS = INPUT_SAMPLE_RATE * 2 / 1000
INPUT_BYTES_PER_S = INPUT_SAMPLE_RATE * 2

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
# note (luojiaxuan): Legacy terminals: completed responses carry the generation
# finish reason; a turn-detected cancel is legitimate only after speech_started.
LEGACY_TERMINALS = {("completed", "stop"), ("completed", "length")}
LEGACY_INTERRUPTED_TERMINAL = ("cancelled", "turn_detected")
LEGACY_OBSERVATIONS = (
    "input_audio_buffer.speech_stopped",
    "input_audio_buffer.committed",
    "conversation.item.input_audio_transcription.delta",
    "conversation.item.input_audio_transcription.completed",
    "conversation.item.truncated",
)


def pcm_bytes(value: str | None) -> bytes:
    if value is None:
        raise ValueError("missing PCM payload")
    else:
        pcm_data = base64.b64decode(value, validate=True)
        if not pcm_data or len(pcm_data) % 2:
            raise ValueError("PCM16 payload must contain whole nonempty samples")
        else:
            return pcm_data


def evaluate_trace(
    records: list[dict[str, JsonValue]],
    *,
    profile: ProfileName = DEFAULT_PROFILE,
    max_admission_attempts: int | None = None,
) -> dict[str, JsonValue]:
    """Report protocol failures separately from an unexercised duplex scenario.

    max_admission_attempts is the retry limit the recording client used, read from
    its manifest; None checks only that denials count up from one.
    """
    profile_contract = PROFILES[profile]
    if profile_contract.protocol == "legacy":
        return evaluate_legacy_trace(
            records,
            profile_contract=profile_contract,
            max_admission_attempts=max_admission_attempts,
        )
    else:
        pass
    unit_duration_ms = profile_contract.native_unit_ms
    input_bytes_per_unit = INPUT_SAMPLE_RATE * unit_duration_ms // 1000 * 2
    required_capabilities = {
        **REQUIRED_CAPABILITIES,
        "native_unit_ms": unit_duration_ms,
        "output_audio_format": {
            "type": "audio/pcm",
            "rate": profile_contract.output_sample_rate,
        },
    }
    violations: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            violations.append(message)
        else:
            pass

    counts: Counter[tuple[str, str]] = Counter()
    event_ids: dict[str, set[str]] = {"send": set(), "receive": set()}
    sent_commands: dict[str, SentCommand] = {}
    acknowledged: set[tuple[str, str | None]] = set()
    response_states: dict[str, ResponseState] = {}
    input_times: list[float] = []
    audio_times: list[float] = []
    units: list[tuple[int, float]] = []
    input_byte_count = output_byte_count = next_sequence = admissions = 0
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

    for index, raw_record in enumerate(records):
        try:
            record = TraceRecord.model_validate(raw_record)
            check(record.time_s >= previous_time, f"record {index}: clock regressed")
            previous_time = record.time_s
            if record.direction == "admission":
                admission = AdmissionEvent.model_validate(record.event)
                check(
                    not event_ids["send"] and not event_ids["receive"],
                    "admission diagnostic after the session started",
                )
                check(
                    admission.attempt == admissions + 1,
                    "admission attempts are not contiguous from one",
                )
                check(
                    max_admission_attempts is None
                    or admission.attempt <= max_admission_attempts,
                    f"admission attempt {admission.attempt} exceeds the recorded "
                    f"limit {max_admission_attempts}",
                )
                admissions += 1
                continue
            else:
                pass
            if record.direction == "error":
                violations.append(f"client error: {record.event.get('message')}")
                continue
            else:
                pass
            event = WireEvent.model_validate(record.event)
            direction, event_type, event_time_s = (
                record.direction,
                event.type,
                record.time_s,
            )
            check(
                event.event_id not in event_ids[direction],
                f"duplicate {direction} event_id",
            )
            event_ids[direction].add(event.event_id)
            counts[direction, event_type] += 1
            check(not closed, f"{event_type} after session.closed")
            if direction == "send":
                if event_type == "session.update":
                    check(
                        session_id is not None, "session.update before session.created"
                    )
                elif event_type == "input_audio_buffer.append":
                    check(updated, "audio input before session.updated")
                    check(
                        not counts["send", "sglang.input_audio.end"], "input after EOS"
                    )
                    check(
                        event.sglang.seq == next_sequence,
                        "input sequence is not contiguous",
                    )
                    check(
                        event.sglang.t_start_ms is None
                        or math.isclose(
                            event.sglang.t_start_ms,
                            input_byte_count / INPUT_BYTES_PER_MS,
                            rel_tol=0,
                            abs_tol=1e-7,
                        ),
                        "input media time is not sample-contiguous",
                    )
                    input_byte_count += len(pcm_bytes(event.audio))
                    next_sequence += 1
                    input_times.append(event_time_s)
                elif event_type == "sglang.input_audio.end":
                    check(updated, "EOS before session.updated")
                elif event_type == "session.close":
                    check(drained, "session.close before input drained")
                else:
                    violations.append(
                        f"client command outside this scenario: {event_type}"
                    )
                sent_commands[event.event_id] = SentCommand(
                    event, event_time_s, input_byte_count / INPUT_BYTES_PER_MS
                )
                continue
            else:
                pass

            client_command = sent_commands.get(event.client_event_id)
            if event_type in acknowledgments:
                check(
                    client_command is not None
                    and client_command.event.type == acknowledgments[event_type],
                    f"{event_type}: missing matching prior client event",
                )
                check(
                    (event_type, event.client_event_id) not in acknowledged,
                    f"duplicate {event_type}",
                )
                acknowledged.add((event_type, event.client_event_id))
            else:
                pass
            if event_type == "session.created":
                check(session_id is None, "duplicate session.created")
                session_fields = event.session or {}
                session_id = session_fields.get("id")
                check(
                    isinstance(session_id, str) and bool(session_id),
                    "missing session ID",
                )
            elif event_type == "session.updated":
                session_fields = event.session or {}
                check(
                    session_id is not None and session_fields.get("id") == session_id,
                    "session ID changed",
                )
                session_extension = session_fields.get("sglang")
                granted_capabilities = (
                    session_extension.get("granted")
                    if isinstance(session_extension, dict)
                    else None
                )
                check(
                    isinstance(granted_capabilities, dict),
                    "missing granted capabilities",
                )
                if isinstance(granted_capabilities, dict):
                    for (
                        capability_name,
                        required_value,
                    ) in required_capabilities.items():
                        check(
                            type(granted_capabilities.get(capability_name))
                            is type(required_value)
                            and granted_capabilities.get(capability_name)
                            == required_value,
                            f"unsupported capability: {capability_name}",
                        )
                    modalities = granted_capabilities.get("output_modalities")
                    check(
                        isinstance(modalities, list)
                        and all(isinstance(item, str) for item in modalities)
                        and "audio" in modalities,
                        "unsupported capability: output_modalities",
                    )
                else:
                    pass
                updated = True
            elif event_type == "sglang.input_audio.accepted":
                if client_command is not None:
                    check(
                        event.seq == client_command.event.sglang.seq,
                        "accepted input sequence mismatch",
                    )
                    check(
                        event.accepted_end_ms == client_command.input_end_ms,
                        "accepted input duration mismatch",
                    )
                else:
                    pass
            elif event_type == "sglang.input_audio.ended":
                check(
                    event.accepted_end_ms == input_byte_count / INPUT_BYTES_PER_MS,
                    "ended input duration mismatch",
                )
                check(event.tail_policy == "pad", "ended tail policy mismatch")
                ended = True
            elif event_type == "sglang.input_audio.drained":
                check(ended, "input drained before ended receipt")
                expected_ms = input_byte_count / INPUT_BYTES_PER_MS
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
                    math.ceil(expected_ms / unit_duration_ms) * unit_duration_ms
                    - expected_ms
                )
                check(event.padding_ms == expected_padding, "drained padding mismatch")
                drained = True
                if client_command is not None:
                    drain_after_eos_s = event_time_s - client_command.time_s
                else:
                    pass
            elif event_type == "sglang.unit.done":
                unit_prefix, _, unit_number = (event.unit_id or "").partition("_")
                media = event.sglang.media_time
                if unit_prefix != "unit" or not unit_number.isdigit():
                    violations.append(f"invalid native unit ID: {event.unit_id}")
                elif media is None:
                    violations.append("unit receipt missing media time")
                else:
                    unit_index = int(unit_number)
                    check(
                        not units or unit_index > units[-1][0],
                        "native unit receipts are not increasing",
                    )
                    check(
                        media.t_start_ms == unit_index * unit_duration_ms,
                        "native unit media time is not unit-aligned",
                    )
                    units.append((unit_index, media.duration_ms))
            elif event_type == "response.created":
                check(updated, "response before session.updated")
                if event.response is None:
                    violations.append("response.created missing response")
                else:
                    check(
                        event.response.id not in response_states,
                        "duplicate response.created",
                    )
                    check(
                        event.response.status == "in_progress",
                        "invalid created response status",
                    )
                    response_states[event.response.id] = ResponseState()
            elif event_type.startswith("response."):
                close_requests = counts["send", "session.close"]
                check(
                    not drained
                    or (event_type not in MEDIA_DELTAS and bool(close_requests)),
                    f"{event_type} after drain",
                )
                response_id = event.response.id if event.response else event.response_id
                response_state = response_states.get(response_id)
                check(response_state is not None, f"{event_type}: unknown response")
                if response_state is not None:
                    check(
                        response_state.terminal is None,
                        f"{event_type}: output after response terminal",
                    )
                    if event_type == "response.done":
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
                            expected_status, required_command_type = TERMINAL_REASONS[
                                reason
                            ]
                            check(
                                status == expected_status,
                                f"terminal status {status} contradicts reason {reason}",
                            )
                            if reason != "stop" or profile_contract.stop_requires_eos:
                                check(
                                    bool(counts["send", required_command_type]),
                                    f"terminal reason {reason} without its client command",
                                )
                            else:
                                pass
                        else:
                            pass
                        # note (wenyao): The pump consumes EOS before enqueueing drain.
                        check(
                            reason != "stop" or not drained,
                            "native completed terminal after the drain receipt",
                        )
                        check(
                            not response_state.audio_seen or response_state.audio_done,
                            "missing output_audio.done",
                        )
                        response_state.terminal, response_state.reason = status, reason
                    elif event_type == "response.output_audio.delta":
                        check(
                            not response_state.audio_done,
                            "audio after output_audio.done",
                        )
                        check(event.item_id is not None, "audio missing item ID")
                        check(
                            response_state.item_id is None
                            or response_state.item_id == event.item_id,
                            "audio item ID changed",
                        )
                        check(
                            event.content_index == 0 and event.output_index == 0,
                            "invalid audio indexes",
                        )
                        output_byte_count += len(pcm_bytes(event.delta))
                        audio_times.append(event_time_s)
                        response_state.audio_seen = True
                        response_state.item_id = event.item_id
                    elif event_type == "response.output_audio.done":
                        check(
                            not response_state.audio_done, "duplicate output_audio.done"
                        )
                        response_state.audio_done = True
                    elif event_type in (
                        "response.output_audio_transcript.delta",
                        "response.output_text.delta",
                    ):
                        check(event.delta is not None, "text delta missing content")
                    elif event_type not in (
                        "response.output_audio_transcript.done",
                        "response.output_text.done",
                    ):
                        violations.append(f"unsupported server event: {event_type}")
                    else:
                        pass
                else:
                    pass
            elif event_type == "session.closed":
                check(drained, "session.closed before input drained")
                check(
                    event.reason == "client_closed", "unexpected session close reason"
                )
                closed = True
                if client_command is not None:
                    close_ack_s = event_time_s - client_command.time_s
                else:
                    pass
            elif event_type == "error":
                error = event.error or {}
                violations.append(
                    f"server error: code={error.get('code')} "
                    f"fatal={event.sglang.fatal}: {error.get('message')}"
                )
            else:
                violations.append(f"unsupported server event: {event_type}")
        except (ValidationError, ValueError, binascii.Error) as exc:
            violations.append(f"record {index}: {exc}")

    for direction, event_type in (
        ("receive", "session.created"),
        ("send", "session.update"),
        ("receive", "session.updated"),
        ("send", "sglang.input_audio.end"),
        ("receive", "sglang.input_audio.ended"),
        ("receive", "sglang.input_audio.drained"),
        ("send", "session.close"),
        ("receive", "session.closed"),
    ):
        check(
            counts[direction, event_type] == 1,
            f"expected exactly one {direction} {event_type}",
        )
    check(bool(input_times), "missing audio input")
    input_ms = input_byte_count / INPUT_BYTES_PER_MS
    expected_units = math.ceil(input_byte_count / input_bytes_per_unit)
    # note (wenyao): Only unit-aligned input produces an extra empty EOS unit.
    nonempty_unit_durations = [duration for _, duration in units if duration]
    check(
        [unit_index for unit_index, _ in units] == list(range(len(units))),
        "native unit receipts are not contiguous from zero",
    )
    check(
        len(nonempty_unit_durations) == expected_units,
        f"expected {expected_units} nonempty native units, "
        f"observed {len(nonempty_unit_durations)}",
    )
    check(
        len(units) == len(nonempty_unit_durations)
        or (
            len(units) == len(nonempty_unit_durations) + 1
            and not units[-1][1]
            and not input_byte_count % input_bytes_per_unit
        ),
        "an empty EOS unit requires exactly unit-aligned input",
    )
    check(
        sum(duration for _, duration in units) == input_ms,
        "native unit durations do not account for the accepted input",
    )
    if profile_contract.continuous_output:
        expected_output = (
            expected_units
            * profile_contract.output_sample_rate
            * unit_duration_ms
            // 1000
            * 2
        )
        check(
            output_byte_count == expected_output,
            f"output audio is not conserved: {output_byte_count} of "
            f"{expected_output} bytes for {expected_units} units",
        )
    else:
        pass
    for event_id, sent_command in sent_commands.items():
        if sent_command.event.type == "input_audio_buffer.append":
            check(
                ("sglang.input_audio.accepted", event_id) in acknowledged,
                "missing input acceptance receipt",
            )
        else:
            pass
    for response_id, response_state in response_states.items():
        check(
            response_state.terminal == "completed" and response_state.reason == "stop",
            f"response {response_id} ended {response_state.terminal}/{response_state.reason}, "
            "expected completed/stop",
        )
    coverage = {
        "input_output_overlap": bool(audio_times)
        and any(
            audio_times[0] < event_time_s < audio_times[-1]
            for event_time_s in input_times
        )
    }
    metrics = {
        "input_audio_s": input_byte_count / INPUT_BYTES_PER_S,
        "output_audio_s": output_byte_count / (profile_contract.output_sample_rate * 2),
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
                if not profile_contract.continuous_output or all(coverage.values())
                else "not_exercised"
            )
        ),
        "violations": violations,
        "coverage": coverage,
        "metrics": metrics,
    }


def evaluate_legacy_trace(
    records: list[dict[str, JsonValue]],
    *,
    profile_contract: DuplexProfile,
    max_admission_attempts: int | None,
) -> dict[str, JsonValue]:
    """Validate a turn-based session recorded over the legacy conversation protocol.

    The facade acknowledges session.update only, so the n-th session.updated answers
    the n-th session.update: the configure update must echo the requested
    turn_detection.type and grant audio, and the liveness probe after the input must
    be answered. Responses still open when the client closes are not violations.
    """
    violations: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            violations.append(message)
        else:
            pass

    counts: Counter[tuple[str, str]] = Counter()
    event_ids: dict[str, set[str]] = {"send": set(), "receive": set()}
    response_states: dict[str, ResponseState] = {}
    interrupted: set[str] = set()
    input_times: list[float] = []
    audio_times: list[float] = []
    input_byte_count = output_byte_count = next_sequence = admissions = 0
    updates_sent = updates_acknowledged = 0
    session_id = requested_type = None
    probe_sent_s = probe_ack_s = None
    previous_time = -math.inf

    for index, raw_record in enumerate(records):
        try:
            record = TraceRecord.model_validate(raw_record)
            check(record.time_s >= previous_time, f"record {index}: clock regressed")
            previous_time = record.time_s
            if record.direction == "admission":
                admission = AdmissionEvent.model_validate(record.event)
                check(
                    not event_ids["send"] and not event_ids["receive"],
                    "admission diagnostic after the session started",
                )
                check(
                    admission.attempt == admissions + 1,
                    "admission attempts are not contiguous from one",
                )
                check(
                    max_admission_attempts is None
                    or admission.attempt <= max_admission_attempts,
                    f"admission attempt {admission.attempt} exceeds the recorded "
                    f"limit {max_admission_attempts}",
                )
                admissions += 1
                continue
            else:
                pass
            if record.direction == "error":
                violations.append(f"client error: {record.event.get('message')}")
                continue
            else:
                pass
            event = WireEvent.model_validate(record.event)
            direction, event_type, event_time_s = (
                record.direction,
                event.type,
                record.time_s,
            )
            check(
                event.event_id not in event_ids[direction],
                f"duplicate {direction} event_id",
            )
            event_ids[direction].add(event.event_id)
            counts[direction, event_type] += 1
            session_fields = event.session or {}
            if direction == "send":
                if event_type == "session.update":
                    check(
                        session_id is not None, "session.update before session.created"
                    )
                    if updates_sent == 0:
                        turn = session_fields.get("turn_detection")
                        requested_type = (
                            turn.get("type") if isinstance(turn, dict) else None
                        )
                        check(
                            isinstance(requested_type, str),
                            "configure update lacks turn_detection.type",
                        )
                        modalities = session_fields.get("modalities")
                        check(
                            isinstance(modalities, list) and "audio" in modalities,
                            "configure update does not request audio output",
                        )
                    else:
                        check(
                            session_fields == {},
                            "liveness probe is not a no-op session.update",
                        )
                        check(bool(input_times), "liveness probe before audio input")
                        probe_sent_s = event_time_s
                    updates_sent += 1
                elif event_type == "input_audio_buffer.append":
                    check(
                        updates_acknowledged > 0, "audio input before session.updated"
                    )
                    check(probe_sent_s is None, "input after the liveness probe")
                    check(
                        event.sglang.seq == next_sequence,
                        "input sequence is not contiguous",
                    )
                    check(
                        event.sglang.t_start_ms is None
                        or math.isclose(
                            event.sglang.t_start_ms,
                            input_byte_count / INPUT_BYTES_PER_MS,
                            rel_tol=0,
                            abs_tol=1e-7,
                        ),
                        "input media time is not sample-contiguous",
                    )
                    input_byte_count += len(pcm_bytes(event.audio))
                    next_sequence += 1
                    input_times.append(event_time_s)
                else:
                    violations.append(
                        f"client command outside this scenario: {event_type}"
                    )
                continue
            else:
                pass

            if event_type == "session.created":
                check(session_id is None, "duplicate session.created")
                session_id = session_fields.get("id")
                check(
                    isinstance(session_id, str) and bool(session_id),
                    "missing session ID",
                )
            elif event_type == "session.updated":
                check(
                    session_id is not None and session_fields.get("id") == session_id,
                    "session ID changed",
                )
                check(
                    updates_acknowledged < updates_sent,
                    "session.updated without a pending session.update",
                )
                updates_acknowledged += 1
                if updates_acknowledged == 1:
                    turn = session_fields.get("turn_detection")
                    echoed_type = turn.get("type") if isinstance(turn, dict) else None
                    check(
                        echoed_type == requested_type,
                        f"turn_detection.type {echoed_type!r} does not echo the "
                        f"requested {requested_type!r}",
                    )
                    modalities = session_fields.get("modalities")
                    check(
                        isinstance(modalities, list) and "audio" in modalities,
                        "session.updated does not grant audio output",
                    )
                elif probe_sent_s is not None:
                    probe_ack_s = event_time_s - probe_sent_s
                else:
                    pass
            elif event_type == "input_audio_buffer.speech_started":
                interrupted.update(
                    response_id
                    for response_id, response_state in response_states.items()
                    if response_state.terminal is None
                )
            elif event_type in LEGACY_OBSERVATIONS:
                pass
            elif event_type == "output_audio_buffer.cleared":
                check(
                    event.response_id in response_states,
                    "output_audio_buffer.cleared: unknown response",
                )
            elif event_type == "input_audio_buffer.cleared":
                violations.append("server discarded buffered input")
            elif event_type == "response.created":
                check(updates_acknowledged > 0, "response before session.updated")
                if event.response is None:
                    violations.append("response.created missing response")
                else:
                    check(
                        event.response.id not in response_states,
                        "duplicate response.created",
                    )
                    check(
                        event.response.status == "in_progress",
                        "invalid created response status",
                    )
                    response_states[event.response.id] = ResponseState()
            elif event_type.startswith("response."):
                response_id = event.response.id if event.response else event.response_id
                response_state = response_states.get(response_id)
                check(response_state is not None, f"{event_type}: unknown response")
                if response_state is not None:
                    check(
                        response_state.terminal is None,
                        f"{event_type}: output after response terminal",
                    )
                    if event_type == "response.done":
                        status = event.response.status if event.response else None
                        details = (
                            event.response.status_details if event.response else None
                        )
                        reason = details.reason if details is not None else None
                        if (status, reason) == LEGACY_INTERRUPTED_TERMINAL:
                            check(
                                response_id in interrupted,
                                f"response {response_id} cancelled by turn detection "
                                "without a preceding speech_started",
                            )
                        else:
                            check(
                                (status, reason) in LEGACY_TERMINALS,
                                f"unsupported response terminal {status}/{reason}",
                            )
                        check(
                            status != "completed"
                            or not response_state.audio_seen
                            or response_state.audio_done,
                            "missing audio.done",
                        )
                        response_state.terminal, response_state.reason = status, reason
                    elif event_type == "response.audio.delta":
                        check(not response_state.audio_done, "audio after audio.done")
                        check(event.item_id is not None, "audio missing item ID")
                        check(
                            response_state.item_id is None
                            or response_state.item_id == event.item_id,
                            "audio item ID changed",
                        )
                        check(
                            event.content_index == 1 and event.output_index == 0,
                            "invalid audio indexes",
                        )
                        output_byte_count += len(pcm_bytes(event.delta))
                        audio_times.append(event_time_s)
                        response_state.audio_seen = True
                        response_state.item_id = event.item_id
                    elif event_type == "response.audio.done":
                        check(not response_state.audio_done, "duplicate audio.done")
                        response_state.audio_done = True
                    elif event_type == "response.text.delta":
                        check(event.delta is not None, "text delta missing content")
                    elif event_type != "response.text.done":
                        violations.append(f"unsupported server event: {event_type}")
                    else:
                        pass
                else:
                    pass
            elif event_type == "error":
                error = event.error or {}
                violations.append(
                    f"server error: code={error.get('code')} "
                    f"fatal={event.sglang.fatal}: {error.get('message')}"
                )
            else:
                violations.append(f"unsupported server event: {event_type}")
        except (ValidationError, ValueError, binascii.Error) as exc:
            violations.append(f"record {index}: {exc}")

    check(
        counts["receive", "session.created"] == 1,
        "expected exactly one receive session.created",
    )
    for direction, event_type in (
        ("send", "session.update"),
        ("receive", "session.updated"),
    ):
        check(
            counts[direction, event_type] == 2,
            f"expected exactly two {direction} {event_type}",
        )
    check(bool(input_times), "missing audio input")
    coverage = {
        "input_output_overlap": bool(audio_times)
        and any(
            audio_times[0] < event_time_s < audio_times[-1]
            for event_time_s in input_times
        )
    }
    metrics = {
        "input_audio_s": input_byte_count / INPUT_BYTES_PER_S,
        "output_audio_s": output_byte_count / (profile_contract.output_sample_rate * 2),
        "admission_denials": admissions,
        "first_audio_packet_s": (
            audio_times[0] - input_times[0] if audio_times and input_times else None
        ),
        "audio_packet_gap_max_s": max(
            (right - left for left, right in zip(audio_times, audio_times[1:])),
            default=None,
        ),
        "input_output_overlap": coverage["input_output_overlap"],
        "probe_ack_s": probe_ack_s,
        "responses_created": len(response_states),
        "responses_overlapped_by_speech": len(interrupted),
        "responses_open_at_close": sum(
            response_state.terminal is None
            for response_state in response_states.values()
        ),
    }
    return {
        "status": "fail" if violations else "pass",
        "violations": violations,
        "coverage": coverage,
        "metrics": metrics,
    }
