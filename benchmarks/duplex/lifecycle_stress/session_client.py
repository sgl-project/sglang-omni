# SPDX-License-Identifier: Apache-2.0
"""One short realtime session that can leave early, cancel or go idle, which the duplex recorder cannot."""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, JsonValue, ValidationError
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus

from benchmarks.duplex.client import (
    ADMISSION_DENIED_STATUS,
    MAX_MESSAGE_BYTES,
    PACKET_BYTES,
    PACKET_MS,
    SAMPLE_RATE,
    TurnDetection,
    run_session,
)
from benchmarks.duplex.lifecycle_stress.records import (
    MS_PER_SECOND,
    CapacityProbe,
    IdlePoint,
    LeavePoint,
    LeaveStyle,
    Outcome,
    Protocol,
    SessionOutcome,
)
from benchmarks.duplex.oracle_models import is_admission_rejection

NORMAL_CLOSE_CODE = 1000
POLICY_CLOSE_CODE = 1008
CANCEL_REFUSAL_CODE = "not_supported"
# note (Yi Zhao): the recorder's profile for a server that takes turns by its own voice detection, and how such a server marks a reply the user talked over.
TURN_BASED_PROFILE = "turn-based"
INTERRUPTED_REASON = "interrupted"
# note (Yi Zhao): the server closes an idle session with the name of the limit that expired, and publishes that limit under the same name plus _s.
EXPIRY_LIMITS: dict[IdlePoint, str] = {
    "created": "session_update_timeout",
    "open": "idle_timeout",
}
# note (Yi Zhao): the native value is what the duplex recorder sends, so both clients open the same kind of session.
SESSION_CONFIGURATION: dict[Protocol, dict[str, JsonValue]] = {
    "native": {"output_modalities": ["audio"]},
    "legacy": {},
}


@dataclass(kw_only=True)
class SessionPlan:
    protocol: Protocol
    pcm: bytes
    packets: int
    leave_point: LeavePoint
    leave_style: LeaveStyle
    idle_point: IdlePoint
    should_cancel_reply: bool
    # note (Yi Zhao): what the deployment publishes; one that does not cancel has to refuse response.cancel and keep the session going.
    supports_cancel: bool
    timeout_s: float


class ServerEvent(BaseModel):
    """The fields of a server event this client reads; its response is validated where it is read."""

    type: str
    response_id: str | None = None
    response: JsonValue = None
    error: JsonValue = None
    sglang: JsonValue = None


class EventResponse(BaseModel):
    """The response object that response.created and response.done carry."""

    id: str
    status: str


class EventError(BaseModel):
    """The error object of an error event."""

    code: str


class DoneDetails(BaseModel):
    reason: str | None = None


class DoneResponse(EventResponse):
    """The response of a response.done read from a trace, with why it ended."""

    status_details: DoneDetails = DoneDetails()


class TraceRecord(BaseModel):
    """One line of the duplex recorder's trace."""

    direction: str
    event: dict[str, JsonValue]


class ServerErrorEvent(RuntimeError):
    """The server answered with an error event while the client waited for another one."""

    def __init__(self, event: ServerEvent) -> None:
        super().__init__(json.dumps(event.error))
        self.is_admission_rejection: bool = is_admission_rejection(event.model_dump())


def client_event(event_type: str, fields: dict[str, JsonValue]) -> str:
    return json.dumps({"event_id": uuid.uuid4().hex, "type": event_type, **fields})


async def receive_until(
    connection: ClientConnection, event_type: str
) -> list[ServerEvent]:
    """Read server events up to the first of the given type, which is the last one returned; an error event ends the wait."""
    events: list[ServerEvent] = []
    while True:
        event = ServerEvent.model_validate_json(await connection.recv())
        events.append(event)
        if event.type == event_type:
            return events
        elif event.type == "error":
            raise ServerErrorEvent(event)
        else:
            pass


async def receive_response(
    connection: ClientConnection, event_type: str
) -> EventResponse:
    """The response object of the next server event of the given type."""
    events = await receive_until(connection, event_type)
    return EventResponse.model_validate(events[-1].response)


async def await_expiry(connection: ClientConnection, idle_point: IdlePoint) -> Outcome:
    """Send nothing more and read until the server closes; expired only if it names the timeout of this idle point."""
    try:
        while True:
            await connection.recv()
    except ConnectionClosed:
        pass
    if (
        connection.close_code == POLICY_CLOSE_CODE
        and connection.close_reason == EXPIRY_LIMITS[idle_point]
    ):
        return "expired"
    else:
        return "errored"


async def leave_early(connection: ClientConnection, leave_style: LeaveStyle) -> None:
    """Leave without reading what the server still has to say: a TCP reset or a close frame."""
    if leave_style == "reset":
        connection.transport.abort()
    else:
        await connection.close()
    await connection.wait_closed()


async def send_packets(
    connection: ClientConnection,
    plan: SessionPlan,
    reply_task: asyncio.Task[None] | None,
    last_sent_s: float,
) -> float:
    """Send the plan's packets at their pace and return when the last one went out; a plan that leaves mid-reply stops at the first packet after the reply has started."""
    first_packet_s = time.perf_counter()
    for packet_index in range(plan.packets):
        due_s = first_packet_s + packet_index * PACKET_MS / MS_PER_SECOND
        await asyncio.sleep(max(0.0, due_s - time.perf_counter()))
        if plan.leave_point == "reply" and reply_task is not None and reply_task.done():
            break
        else:
            pass
        byte_offset = packet_index * PACKET_BYTES
        last_sent_s = time.perf_counter()
        await connection.send(
            client_event(
                "input_audio_buffer.append",
                {
                    "audio": base64.b64encode(
                        plan.pcm[byte_offset : byte_offset + PACKET_BYTES]
                    ).decode("ascii"),
                    "sglang": {
                        "seq": packet_index,
                        "t_start_ms": byte_offset / 2 / SAMPLE_RATE * MS_PER_SECOND,
                    },
                },
            )
        )
    return last_sent_s


async def drive_session(
    connection: ClientConnection, plan: SessionPlan, connect_started_s: float
) -> SessionOutcome:
    """Run the plan on an open socket up to the point where the client or the server ends the session."""
    # note (Yi Zhao): the server's idle clock restarts no earlier than the client's last send, so a wait measured from there that is shorter than the limit is an early close.
    last_sent_s = connect_started_s
    expire_ms: float | None = None
    cancel_ms: float | None = None
    cancel_status: str | None = None
    late_outputs: int | None = None

    def ended(outcome: Outcome) -> SessionOutcome:
        return SessionOutcome(
            outcome=outcome,
            ended_s=time.perf_counter(),
            close_code=connection.close_code,
            close_reason=connection.close_reason,
            expire_ms=expire_ms,
            cancel_ms=cancel_ms,
            cancel_status=cancel_status,
            late_outputs=late_outputs,
        )

    await receive_until(connection, "session.created")
    if plan.idle_point == "created":
        outcome = await await_expiry(connection, plan.idle_point)
        expire_ms = (time.perf_counter() - last_sent_s) * MS_PER_SECOND
        return ended(outcome)
    else:
        pass
    last_sent_s = time.perf_counter()
    await connection.send(
        client_event(
            "session.update", {"session": SESSION_CONFIGURATION[plan.protocol]}
        )
    )
    if plan.leave_point == "update":
        await leave_early(connection, plan.leave_style)
        return ended("dropped")
    else:
        await receive_until(connection, "session.updated")

    cancelled_response_id: str | None = None

    async def act_on_reply() -> None:
        """Wait for the first reply and cancel it if the plan says so; the packets keep going meanwhile."""
        nonlocal cancel_ms, cancel_status, cancelled_response_id
        created = await receive_response(connection, "response.created")
        if not plan.should_cancel_reply:
            return
        else:
            pass
        cancel_sent_s = time.perf_counter()
        await connection.send(client_event("response.cancel", {}))
        if plan.leave_point == "reply":
            return
        elif plan.supports_cancel:
            cancelled_response_id = created.id
            while cancel_status is None:
                done = await receive_response(connection, "response.done")
                if done.id == cancelled_response_id:
                    cancel_status = done.status
                else:
                    pass
        else:
            refusal = (await receive_until(connection, "error"))[-1]
            cancel_status = EventError.model_validate(refusal.error).code
            if cancel_status != CANCEL_REFUSAL_CODE:
                raise ServerErrorEvent(refusal)
            else:
                pass
        cancel_ms = (time.perf_counter() - cancel_sent_s) * MS_PER_SECOND

    # note (Yi Zhao): a server that takes turns answers only once it has heard the turn end, which takes the packets after the speech, so the client must not stop sending while it waits for the reply. Reading the reply as it arrives also keeps it in progress when the client acts on it.
    reply_task = (
        asyncio.create_task(act_on_reply())
        if plan.should_cancel_reply or plan.leave_point == "reply"
        else None
    )
    try:
        last_sent_s = await send_packets(connection, plan, reply_task, last_sent_s)
        if reply_task is None:
            pass
        else:
            await reply_task
    finally:
        if reply_task is None:
            pass
        else:
            reply_task.cancel()
    if plan.leave_point in ("reply", "append"):
        await leave_early(connection, plan.leave_style)
        return ended("dropped")
    elif plan.idle_point == "open":
        outcome = await await_expiry(connection, plan.idle_point)
        expire_ms = (time.perf_counter() - last_sent_s) * MS_PER_SECOND
        return ended(outcome)
    elif plan.leave_style == "goodbye":
        # note (Yi Zhao): the recorder's goodbye: end the input, wait until the server has answered all of it, then close.
        await connection.send(client_event("sglang.input_audio.end", {}))
        closing_events = await receive_until(connection, "sglang.input_audio.drained")
        await connection.send(client_event("session.close", {}))
        closing_events += await receive_until(connection, "session.closed")
        if cancelled_response_id is not None:
            late_outputs = sum(
                event.response_id == cancelled_response_id for event in closing_events
            )
        else:
            pass
        await connection.wait_closed()
    else:
        await connection.close()
    if connection.close_code == NORMAL_CLOSE_CODE:
        return ended("completed")
    else:
        return ended("errored")


async def run_lifecycle_session(url: str, plan: SessionPlan) -> SessionOutcome:
    """Connect, run the plan and classify how the session ended; never raises."""
    started_s = time.perf_counter()
    outcome: Outcome
    try:
        async with asyncio.timeout(plan.timeout_s):
            # note (Yi Zhao): a session that ends in an error still holds its connection slot until the socket closes.
            async with connect(
                url,
                max_size=MAX_MESSAGE_BYTES,
                compression=None,
                ping_interval=None,
                open_timeout=plan.timeout_s,
            ) as connection:
                return await drive_session(connection, plan, started_s)
    except InvalidStatus as exc:
        if exc.response.status_code == ADMISSION_DENIED_STATUS:
            outcome = "denied"
        else:
            outcome = "errored"
        error = f"{type(exc).__name__}: {exc}"
    except ServerErrorEvent as exc:
        # note (Yi Zhao): a server whose model is full admits the socket and rejects session.update; that is a denial, like a 503.
        if exc.is_admission_rejection:
            outcome = "denied"
        else:
            outcome = "errored"
        error = f"{type(exc).__name__}: {exc}"
    except (
        ConnectionClosed,
        InvalidHandshake,
        OSError,
        TimeoutError,
        ValidationError,
    ) as exc:
        outcome = "errored"
        error = f"{type(exc).__name__}: {exc}"
    return SessionOutcome(outcome=outcome, ended_s=time.perf_counter(), error=error)


async def run_recorded_session(
    url: str,
    pcm: bytes,
    trace_path: Path,
    timeout_s: float,
    turn_detection: TurnDetection | None,
) -> SessionOutcome:
    """A whole conversation by the duplex recorder; its trace is the raw record and tells how the session ended and which replies the user talked over."""
    trace_path.parent.mkdir(parents=True)
    if turn_detection is None:
        await run_session(url, pcm, trace_path=trace_path, timeout_s=timeout_s)
    else:
        await run_session(
            url,
            pcm,
            trace_path=trace_path,
            timeout_s=timeout_s,
            profile=TURN_BASED_PROFILE,
            turn_detection=turn_detection,
        )
    ended_s = time.perf_counter()
    records = [
        TraceRecord.model_validate_json(line)
        for line in trace_path.read_text().splitlines()
    ]
    errors = [record.event for record in records if record.direction == "error"]
    received_types = [
        record.event["type"] for record in records if record.direction == "receive"
    ]
    if errors:
        return SessionOutcome(
            outcome="errored", ended_s=ended_s, error=json.dumps(errors[0])
        )
    elif "session.closed" in received_types:
        interrupted_ids: set[str] = set()
        late_outputs = 0
        for record in records:
            if record.direction != "receive":
                pass
            elif record.event["type"] == "response.done":
                done = DoneResponse.model_validate(record.event["response"])
                if (
                    done.status == "cancelled"
                    and done.status_details.reason == INTERRUPTED_REASON
                ):
                    interrupted_ids.add(done.id)
                else:
                    pass
            elif record.event.get("response_id") in interrupted_ids:
                late_outputs += 1
            else:
                pass
        if turn_detection is None:
            return SessionOutcome(outcome="completed", ended_s=ended_s)
        else:
            return SessionOutcome(
                outcome="completed",
                ended_s=ended_s,
                interrupted_replies=len(interrupted_ids),
                late_outputs=late_outputs,
            )
    elif any(record.direction == "admission" for record in records):
        # note (Yi Zhao): the recorder retries a denied connection a few times before it gives up without an error line.
        return SessionOutcome(outcome="denied", ended_s=ended_s)
    else:
        return SessionOutcome(
            outcome="errored", ended_s=ended_s, error="trace has no session.closed"
        )


async def take_slot(
    url: str, timeout_s: float
) -> ClientConnection | Literal["denied", "errored"]:
    """Open one session up to session.updated and keep it open for the caller to close; a server may refuse at the connect or at session.update, and either is a denial."""
    try:
        async with asyncio.timeout(timeout_s):
            connection = await connect(
                url, max_size=MAX_MESSAGE_BYTES, compression=None, ping_interval=None
            )
    except InvalidStatus as exc:
        if exc.response.status_code == ADMISSION_DENIED_STATUS:
            return "denied"
        else:
            return "errored"
    except (InvalidHandshake, OSError, TimeoutError):
        return "errored"
    try:
        async with asyncio.timeout(timeout_s):
            await receive_until(connection, "session.created")
            await connection.send(
                client_event(
                    "session.update", {"session": SESSION_CONFIGURATION["native"]}
                )
            )
            await receive_until(connection, "session.updated")
    except ServerErrorEvent as exc:
        await connection.close()
        return "denied" if exc.is_admission_rejection else "errored"
    except (ConnectionClosed, TimeoutError, ValidationError):
        await connection.close()
        return "errored"
    return connection


async def probe_capacity(url: str, slots: int, timeout_s: float) -> CapacityProbe:
    """Hold the given number of admitted sessions open together, then release them; a denial means a slot is still taken, whether the server counts slots at the connect or at session.update."""
    results = await asyncio.gather(*(take_slot(url, timeout_s) for _ in range(slots)))
    connections = [result for result in results if isinstance(result, ClientConnection)]
    await asyncio.gather(*(connection.close() for connection in connections))
    return CapacityProbe(
        admitted=len(connections),
        denied=results.count("denied"),
        errored=results.count("errored"),
    )
