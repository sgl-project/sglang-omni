# SPDX-License-Identifier: Apache-2.0
"""One realtime session's lifecycle, the capacity probe and the server snapshot fetch."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import JsonValue
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus

from benchmarks.duplex.client import (
    ADMISSION_DENIED_STATUS,
    MAX_MESSAGE_BYTES,
    PACKET_BYTES,
    PACKET_MS,
)
from benchmarks.duplex.lifecycle_stress.host_metrics import (
    read_open_fd_count,
    read_rss_megabytes,
)
from benchmarks.duplex.lifecycle_stress.records import (
    CapacityProbe,
    CloseStyle,
    DropPoint,
    DropStyle,
    IdlePoint,
    Outcome,
    ServerSnapshot,
    SessionOutcome,
)
from benchmarks.duplex.oracle_models import is_admission_rejection

SESSION_TIMEOUT_S = 30.0
MS_PER_SECOND = 1000
NORMAL_CLOSE_CODE = 1000
POLICY_CLOSE_CODE = 1008
DEBUG_ROUTE_PATH = "/debug/lifecycle"
CAPABILITIES_ROUTE_PATH = "/v1/realtime/capabilities"
# note (Yi Zhao): the server closes an idle session with the name of the limit that expired, and publishes that limit under the same name plus _s.
EXPIRY_LIMITS: dict[IdlePoint, str] = {
    "created": "session_update_timeout",
    "open": "idle_timeout",
}
SNAPSHOT_TIMEOUT_S = 5.0
# One 80 ms packet of 16 kHz PCM16 silence, the same packet the duplex client sends.
SILENCE_PACKET = base64.b64encode(bytes(PACKET_BYTES)).decode("ascii")


@dataclass(kw_only=True)
class Hold:
    """Keeps an admitted session open until the cycle that started it lets it close."""

    release: asyncio.Event
    is_admitted: bool = False


@dataclass(kw_only=True)
class SessionPlan:
    appends: int
    drop_point: DropPoint = "none"
    drop_style: DropStyle = "abort"
    close_style: CloseStyle = "event"
    idle_point: IdlePoint = "none"
    should_cancel_reply: bool = False
    hold: Hold | None = None
    timeout_s: float = SESSION_TIMEOUT_S
    append_interval_s: float = PACKET_MS / MS_PER_SECOND


class ServerErrorEvent(RuntimeError):
    """The server answered with an error event while the client waited for another one."""

    def __init__(self, event: dict[str, JsonValue]) -> None:
        super().__init__(json.dumps(event.get("error")))
        self.is_admission_rejection = is_admission_rejection(event)


def client_event(
    event_type: str, event_id: str, fields: dict[str, JsonValue] | None = None
) -> str:
    return json.dumps({"event_id": event_id, "type": event_type, **(fields or {})})


async def receive_until(
    connection: ClientConnection, event_type: str
) -> list[dict[str, JsonValue]]:
    """Read server events up to the first of the given type, which is the last one returned; an error event ends the wait."""
    events: list[dict[str, JsonValue]] = []
    while True:
        event = json.loads(await connection.recv())
        events.append(event)
        if event["type"] == event_type:
            return events
        elif event["type"] == "error":
            raise ServerErrorEvent(event)
        else:
            pass


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


async def leave_early(connection: ClientConnection, drop_style: DropStyle) -> None:
    """Leave without reading what the server still has to say: a TCP reset or a close frame."""
    if drop_style == "abort":
        connection.transport.abort()
    else:
        await connection.close()
    await connection.wait_closed()


async def drive_session(
    connection: ClientConnection, plan: SessionPlan, connect_started_s: float
) -> SessionOutcome:
    """Run the plan on an open socket; the caller has already measured the connect."""
    started_s = time.perf_counter()
    await receive_until(connection, "session.created")
    connect_ms = (time.perf_counter() - started_s) * MS_PER_SECOND
    outcome: Outcome
    admit_ms: float | None = None
    expire_ms: float | None = None
    cancel_ms: float | None = None
    cancel_status: str | None = None
    late_outputs: int | None = None
    # note (Yi Zhao): the server's idle clock restarts no earlier than the client's last send, so a wait measured from there that is shorter than the limit is an early close.
    last_sent_s = connect_started_s
    if plan.idle_point == "created":
        outcome = await await_expiry(connection, plan.idle_point)
        expire_ms = (time.perf_counter() - last_sent_s) * MS_PER_SECOND
    else:
        last_sent_s = time.perf_counter()
        await connection.send(
            client_event("session.update", "update_0", {"session": {}})
        )
        if plan.drop_point == "update":
            await leave_early(connection, plan.drop_style)
            outcome = "dropped"
        else:
            await receive_until(connection, "session.updated")
            admit_ms = (time.perf_counter() - last_sent_s) * MS_PER_SECOND
            cancelled_response_id: str | None = None
            first_append_s = time.perf_counter()
            for append_index in range(plan.appends):
                due_s = first_append_s + append_index * plan.append_interval_s
                await asyncio.sleep(max(0.0, due_s - time.perf_counter()))
                last_sent_s = time.perf_counter()
                await connection.send(
                    client_event(
                        "input_audio_buffer.append",
                        f"append_{append_index}",
                        {
                            "audio": SILENCE_PACKET,
                            "sglang": {
                                "seq": append_index,
                                "t_start_ms": append_index * PACKET_MS,
                            },
                        },
                    )
                )
                if plan.should_cancel_reply and append_index == 0:
                    # note (Yi Zhao): cancelling on the first response.created read live keeps the reply in progress; one read from a backlog may already be over.
                    created = await receive_until(connection, "response.created")
                    cancelled_response_id = created[-1]["response"]["id"]
                    cancel_sent_s = time.perf_counter()
                    await connection.send(client_event("response.cancel", "cancel_0"))
                    while cancel_status is None:
                        done = await receive_until(connection, "response.done")
                        if done[-1]["response"]["id"] == cancelled_response_id:
                            cancel_status = done[-1]["response"]["status"]
                        else:
                            pass
                    cancel_ms = (time.perf_counter() - cancel_sent_s) * MS_PER_SECOND
                else:
                    pass
            if plan.hold is not None:
                plan.hold.is_admitted = True
                await plan.hold.release.wait()
            else:
                pass
            if plan.idle_point == "open":
                outcome = await await_expiry(connection, plan.idle_point)
                expire_ms = (time.perf_counter() - last_sent_s) * MS_PER_SECOND
            elif plan.drop_point == "append":
                await leave_early(connection, plan.drop_style)
                outcome = "dropped"
            elif plan.close_style == "event":
                await connection.send(client_event("session.close", "close_0"))
                closing_events = await receive_until(connection, "session.closed")
                if cancelled_response_id is not None:
                    late_outputs = sum(
                        event.get("response_id") == cancelled_response_id
                        for event in closing_events
                    )
                else:
                    pass
                await connection.wait_closed()
                outcome = (
                    "completed"
                    if connection.close_code == NORMAL_CLOSE_CODE
                    else "errored"
                )
            else:
                await connection.close()
                outcome = (
                    "completed"
                    if connection.close_code == NORMAL_CLOSE_CODE
                    else "errored"
                )
    return SessionOutcome(
        outcome=outcome,
        close_code=connection.close_code,
        close_reason=connection.close_reason,
        connect_ms=connect_ms,
        admit_ms=admit_ms,
        expire_ms=expire_ms,
        cancel_ms=cancel_ms,
        cancel_status=cancel_status,
        late_outputs=late_outputs,
        wall_ms=(time.perf_counter() - started_s) * MS_PER_SECOND,
        ended_s=time.perf_counter(),
    )


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
    except (ConnectionClosed, InvalidHandshake, OSError, TimeoutError) as exc:
        outcome = "errored"
        error = f"{type(exc).__name__}: {exc}"
    return SessionOutcome(
        outcome=outcome,
        error=error,
        wall_ms=(time.perf_counter() - started_s) * MS_PER_SECOND,
        ended_s=time.perf_counter(),
    )


async def probe_capacity(
    url: str, max_connections: int, timeout_s: float
) -> CapacityProbe:
    """Open the whole connection cap at once; a denial means a slot is still held."""
    started_s = time.perf_counter()
    outcomes = await asyncio.gather(
        *(
            run_lifecycle_session(
                url, SessionPlan(appends=0, close_style="frame", timeout_s=timeout_s)
            )
            for _ in range(max_connections)
        )
    )
    return CapacityProbe(
        admitted=sum(outcome.outcome == "completed" for outcome in outcomes),
        denied=sum(outcome.outcome == "denied" for outcome in outcomes),
        errored=sum(outcome.outcome == "errored" for outcome in outcomes),
        wall_ms=(time.perf_counter() - started_s) * MS_PER_SECOND,
    )


def http_url_of(realtime_url: str, route_path: str) -> str:
    """An HTTP route on the same host as the realtime route."""
    parts = urlsplit(realtime_url)
    scheme = "https" if parts.scheme == "wss" else "http"
    return urlunsplit((scheme, parts.netloc, route_path, "", ""))


@dataclass(kw_only=True)
class SnapshotSource:
    """Where server state comes from: the mock's debug route, a local pid, or nothing."""

    debug_url: str | None
    server_pid: int | None
    has_debug_route: bool = True

    async def fetch(self) -> ServerSnapshot | None:
        if self.debug_url is not None and self.has_debug_route:
            async with httpx.AsyncClient(timeout=SNAPSHOT_TIMEOUT_S) as http:
                response = await http.get(self.debug_url)
            if response.status_code == httpx.codes.NOT_FOUND:
                self.has_debug_route = False
            else:
                response.raise_for_status()
                snapshot = ServerSnapshot.model_validate_json(response.text)
                return snapshot.model_copy(update={"taken_s": time.perf_counter()})
        else:
            pass
        if self.server_pid is not None:
            return ServerSnapshot(
                taken_s=time.perf_counter(),
                rss_megabytes=read_rss_megabytes(self.server_pid),
                open_fd_count=read_open_fd_count(self.server_pid),
            )
        else:
            return None
