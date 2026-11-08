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
    Outcome,
    ServerSnapshot,
    SessionOutcome,
)

SESSION_TIMEOUT_S = 30.0
MS_PER_SECOND = 1000
NORMAL_CLOSE_CODE = 1000
DEBUG_ROUTE_PATH = "/debug/lifecycle"
SNAPSHOT_TIMEOUT_S = 5.0
# One 80 ms packet of 16 kHz PCM16 silence, the same packet the duplex client sends.
SILENCE_PACKET = base64.b64encode(bytes(PACKET_BYTES)).decode("ascii")


@dataclass(kw_only=True)
class SessionPlan:
    appends: int
    drop_point: DropPoint = "none"
    drop_style: DropStyle = "abort"
    close_style: CloseStyle = "event"
    timeout_s: float = SESSION_TIMEOUT_S
    append_interval_s: float = PACKET_MS / MS_PER_SECOND


def client_event(
    event_type: str, event_id: str, fields: dict[str, JsonValue] | None = None
) -> str:
    return json.dumps({"event_id": event_id, "type": event_type, **(fields or {})})


async def receive_until(connection: ClientConnection, event_type: str) -> None:
    """Read server events until one of the given type arrives."""
    while True:
        event = json.loads(await connection.recv())
        if event["type"] == event_type:
            return
        else:
            pass


async def leave_early(connection: ClientConnection, drop_style: DropStyle) -> None:
    """Leave without reading what the server still has to say: a TCP reset or a close frame."""
    if drop_style == "abort":
        connection.transport.abort()
    else:
        await connection.close()
    await connection.wait_closed()


async def drive_session(
    connection: ClientConnection, plan: SessionPlan
) -> SessionOutcome:
    """Run the plan on an open socket; the caller has already measured the connect."""
    started_s = time.perf_counter()
    await receive_until(connection, "session.created")
    connect_ms = (time.perf_counter() - started_s) * MS_PER_SECOND
    update_sent_s = time.perf_counter()
    await connection.send(client_event("session.update", "update_0", {"session": {}}))
    if plan.drop_point == "update":
        await leave_early(connection, plan.drop_style)
        return SessionOutcome(
            outcome="dropped",
            close_code=connection.close_code,
            close_reason=connection.close_reason,
            connect_ms=connect_ms,
            wall_ms=(time.perf_counter() - started_s) * MS_PER_SECOND,
            ended_s=time.perf_counter(),
        )
    else:
        pass
    await receive_until(connection, "session.updated")
    admit_ms = (time.perf_counter() - update_sent_s) * MS_PER_SECOND
    first_append_s = time.perf_counter()
    for append_index in range(plan.appends):
        due_s = first_append_s + append_index * plan.append_interval_s
        await asyncio.sleep(max(0.0, due_s - time.perf_counter()))
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
    outcome: Outcome
    if plan.drop_point == "append":
        await leave_early(connection, plan.drop_style)
        outcome = "dropped"
    elif plan.close_style == "event":
        await connection.send(client_event("session.close", "close_0"))
        await receive_until(connection, "session.closed")
        await connection.wait_closed()
        outcome = (
            "completed" if connection.close_code == NORMAL_CLOSE_CODE else "errored"
        )
    else:
        await connection.close()
        outcome = (
            "completed" if connection.close_code == NORMAL_CLOSE_CODE else "errored"
        )
    return SessionOutcome(
        outcome=outcome,
        close_code=connection.close_code,
        close_reason=connection.close_reason,
        connect_ms=connect_ms,
        admit_ms=admit_ms,
        wall_ms=(time.perf_counter() - started_s) * MS_PER_SECOND,
        ended_s=time.perf_counter(),
    )


async def run_lifecycle_session(url: str, plan: SessionPlan) -> SessionOutcome:
    """Connect, run the plan and classify how the session ended; never raises."""
    started_s = time.perf_counter()
    try:
        async with asyncio.timeout(plan.timeout_s):
            connection = await connect(
                url,
                max_size=MAX_MESSAGE_BYTES,
                compression=None,
                ping_interval=None,
                open_timeout=plan.timeout_s,
            )
            return await drive_session(connection, plan)
    except InvalidStatus as exc:
        outcome: Outcome
        if exc.response.status_code == ADMISSION_DENIED_STATUS:
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


def debug_url_of(realtime_url: str) -> str:
    """The mock's debug route on the same host as its realtime route."""
    parts = urlsplit(realtime_url)
    scheme = "https" if parts.scheme == "wss" else "http"
    return urlunsplit((scheme, parts.netloc, DEBUG_ROUTE_PATH, "", ""))


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
