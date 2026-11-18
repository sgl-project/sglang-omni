# SPDX-License-Identifier: Apache-2.0
"""HTTP routes served by the stage process that owns their runtime.

A stage whose config lists ``http_routes`` serves an ASGI app from its own
process, on a Unix socket in the pipeline's IPC runtime directory. The Omni
HTTP server relays every request under those path prefixes to that socket as
ASGI messages. The routes therefore start with their stage, and the HTTP
process never imports the runtime they need.

The stage side is :class:`StageHttpServer`, the HTTP side is
:class:`StageRoutes`. Both use only the standard library and Starlette, and the
relay is model agnostic: request and response bodies stream in the chunks the
servers produce, a client disconnect reaches the stage app, and a stage that
stops serving answers 503 on its routes.

A connection carries one request at a time and returns to a pool of idle
connections after it, so a request costs no connection setup. An idle
connection closes after one to two keep-alive periods of the Omni server,
which also ends its task on the stage side. Every frame is a
4 byte big endian length and a pickled ASGI message. The HTTP side first sends
the request scope, then the client's messages. The stage side answers with the
app's messages and ends with an end frame, or an error frame when its app
raised. The HTTP side then sends a done frame, after which the stage reads the
next request on the same connection. Both peers are processes of one pipeline,
and the socket sits in the pipeline's private runtime directory.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import os
import pickle
import socket
import struct
import threading
from collections.abc import Iterable, MutableMapping
from typing import Any

from starlette.convertors import PathConvertor
from starlette.datastructures import URLPath
from starlette.responses import JSONResponse
from starlette.routing import BaseRoute, Match, Mount, NoMatchFound, compile_path
from starlette.types import ASGIApp, Receive, Scope, Send

from sglang_omni.config.schema import normalize_route_prefixes

logger = logging.getLogger(__name__)

FRAME_HEADER = struct.Struct("!I")
END_FRAME = {"type": "relay.end"}
DONE_FRAME = {"type": "relay.done"}
ERROR_FRAME_TYPE = "relay.error"
# Scope keys a relayed request keeps. The rest belong to the receiving server
# (app, router, endpoint, path params) or cannot cross a process (state).
RELAYED_SCOPE_KEYS = (
    "type",
    "asgi",
    "http_version",
    "method",
    "scheme",
    "path",
    "raw_path",
    "query_string",
    "root_path",
    "headers",
    "client",
    "server",
    "subprotocols",
)
DISCONNECT = {"http": "http.disconnect", "websocket": "websocket.disconnect"}
# The port the Omni HTTP server serves on, handed to the stage processes it
# spawns, since it binds that port only after they started.
SERVER_PORT_ENV = "SGLANG_OMNI_SERVER_PORT"
# Seconds the Omni HTTP server keeps an idle client connection open, which the
# relay applies to its idle stage connections too.
KEEP_ALIVE_S = 120
# note (Yuhao Chen): Some Python builds expose SOMAXCONN as 128 despite a larger kernel limit.
LISTEN_BACKLOG = max(socket.SOMAXCONN, 4096)


def export_server_port(port: int) -> None:
    """Hand the HTTP server's port to the stage processes it spawns."""
    os.environ[SERVER_PORT_ENV] = str(port)


def server_port() -> int | None:
    """The Omni HTTP server's port, in a stage process it spawned."""
    value = os.environ.get(SERVER_PORT_ENV, "")
    return int(value) if value.isdigit() else None


class StageRoutesError(RuntimeError):
    """A stage cannot serve the routes its config declares."""


def covers(prefix: str, path: str) -> bool:
    return path == prefix or path.startswith(prefix + "/")


def template_reaches(template: str, prefix: str) -> bool:
    """Whether a Starlette path template matches the prefix or a path below it.

    A path is at or below the prefix when its leading segments equal the
    prefix's, so the template reaches the prefix when each of its leading
    segments can match the prefix segment in its place. A path parameter
    crosses segments and can match any rest.
    """
    template_segments = template.split("/")[1:]
    for index, segment in enumerate(prefix.split("/")[1:]):
        if index >= len(template_segments):
            return False
        else:
            pass
        template_segment = template_segments[index]
        regex, _, convertors = compile_path("/" + template_segment)
        if any(isinstance(c, PathConvertor) for c in convertors.values()):
            # Text before the first parameter must open the segment.
            return segment.startswith(template_segment.split("{", 1)[0])
        elif regex.match("/" + segment) is None:
            return False
        else:
            pass
    return True


def check_route_ownership(routes: Iterable[BaseRoute], stage: StageRoutes) -> None:
    """Refuse a stage prefix that reaches a route the Omni server serves itself.

    Starlette serves a path through the first route that fully matches it, and
    only then through the first one that matched the path with another method,
    so a stage prefix over an Omni route would take that route's other methods.
    A route reaches a prefix when its path template matches the prefix or any
    path below it. A route without a template is probed at the prefix. The
    pipeline config already gives each stage prefix one owning stage.
    """
    for route in routes:
        path = getattr(route, "path", None)
        if isinstance(route, Mount):
            template = route.path + "/{path:path}"
        else:
            template = path
        for prefix in stage.prefixes:
            probes = [
                {"type": kind, "path": prefix, "method": "GET", "headers": []}
                for kind in DISCONNECT
            ]
            if (template is not None and template_reaches(template, prefix)) or any(
                route.matches(scope)[0] != Match.NONE for scope in probes
            ):
                raise StageRoutesError(
                    f"Stage {stage.stage} http_routes entry {prefix!r} reaches "
                    f"the route {path or route!r} that the server already serves. "
                    "Give each path one owner."
                )
            else:
                pass


async def write_frame(writer: asyncio.StreamWriter, message: object) -> None:
    data = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    writer.write(FRAME_HEADER.pack(len(data)) + data)
    # The socket takes small frames at once, so wait only under backpressure.
    if writer.transport.get_write_buffer_size():
        await writer.drain()
    else:
        pass


async def read_frame(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    """The next message, or None once the peer closed the connection."""
    try:
        header = await reader.readexactly(FRAME_HEADER.size)
        (size,) = FRAME_HEADER.unpack(header)
        return pickle.loads(await reader.readexactly(size))
    except (asyncio.IncompleteReadError, ConnectionError):
        return None


def relayed_scope(scope: Scope) -> dict[str, object]:
    out = {key: scope[key] for key in RELAYED_SCOPE_KEYS if key in scope}
    # Header and address tuples cross as lists of plain values.
    out["headers"] = [(bytes(k), bytes(v)) for k, v in scope.get("headers", [])]
    for key in ("client", "server"):
        if out.get(key) is not None:
            out[key] = tuple(out[key])
        else:
            pass
    return out


class StageRoutes(BaseRoute):
    """Relay the requests under a stage's path prefixes to its socket.

    The route matches HTTP and WebSocket requests whose path is a prefix or
    lies below one, keeps the path unchanged, and answers 503 with the cause
    when the stage is not serving. Omni's own routes never depend on being
    registered before this one, since it matches only the declared prefixes.
    """

    def __init__(
        self,
        stage: str,
        prefixes: Iterable[str],
        socket_path: str,
        keep_alive_s: float = KEEP_ALIVE_S,
    ) -> None:
        self.stage = stage
        self.prefixes = normalize_route_prefixes(stage, prefixes)
        self.socket_path = socket_path
        self.keep_alive_s = keep_alive_s
        # Idle connections of the event loop that opened them, oldest first.
        # A request reuses the newest, so the oldest ones up to the fewest the
        # pool held since the last sweep stayed idle all along. One timer per
        # keep-alive, armed while any is idle, closes those, so a connection
        # closes after one to two keep-alive periods idle and no request
        # starts a timer.
        self.idle: collections.deque[
            tuple[asyncio.StreamReader, asyncio.StreamWriter]
        ] = collections.deque()
        self.untouched = 0
        self.sweep: asyncio.TimerHandle | None = None
        self.idle_loop: asyncio.AbstractEventLoop | None = None

    def matches(self, scope: Scope) -> tuple[Match, Scope]:
        if scope["type"] in DISCONNECT and any(
            covers(prefix, scope["path"]) for prefix in self.prefixes
        ):
            return Match.FULL, {}
        else:
            return Match.NONE, {}

    def url_path_for(self, name: str, /, **path_params: object) -> URLPath:
        raise NoMatchFound(name, path_params)

    async def connect(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        loop = asyncio.get_running_loop()
        if self.idle_loop is not loop:
            self.idle = collections.deque()
            self.untouched = 0
            self.sweep = None
            self.idle_loop = loop
        else:
            pass
        while self.idle:
            reader, writer = self.idle.pop()
            if len(self.idle) < self.untouched:
                self.untouched = len(self.idle)
            else:
                pass
            # A stage that stopped closed its idle connections.
            if reader.at_eof() or writer.is_closing():
                writer.close()
            else:
                return reader, writer
        return await asyncio.open_unix_connection(self.socket_path)

    def park(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Keep a connection for the next request until the keep-alive ends."""
        self.idle.append((reader, writer))
        if self.sweep is None:
            loop = self.idle_loop
            self.sweep = loop.call_later(self.keep_alive_s, self.close_idle, loop)
        else:
            pass

    def close_idle(self, loop: asyncio.AbstractEventLoop) -> None:
        """Close the connections that stayed idle since the last sweep."""
        self.sweep = None
        if loop is not self.idle_loop:
            return
        else:
            pass
        for _ in range(self.untouched):
            self.idle.popleft()[1].close()
        self.untouched = len(self.idle)
        if self.idle:
            self.sweep = loop.call_later(self.keep_alive_s, self.close_idle, loop)
        else:
            pass

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Serve one request through the stage's app."""
        stage = self.stage
        try:
            reader, writer = await self.connect()
        except OSError as exc:
            await refuse(scope, receive, send, f"Stage {stage} is not serving: {exc}")
            return
        started = False
        reusable = False
        disconnect = DISCONNECT[scope["type"]]

        async def forward_client() -> None:
            while True:
                message = await receive()
                await write_frame(writer, message)
                if message["type"] == disconnect:
                    return
                else:
                    pass

        upstream = asyncio.ensure_future(forward_client())

        async def close_upstream() -> None:
            upstream.cancel()
            with contextlib.suppress(BaseException):
                await upstream

        try:
            await write_frame(writer, relayed_scope(scope))
            while True:
                message = await read_frame(reader)
                if message is None and started:
                    raise ConnectionError(
                        f"Stage {stage} closed the connection mid-response"
                    )
                elif message is None:
                    await close_upstream()
                    await refuse(
                        scope, receive, send, f"Stage {stage} stopped before answering"
                    )
                    return
                elif message == END_FRAME:
                    await close_upstream()
                    await write_frame(writer, DONE_FRAME)
                    reusable = True
                    return
                elif message.get("type") == ERROR_FRAME_TYPE:
                    raise RuntimeError(f"Stage {stage} failed while answering")
                else:
                    started = started or message["type"] in (
                        "http.response.start",
                        "websocket.accept",
                        "websocket.close",
                    )
                    await send(message)
        except OSError:
            if started:
                raise
            else:
                await close_upstream()
                await refuse(scope, receive, send, f"Stage {stage} stopped serving")
        finally:
            await close_upstream()
            if reusable and asyncio.get_running_loop() is self.idle_loop:
                self.park(reader, writer)
            else:
                writer.close()


async def refuse(scope: Scope, receive: Receive, send: Send, reason: str) -> None:
    if scope["type"] == "http":
        await JSONResponse({"detail": reason}, status_code=503)(scope, receive, send)
    else:
        # Close the handshake with "try again later".
        await send({"type": "websocket.close", "code": 1013, "reason": reason[:120]})


class StageHttpServer:
    """Serve one ASGI app on a Unix socket from a thread with its own loop.

    start runs the app's lifespan startup, then listens. It returns once the
    socket accepts connections and raises the startup failure otherwise. stop
    closes the socket, ends the requests still running, runs the lifespan
    shutdown, and removes the socket file. The app keeps its own event loop,
    apart from the stage's, so its tasks and clients bind to that loop.
    """

    def __init__(self, name: str, app: ASGIApp, socket_path: str) -> None:
        self.name = name
        self.app = app
        self.socket_path = socket_path
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self.started = threading.Event()
        self.startup_error: BaseException | None = None
        self.stopping: asyncio.Event | None = None
        self.state: dict[str, object] = {}
        self.connections: set[asyncio.Task[None]] = set()

    def start(self, timeout: float | None = None) -> None:
        if self.thread is not None:
            raise RuntimeError(f"Stage {self.name} HTTP server already started")
        else:
            pass
        self.thread = threading.Thread(
            target=self.run, name=f"stage-http-{self.name}", daemon=True
        )
        self.thread.start()
        if not self.started.wait(timeout):
            raise TimeoutError(f"Stage {self.name} HTTP server did not start")
        elif self.startup_error is not None:
            self.thread.join()
            raise StageRoutesError(
                f"Stage {self.name} HTTP routes failed to start: "
                f"{self.startup_error}"
            ) from self.startup_error
        else:
            pass

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        self.loop = loop
        try:
            loop.run_until_complete(self.serve())
        except BaseException as exc:
            if not self.started.is_set():
                self.startup_error = exc
            else:
                logger.exception("Stage %s HTTP server failed", self.name)
        finally:
            self.started.set()
            loop.close()

    async def serve(self) -> None:
        self.stopping = asyncio.Event()
        lifespan = Lifespan(self.app, self.state)
        await lifespan.startup()
        server = None
        try:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(self.socket_path)
            server = await asyncio.start_unix_server(
                self.accept, path=self.socket_path, backlog=LISTEN_BACKLOG
            )
            os.chmod(self.socket_path, 0o600)
            self.started.set()
            await self.stopping.wait()
        finally:
            if server is not None:
                server.close()
            else:
                pass
            for task in list(self.connections):
                task.cancel()
            await asyncio.gather(*self.connections, return_exceptions=True)
            with contextlib.suppress(FileNotFoundError):
                os.unlink(self.socket_path)
            await lifespan.shutdown()

    def stop(self, timeout: float | None = None) -> None:
        loop = self.loop
        thread = self.thread
        if thread is None or loop is None:
            return
        else:
            pass
        if not loop.is_closed():
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self.request_stop)
        else:
            pass
        thread.join(timeout)
        if thread.is_alive():
            # A request that outlives the stop may still reach the scheduler.
            raise StageRoutesError(
                f"Stage {self.name} HTTP server did not stop within {timeout} s"
            )
        else:
            pass

    def request_stop(self) -> None:
        if self.stopping is not None:
            self.stopping.set()
        else:
            pass

    async def accept(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        assert task is not None, "asyncio runs every connection in a task"
        self.connections.add(task)
        try:
            await self.serve_connection(reader, writer)
        finally:
            self.connections.discard(task)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def serve_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Serve the requests that arrive on one connection, one at a time."""
        while True:
            scope = await read_frame(reader)
            if not isinstance(scope, dict) or scope.get("type") not in DISCONNECT:
                return
            else:
                pass
            last = await self.serve_request(scope, reader, writer)
            try:
                await write_frame(writer, last)
            except OSError:
                return
            # The relay forwards client messages until it reads the last
            # frame, then marks the request done. Closing earlier would fail
            # its next write, which discards the answer it has not read.
            message = await read_frame(reader)
            while message is not None and message != DONE_FRAME:
                message = await read_frame(reader)
            if message is None:
                return
            else:
                pass

    async def serve_request(
        self,
        scope: dict[str, Any],
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> dict[str, str]:
        scope["state"] = dict(self.state)
        disconnect = {"type": DISCONNECT[scope["type"]]}
        gone = False

        async def receive() -> MutableMapping[str, Any]:
            nonlocal gone
            if gone:
                return dict(disconnect)
            else:
                pass
            message = await read_frame(reader)
            if message is None or message.get("type") == disconnect["type"]:
                gone = True
                return dict(disconnect)
            else:
                return message

        async def send(message: MutableMapping[str, Any]) -> None:
            await write_frame(writer, message)

        try:
            await self.app(scope, receive, send)
        except Exception:
            logger.exception("Stage %s HTTP route failed", self.name)
            return {"type": ERROR_FRAME_TYPE}
        else:
            return END_FRAME


class Lifespan:
    """Drive an ASGI app's lifespan the way an ASGI server does.

    An app that does not support lifespan, which raises on the lifespan scope
    before it sends anything, starts and stops without it.
    """

    def __init__(self, app: ASGIApp, state: dict[str, object]) -> None:
        self.app = app
        self.state = state
        self.inbox: asyncio.Queue[dict[str, str]] = asyncio.Queue()
        self.started: asyncio.Future[None] | None = None
        self.stopped: asyncio.Future[None] | None = None
        self.task: asyncio.Task[None] | None = None
        self.supported = True

    async def startup(self) -> None:
        loop = asyncio.get_running_loop()
        self.started = loop.create_future()
        self.stopped = loop.create_future()
        self.task = asyncio.ensure_future(self.run())
        await self.inbox.put({"type": "lifespan.startup"})
        await asyncio.wait(
            (self.started, self.task), return_when=asyncio.FIRST_COMPLETED
        )
        if self.started.done():
            self.started.result()
        elif self.supported:
            raise StageRoutesError("the app ended its lifespan during startup")
        else:
            pass

    async def run(self) -> None:
        scope = {
            "type": "lifespan",
            "asgi": {"version": "3.0", "spec_version": "2.0"},
            "state": self.state,
        }
        sent = False

        async def receive() -> dict[str, str]:
            return await self.inbox.get()

        async def send(message: MutableMapping[str, Any]) -> None:
            nonlocal sent
            sent = True
            settle = {
                "lifespan.startup.complete": (self.started, None),
                "lifespan.startup.failed": (self.started, "startup"),
                "lifespan.shutdown.complete": (self.stopped, None),
                "lifespan.shutdown.failed": (self.stopped, "shutdown"),
            }.get(message["type"])
            if settle is None:
                pass
            elif settle[1] is None:
                settle[0].set_result(None)
            else:
                settle[0].set_exception(
                    StageRoutesError(
                        f"lifespan {settle[1]} failed: {message.get('message', '')}"
                    )
                )

        try:
            await self.app(scope, receive, send)
        except Exception as exc:
            if not sent:
                self.supported = False
            elif not self.started.done():
                self.started.set_exception(
                    StageRoutesError(f"lifespan startup failed: {exc}")
                )
            elif self.started.exception() is not None:
                pass  # The app reported its startup failure, which start raises.
            elif not self.stopped.done():
                self.stopped.set_exception(
                    StageRoutesError(f"lifespan shutdown failed: {exc}")
                )
            else:
                logger.warning("Lifespan ended with an error: %s", exc)

    async def shutdown(self) -> None:
        task = self.task
        if task is None or task.done() or not self.supported:
            return
        else:
            pass
        await self.inbox.put({"type": "lifespan.shutdown"})
        await asyncio.wait((self.stopped, task), return_when=asyncio.FIRST_COMPLETED)
        if self.stopped.done() and self.stopped.exception() is not None:
            logger.warning("%s", self.stopped.exception())
        else:
            pass
