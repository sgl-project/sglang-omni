# SPDX-License-Identifier: Apache-2.0
"""HTTP routes a stage serves from its own process, relayed by the Omni server."""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from sglang_omni.config.schema import StageConfig
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.serve.stage_routes import (
    StageHttpServer,
    StageRoutes,
    StageRoutesError,
)


def stage_app(events: dict) -> Starlette:
    @asynccontextmanager
    async def lifespan(app):
        events["started"] = True
        yield {"greeting": "ready"}
        events["stopped"] = True

    async def echo(request: Request) -> JSONResponse:
        body = await request.body()
        return JSONResponse(
            {
                "path": request.url.path,
                "query": request.url.query,
                "header": request.headers.get("x-probe"),
                "bytes": len(body),
                "state": request.state.greeting,
            },
            headers={"x-stage": "yes"},
        )

    async def chunks(request: Request) -> StreamingResponse:
        async def parts():
            for i in range(3):
                yield f"part{i},".encode()

        return StreamingResponse(parts(), media_type="text/plain")

    async def hold(request: Request) -> JSONResponse:
        events["held"].set()
        await asyncio.Event().wait()
        return JSONResponse({})

    return Starlette(
        routes=[
            Route("/native/echo/{name}", echo, methods=["POST"]),
            Route("/native/chunks", chunks),
            Route("/native/hold", hold),
        ],
        lifespan=lifespan,
    )


def serve(app, socket_dir: str) -> tuple[StageHttpServer, Starlette]:
    socket_path = os.path.join(socket_dir, "http_gen.sock")
    server = StageHttpServer("gen", app, socket_path)
    server.start(timeout=10)
    front = Starlette(routes=[StageRoutes("gen", ["/native"], socket_path)])
    return server, front


async def call(front: Starlette, method: str, path: str, **kwargs) -> httpx.Response:
    transport = httpx.ASGITransport(app=front)
    async with httpx.AsyncClient(transport=transport, base_url="http://omni") as c:
        return await c.request(method, path, **kwargs)


def test_requests_relay_to_the_stage_app_with_bodies_streams_and_lifespan():
    events = {"held": threading.Event()}
    with tempfile.TemporaryDirectory() as socket_dir:
        server, front = serve(stage_app(events), socket_dir)
        try:
            assert events.get("started")
            payload = b"x" * (3 << 20)
            echoed = asyncio.run(
                call(
                    front,
                    "POST",
                    "/native/echo/a?b=1",
                    content=payload,
                    headers={"x-probe": "p"},
                )
            )
            streamed = asyncio.run(call(front, "GET", "/native/chunks"))
            outside = asyncio.run(call(front, "GET", "/other"))
        finally:
            server.stop(timeout=10)
        assert events.get("stopped")
        assert not os.path.exists(os.path.join(socket_dir, "http_gen.sock"))
    assert echoed.status_code == 200
    assert echoed.headers["x-stage"] == "yes"
    assert echoed.json() == {
        "path": "/native/echo/a",
        "query": "b=1",
        "header": "p",
        "bytes": len(payload),
        "state": "ready",
    }
    assert streamed.text == "part0,part1,part2,"
    assert outside.status_code == 404


def test_a_stage_that_is_not_serving_answers_503_with_the_cause():
    with tempfile.TemporaryDirectory() as socket_dir:
        front = Starlette(
            routes=[StageRoutes("gen", ["/native"], os.path.join(socket_dir, "x"))]
        )
        response = asyncio.run(call(front, "GET", "/native/chunks"))
    assert response.status_code == 503
    assert "Stage gen is not serving" in response.json()["detail"]


def test_a_request_in_flight_when_the_stage_stops_answers_503():
    events = {"held": threading.Event()}
    with tempfile.TemporaryDirectory() as socket_dir:
        server, front = serve(stage_app(events), socket_dir)

        async def scenario() -> httpx.Response:
            pending = asyncio.ensure_future(call(front, "GET", "/native/hold"))
            await asyncio.to_thread(events["held"].wait, 10)
            await asyncio.to_thread(server.stop, 10)
            return await pending

        response = asyncio.run(scenario())
    assert response.status_code == 503
    assert "Stage gen stopped before answering" in response.json()["detail"]


def test_a_client_disconnect_reaches_the_stage_app():
    seen = []

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            raise RuntimeError("no lifespan")
        else:
            pass
        while True:
            message = await receive()
            seen.append(message["type"])
            if message["type"] == "http.disconnect":
                return
            else:
                pass

    with tempfile.TemporaryDirectory() as socket_dir:
        socket_path = os.path.join(socket_dir, "http_gen.sock")
        server = StageHttpServer("gen", app, socket_path)
        server.start(timeout=10)
        try:
            messages = iter(
                [
                    {"type": "http.request", "body": b"ab", "more_body": False},
                    {"type": "http.disconnect"},
                ]
            )
            sent = []

            async def receive():
                return next(messages)

            async def send(message):
                sent.append(message)

            scope = {
                "type": "http",
                "method": "POST",
                "path": "/native/x",
                "headers": [],
                "query_string": b"",
            }
            route = StageRoutes("gen", ["/native"], socket_path)
            asyncio.run(route.handle(scope, receive, send))
        finally:
            server.stop(timeout=10)
    assert seen == ["http.request", "http.disconnect"]
    # A client that left gets nothing, as when the app serves it directly.
    assert sent == []


def test_a_failed_lifespan_startup_fails_the_start():
    @asynccontextmanager
    async def lifespan(app):
        raise RuntimeError("engine client could not start")
        yield

    with tempfile.TemporaryDirectory() as socket_dir:
        server = StageHttpServer(
            "gen", Starlette(lifespan=lifespan), os.path.join(socket_dir, "s")
        )
        with pytest.raises(StageRoutesError, match="engine client could not start"):
            server.start(timeout=10)


def test_routes_need_a_scheduler_that_provides_an_app_and_valid_prefixes():
    stage = SimpleNamespace(name="gen", scheduler=object())
    with pytest.raises(StageRoutesError, match="provides no http_app"):
        asyncio.run(Stage.start_http_routes(stage, "/unused"))
    assert StageConfig(
        name="g", factory_path="f", http_routes=["/v1/a/"]
    ).http_routes == ["/v1/a"]
    for bad in (["v1"], ["/"], ["/a", "/a/"]):
        with pytest.raises(ValueError):
            StageConfig(name="g", factory_path="f", http_routes=bad)
