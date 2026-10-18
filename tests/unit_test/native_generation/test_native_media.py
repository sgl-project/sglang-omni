# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
import tempfile
from types import SimpleNamespace

import httpx
import pytest

from sglang_omni.native_generation import ROUTE_PREFIXES, media
from sglang_omni.serve.stage_routes import (
    SERVER_PORT_ENV,
    StageHttpServer,
    StageRoutesError,
)


class NativePipeline:
    def supports_action_endpoint(self):
        return True

    def supports_openpi_endpoint(self):
        return True


@pytest.fixture
def native(monkeypatch):
    """Native runtime pieces the routes call, without a scheduler."""
    from sglang.multimodal_gen.runtime import scheduler_client, server_args

    args = SimpleNamespace(
        enable_metrics=False,
        prompt_enhancer_config=None,
        scheduler_endpoints=[],
        pipeline_config=NativePipeline(),
    )
    native = SimpleNamespace(args=args, events=[])
    fields = frozenset({"control_path", "image_path", "prompt", "video_path"})
    native.build = lambda: media.NativeMediaApp(native.args, fields)

    def initialize(self, server_args):
        native.events.append("initialize")

    def close(self):
        native.events.append("close")

    monkeypatch.setattr(
        server_args,
        "set_global_server_args",
        lambda args: native.events.append("build"),
    )
    monkeypatch.setattr(scheduler_client.AsyncSchedulerClient, "initialize", initialize)
    monkeypatch.setattr(scheduler_client.AsyncSchedulerClient, "close", close)
    return native


def serve(app, check):
    """Run check(client, app) against the routes inside their lifespan."""

    async def run():
        async with app.app.router.lifespan_context(app.app):
            # Native handlers past the checks under test may fail without a
            # scheduler, which then answers 500.
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://stage"
            ) as client:
                await check(client, app)

    asyncio.run(run())


def test_native_routes_start_with_their_lifespan(native):
    app = native.build()
    assert native.events == ["build"]

    async def check(client, app):
        assert native.events == ["build", "initialize"]
        assert (await client.get("/v1/videos")).status_code == 200

        @app.app.get("/v1/images/owner")
        def owner():
            return {"owner": "native"}

        assert (await client.get("/v1/images/owner")).json() == {"owner": "native"}

    serve(app, check)
    assert native.events == ["build", "initialize", "close"]


# Native create_app also serves these, none of them behind Omni's admin key.
NATIVE_CONTROL_ROUTES = [
    ("post", "/release_memory_occupation"),
    ("post", "/resume_memory_occupation"),
    ("post", "/get_weights_checksum"),
    ("post", "/update_weights_from_tensor_checker"),
    ("post", "/v1/set_lora"),
    ("post", "/v1/merge_lora_weights"),
    ("post", "/v1/unmerge_lora_weights"),
    ("get", "/v1/list_loras"),
    ("get", "/v1/model_info"),
    ("post", "/rollout/generate"),
    ("get", "/server_info"),
    ("get", "/stats"),
    ("get", "/models"),
    ("post", "/vertex_generate"),
    ("post", "/v1/meshes"),
]


def test_only_native_media_routes_are_served(native):
    async def check(client, app):
        for method, path in NATIVE_CONTROL_ROUTES:
            response = await getattr(client, method)(path)
            assert response.status_code == 404, path
        assert (await client.get("/v1/videos")).json()["object"] == "list"
        response = await client.post("/v1/images/generations", json={})
        assert response.status_code == 422

    serve(native.build(), check)


def test_routes_that_cannot_start_fail_the_stage_start(native, tmp_path):
    # A prompt enhancer config that cannot be read fails the native lifespan.
    native.args.prompt_enhancer_config = str(tmp_path / "missing-enhancer.json")
    with tempfile.TemporaryDirectory() as socket_dir:
        server = StageHttpServer(
            "generation", native.build(), os.path.join(socket_dir, "s")
        )
        with pytest.raises(StageRoutesError, match="missing-enhancer.json"):
            server.start(timeout=30)
    # The scheduler client that started closed again.
    assert native.events == ["build", "initialize", "close"]


def test_in_flight_native_request_fails_when_the_scheduler_is_lost(native):
    import zmq

    context = zmq.Context()
    scheduler = context.socket(zmq.ROUTER)
    port = scheduler.bind_to_random_port("tcp://127.0.0.1")
    native.args.scheduler_endpoints = [f"tcp://127.0.0.1:{port}"]
    entered = asyncio.Event()

    async def waiting():
        # Native SGLang waits like this for a scheduler that never answers.
        entered.set()
        await asyncio.Event().wait()

    async def check(client, app):
        app.app.post("/v1/images/generations-waiting")(waiting)

        async def connected():
            while list(app.scheduler_events.values()) != [zmq.EVENT_CONNECTED]:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(connected(), 5)
        pending = asyncio.ensure_future(client.post("/v1/images/generations-waiting"))
        await asyncio.wait_for(entered.wait(), 5)
        scheduler.close(linger=0)
        response = await asyncio.wait_for(pending, 5)
        assert response.status_code == 503
        response = await client.post("/v1/images/generations-waiting")
        assert response.status_code == 503

    try:
        serve(native.build(), check)
    finally:
        scheduler.close(linger=0)
        context.term()


@pytest.mark.parametrize("configured", [False, True])
def test_native_requests_follow_the_server_media_policy(
    native, monkeypatch, tmp_path, configured
):
    import json

    from sglang_omni.preprocessing.resource_connector import MultiModalResourceConnector

    allowed = tmp_path / "media"
    allowed.mkdir()
    outside = str(tmp_path / "clip.mp4")
    inside = str(allowed / "clip.mp4")
    connector = MultiModalResourceConnector(
        allowed_local_media_path=allowed if configured else None,
        allowed_media_domains=["media.example"] if configured else None,
    )
    monkeypatch.setattr(media, "get_global_resource_connector", lambda: connector)
    remote = "https://media.example/frame.png"
    # Every request reaches a native media route. A refusal names the field.
    requests = [
        ("/v1/videos", {"json": {"prompt": "a", "output_path": "/srv"}}, True),
        ("/v1/videos", {"json": {"prompt": "a", "video_path": outside}}, configured),
        (
            "/v1/videos",
            {"json": {"prompt": "a", "video_path": "/srv/a\0b"}},
            configured,
        ),
        ("/v1/videos", {"json": {"prompt": "a", "reference_url": remote}}, configured),
        (
            "/v1/videos",
            {"json": {"prompt": "a", "reference_url": remote.upper()}},
            configured,
        ),
        (
            "/v1/videos",
            {
                "json": {
                    "prompt": "a",
                    "extra_params": json.dumps({"control_path": [outside]}),
                }
            },
            configured,
        ),
        (
            "/v1/actions/generations",
            {"json": {"input": {"observation": {"image": f"file://{outside}"}}}},
            configured,
        ),
        # Native decodes an inline action image in memory, so it is no path.
        (
            "/v1/actions/generations",
            {"json": {"input": {"observation": {"image": {"b64_json": "aGk="}}}}},
            False,
        ),
        (
            "/v1/images/edits",
            {"data": {"prompt": "a", "url": outside}},
            configured,
        ),
        (
            "/v1/images/edits",
            {"data": {"prompt": "a", "perf_dump_path": "/srv/perf.json"}},
            True,
        ),
        # A body nested past the decoder limit, which native may still decode.
        (
            "/v1/videos",
            {
                "content": "[" * 100000 + "]" * 100000,
                "headers": {"content-type": "application/json"},
            },
            True,
        ),
        # Native decodes a JSON encoded list to the allowed paths it names.
        (
            "/v1/videos",
            {"json": {"prompt": "a", "control_path": json.dumps([inside, inside])}},
            False,
        ),
        # Native decodes these JSON encoded values to the outside path.
        (
            "/v1/videos",
            {"json": {"prompt": "a", "control_path": json.dumps([outside])}},
            configured,
        ),
        (
            "/v1/videos",
            {"data": {"prompt": "a"}, "files": {"video_path": (None, f'"{outside}"')}},
            configured,
        ),
        # Native joins upload names below its upload directory.
        (
            "/v1/actions/generations",
            {"data": {"id": "../r1"}, "files": {"image_reference": ("a.png", b"x")}},
            True,
        ),
        (
            "/v1/images/edits",
            {"data": {"prompt": "a"}, "files": {"image": ("../../a.png", b"x")}},
            True,
        ),
        (
            "/v1/actions/generations",
            {"data": {"id": "r1"}, "files": {"image_reference": ("a.png", b"x")}},
            False,
        ),
    ]

    async def check(client, app):
        # Native resolves relative paths against the server working
        # directory, so the requests run with it outside and inside the
        # allowed directory.
        for cwd in (tmp_path, allowed):
            monkeypatch.chdir(cwd)
            for path, body, refused in requests:
                response = await client.post(path, **body)
                blocked = response.status_code == 400 and (
                    "refused" in response.text or "cannot set" in response.text
                )
                assert blocked == refused, (cwd, path, body, response.text)
            response = await client.post(
                "/v1/videos", json={"prompt": "a", "video_path": inside}
            )
            assert "refused" not in response.text
        # Native opens a file URL as written, relative to its working directory.
        monkeypatch.chdir(tmp_path)
        url = {"prompt": "a", "video_path": f"file://{allowed / 'clip.mp4'}"}
        response = await client.post("/v1/videos", json=url)
        assert ("refused" in response.text) is configured

    serve(native.build(), check)


def test_cosmos3_generation_stage_serves_and_settles_native_routes(monkeypatch):
    from sglang_omni.models.cosmos3.config import Cosmos3PipelineConfig
    from sglang_omni.models.cosmos3.stages import NativeGenerationScheduler

    config = Cosmos3PipelineConfig(model_path="checkpoint")
    assert config.stages[0].http_routes == list(ROUTE_PREFIXES)
    assert callable(NativeGenerationScheduler.http_app)
    monkeypatch.setenv(SERVER_PORT_ENV, "19000")
    claims = iter([29500, 19000, 29502])
    monkeypatch.setattr(media, "claim_tcp_port", lambda: next(claims))
    kwargs = media.native_media_server_kwargs({}, output_dir="/srv/out")
    # The native scheduler socket follows host, so it stays on loopback
    # whatever host the Omni HTTP server binds, and its ports avoid the
    # HTTP port the server binds once its stages started.
    assert kwargs["host"] == "127.0.0.1" and kwargs["port"] == 19000
    assert kwargs["strict_ports"] is True and kwargs["input_save_path"] is None
    assert kwargs["output_path"] == "/srv/out"
    # Its ports are claimed across processes, skipping the HTTP port.
    assert (kwargs["scheduler_port"], kwargs["master_port"]) == (29500, 29502)
    # Native checks an HTTP port without an Omni server too, and one
    # scheduler port per replica, all apart from the ports already set.
    monkeypatch.delenv(SERVER_PORT_ENV)
    claims = iter([29600, 29601, 29603, 29604, 29605])
    kwargs = media.native_media_server_kwargs(
        {"dp_size": 2, "master_port": 29601}, output_dir="/srv/out"
    )
    assert (kwargs["scheduler_port"], kwargs["master_port"], kwargs["port"]) == (
        29603,
        29601,
        29605,
    )
    with pytest.raises(ValueError, match="loopback"):
        media.native_media_server_kwargs({"host": "0.0.0.0"}, output_dir="/srv")
