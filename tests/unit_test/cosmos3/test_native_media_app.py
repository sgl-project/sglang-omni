# SPDX-License-Identifier: Apache-2.0
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang_omni.serve.native_media import mount_native_media_app


def test_native_routes_and_lifetime_are_composed_without_a_proxy():
    events = []

    @asynccontextmanager
    async def omni_lifespan(app):
        events.append("omni_start")
        yield
        events.append("omni_stop")

    @asynccontextmanager
    async def native_lifespan(app):
        events.append("native_start")
        app.state.ready = True
        yield
        events.append("native_stop")

    omni = FastAPI(lifespan=omni_lifespan)
    native = FastAPI(lifespan=native_lifespan)

    @omni.get("/health")
    def health():
        return {"owner": "omni"}

    @native.get("/health")
    def native_health():
        return {"owner": "native"}

    @native.post("/v1/images/generations")
    def generate():
        assert native.state.ready
        return {"owner": "native"}

    mount_native_media_app(omni, native)
    with TestClient(omni) as client:
        assert events == ["omni_start", "native_start"]
        assert client.get("/health").json() == {"owner": "omni"}
        assert client.post("/v1/images/generations").json() == {"owner": "native"}
    assert events == ["omni_start", "native_start", "native_stop", "omni_stop"]


def test_unrelated_models_select_their_own_native_frontend(monkeypatch):
    import sys
    from types import ModuleType
    from typing import ClassVar

    from sglang_omni.config import PipelineConfig, StageConfig
    from sglang_omni.serve.native_media import prepare_native_media_app

    selected = []

    def make_module(name):
        module = ModuleType(name)

        def create(config, *, host, port):
            selected.append((name, config.model_path, host, port))
            app = FastAPI()

            @app.get("/model-adapter")
            def identify():
                return {"adapter": name, "model": config.model_path}

            return app

        module.create = create
        monkeypatch.setitem(sys.modules, name, module)
        return name + ".create"

    class ModelA(PipelineConfig):
        native_media_stage: ClassVar[str] = "native"
        native_media_factory_path: ClassVar[str] = make_module("_umm_test_a")
        stages: list[StageConfig] = [
            StageConfig(
                name="native",
                process="native",
                factory_path="unused.stage_factory",
                terminal=True,
            )
        ]

    class ModelB(ModelA):
        native_media_factory_path: ClassVar[str] = make_module("_umm_test_b")

    for config, name in [
        (ModelA(model_path="checkpoint-a"), "_umm_test_a"),
        (ModelB(model_path="checkpoint-b"), "_umm_test_b"),
    ]:
        app = prepare_native_media_app(config, host="127.0.0.1", port=4321)
        with TestClient(app) as client:
            assert client.get("/model-adapter").json() == {
                "adapter": name,
                "model": config.model_path,
            }

    assert selected == [
        ("_umm_test_a", "checkpoint-a", "127.0.0.1", 4321),
        ("_umm_test_b", "checkpoint-b", "127.0.0.1", 4321),
    ]


def test_inactive_native_frontend_does_not_import_model_factory(monkeypatch):
    from types import SimpleNamespace

    from sglang_omni.serve import native_media

    def unexpected_import(path):
        raise AssertionError("An inactive native frontend imported a model")

    monkeypatch.setattr(native_media, "import_string", unexpected_import)
    config = SimpleNamespace(
        native_media_stage=None,
        native_media_factory_path="unavailable.model.factory",
    )
    assert (
        native_media.prepare_native_media_app(config, host="localhost", port=1) is None
    )


def test_native_frontend_without_model_factory_fails_before_loading():
    from types import SimpleNamespace

    import pytest

    from sglang_omni.serve.native_media import prepare_native_media_app

    config = SimpleNamespace(
        native_media_stage="generation",
        native_media_factory_path=None,
    )
    with pytest.raises(ValueError, match="native_media_factory_path"):
        prepare_native_media_app(config, host="localhost", port=1)


def test_model_factory_must_return_a_native_application(monkeypatch):
    from types import SimpleNamespace

    import pytest

    from sglang_omni.serve import native_media

    monkeypatch.setattr(
        native_media, "import_string", lambda path: lambda *a, **k: None
    )
    config = SimpleNamespace(
        native_media_stage="generation",
        native_media_factory_path="model.factory",
    )
    with pytest.raises(TypeError, match="FastAPI application"):
        native_media.prepare_native_media_app(config, host="localhost", port=1)


def test_cosmos3_declares_model_owned_native_frontend():
    from sglang_omni.models.cosmos3.config import Cosmos3PipelineConfig

    assert Cosmos3PipelineConfig.native_media_factory_path == (
        "sglang_omni.models.cosmos3.media.prepare_native_media_app"
    )


def test_native_lifespan_receives_the_existing_runtime_failure_signal():
    import asyncio

    async def run():
        failure = asyncio.get_running_loop().create_future()
        observed = []

        @asynccontextmanager
        async def native_lifespan(app):
            observed.append(app.state.scheduler_failure)
            yield

        omni = FastAPI()
        native = FastAPI(lifespan=native_lifespan)
        mount_native_media_app(omni, native, runtime_failure=failure)
        async with omni.router.lifespan_context(omni):
            assert observed == [failure]
            assert not failure.done()
        assert not failure.done()
        failure.cancel()

    asyncio.run(run())
