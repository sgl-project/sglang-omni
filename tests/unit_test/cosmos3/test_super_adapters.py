# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests. Native modules are replaced only within each test.

These tests do not load weights or qualify native execution/partial rank startup.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI

from sglang_omni.models.cosmos3 import checkpoint, media, reasoner, stages
from sglang_omni.models.cosmos3.config import Cosmos3PipelineConfig


@pytest.fixture
def snapshot(tmp_path):
    target = tmp_path / "snapshot"
    target.mkdir()
    # Selected public Super metadata. Native owns interpreting this geometry.
    config = {
        "architectures": ["Cosmos3ForConditionalGeneration"],
        "model_type": "cosmos3_omni",
        "text_config": {
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 128,
        },
    }
    (target / "config.json").write_text(json.dumps(config))
    return target


@pytest.mark.parametrize("name", ["nvidia/Cosmos3-Nano", "nvidia/Cosmos3-Super"])
@pytest.mark.parametrize("alias", [None, "my-cosmos"])
def test_pinned_checkpoint_preserves_name_and_metadata(
    monkeypatch, snapshot, name, alias
):
    downloaded = []
    before = (snapshot / "config.json").read_bytes()
    monkeypatch.setattr(
        checkpoint,
        "resolve_checkpoint",
        lambda value: downloaded.append(value) or str(snapshot),
    )
    original = {"model_path": name + "@abc123", "served_model_name": alias}
    resolved = checkpoint.resolve_native_checkpoint(original)
    assert downloaded == [name + "@abc123"]
    assert resolved["model_path"] == str(snapshot)
    assert resolved["served_model_name"] == (alias or name)
    assert original["model_path"] == name + "@abc123"
    assert (snapshot / "config.json").read_bytes() == before


def test_local_path_containing_at_is_not_split(tmp_path):
    local = tmp_path / "super@experiment"
    local.mkdir()
    assert checkpoint.resolve_native_checkpoint({"model_path": str(local)}) == {
        "model_path": str(local),
        "served_model_name": str(local),
    }


@pytest.mark.parametrize("kind", ["generation", "reasoner"])
def test_invalid_placement_fails_before_download_or_native_start(monkeypatch, kind):
    def unexpected(*args):
        pytest.fail("Invalid placement attempted a checkpoint download")

    monkeypatch.setattr(checkpoint, "resolve_checkpoint", unexpected)
    factory = (
        stages.create_generation_scheduler
        if kind == "generation"
        else reasoner.create_reasoner_scheduler
    )
    with pytest.raises(ValueError, match="placement"):
        factory("nvidia/Cosmos3-Super@revision", gpu_id=0, runtime_gpu_ids=[1, 2])


@pytest.mark.parametrize("concurrency", [0, -1, True, 1.5])
def test_invalid_concurrency_fails_before_native_import(concurrency):
    with pytest.raises(ValueError, match="positive integer"):
        reasoner.create_reasoner_scheduler(
            "nvidia/Cosmos3-Super", max_concurrency=concurrency
        )


@pytest.mark.parametrize("kind", ["generation", "reasoner"])
def test_adapter_construction_failure_releases_runtime_and_allows_retry(
    native, snapshot, tmp_path, monkeypatch, kind
):
    owner = stages if kind == "generation" else reasoner
    name = (
        "NativeGenerationScheduler"
        if kind == "generation"
        else "NativeReasonerScheduler"
    )
    factory = (
        stages.create_generation_scheduler
        if kind == "generation"
        else reasoner.create_reasoner_scheduler
    )
    kwargs = {"output_dir": str(tmp_path / "outputs")} if kind == "generation" else {}

    def fail(*args):
        raise RuntimeError("adapter initialization failed")

    with monkeypatch.context() as patch:
        patch.setattr(owner, name, fail)
        with pytest.raises(RuntimeError, match="adapter initialization failed"):
            factory(str(snapshot), runtime_gpu_ids=[0, 1], **kwargs)
    assert native.shutdown == [kind]
    # Retry obtains a new native owner; no failed adapter is returned to serve.
    monkeypatch.setattr(owner, name, lambda *args: args[0])
    runtime = factory(str(snapshot), runtime_gpu_ids=[0, 1], **kwargs)
    runtime.shutdown()
    assert native.shutdown == [kind, kind]


def test_http_and_sdk_share_one_snapshot_even_if_hub_revision_moves(
    native, snapshot, tmp_path, monkeypatch
):
    resolutions = []

    def resolve(value):
        resolutions.append(value)
        if Path(value).is_dir():
            return value
        assert len(resolutions) == 1, "Hub revision was resolved a second time"
        return str(snapshot)

    monkeypatch.setattr(checkpoint, "resolve_checkpoint", resolve)
    monkeypatch.setattr(media, "_unused_port", lambda excluded: max(excluded) + 10)
    config = Cosmos3PipelineConfig(model_path="nvidia/Cosmos3-Super@campaign")
    stage = config.stages[0]
    stage.runtime_gpu_ids = [0, 1, 2, 3]
    stage.factory.output_dir = str(tmp_path / "outputs")
    media.prepare_native_media_app(config, host="127.0.0.1", port=8000)
    from sglang_omni.config.runtime import resolve_stage_typed_kwargs

    # The same resolver used by the runner must pass the frontend's snapshot.
    kwargs = resolve_stage_typed_kwargs(stage)
    scheduler = stages.create_generation_scheduler(
        **kwargs, runtime_gpu_ids=stage.runtime_gpu_ids
    )
    try:
        frontend, worker = native.apps[0], native.generation[0]
        assert frontend.model_path == worker.model_path == str(snapshot)
        assert (
            frontend.served_model_name
            == worker.served_model_name
            == "nvidia/Cosmos3-Super"
        )
        assert frontend.gpu_ids == worker.gpu_ids == [0, 1, 2, 3]
        assert frontend.scheduler_port == worker.scheduler_port
        assert config.model_path == "nvidia/Cosmos3-Super@campaign"
    finally:
        scheduler.stop()


def test_failed_frontend_does_not_publish_partial_factory_state(
    native, snapshot, monkeypatch
):
    monkeypatch.setattr(media, "_unused_port", lambda excluded: max(excluded) + 10)
    module = native.modules["sglang.multimodal_gen.runtime.entrypoints.http_server"]
    original_create = module.create_app

    def fail(args):
        raise RuntimeError("frontend failed")

    monkeypatch.setattr(module, "create_app", fail)
    config = Cosmos3PipelineConfig(model_path=str(snapshot))
    before = config.model_dump()
    with pytest.raises(RuntimeError, match="frontend failed"):
        media.prepare_native_media_app(config, host="127.0.0.1", port=8000)
    assert config.model_dump() == before
    assert native.generation == []
    monkeypatch.setattr(module, "create_app", original_create)
    assert isinstance(
        media.prepare_native_media_app(config, host="127.0.0.1", port=8000), FastAPI
    )


def test_frontend_honors_stage_checkpoint_and_explicit_alias(
    native, snapshot, monkeypatch
):
    monkeypatch.setattr(media, "_unused_port", lambda excluded: max(excluded) + 10)
    config = Cosmos3PipelineConfig(model_path="unused-root-checkpoint")
    config.stages[0].factory.model_path = str(snapshot)
    config.stages[0].factory.server_args_overrides = {
        "served_model_name": "super-local"
    }
    media.prepare_native_media_app(config, host="127.0.0.1", port=8000)
    assert native.apps[0].model_path == str(snapshot)
    assert native.apps[0].served_model_name == "super-local"
