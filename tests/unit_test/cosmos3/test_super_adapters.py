# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests. Native modules are replaced only within each test.

These tests do not load weights or qualify native execution/partial rank startup.
"""

from __future__ import annotations

import json

import pytest

from sglang_omni.config.runtime import resolve_stage_typed_kwargs
from sglang_omni.models.cosmos3 import stages
from sglang_omni.models.cosmos3.config import Cosmos3PipelineConfig
from sglang_omni.scheduling import native_chat_scheduler
from sglang_omni.utils import checkpoint


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
def test_invalid_placement_fails_before_download_or_native_start(
    monkeypatch, native, kind
):
    def unexpected(*args):
        pytest.fail("Invalid placement attempted a checkpoint download")

    monkeypatch.setattr(checkpoint, "resolve_checkpoint", unexpected)
    if kind == "generation":
        factory = stages.create_generation_scheduler
        kwargs = {}
    else:
        factory = native_chat_scheduler.create_native_chat_scheduler
        kwargs = {"stage_name": "reasoner"}
    with pytest.raises(ValueError, match="placement"):
        factory(
            "nvidia/Cosmos3-Super@revision", gpu_id=0, runtime_gpu_ids=[1, 2], **kwargs
        )
    assert not native.generation and not native.reasoner


@pytest.mark.parametrize("concurrency", [0, -1, True, 1.5])
def test_invalid_concurrency_fails_before_native_start(native, concurrency):
    with pytest.raises(ValueError, match="positive integer"):
        native_chat_scheduler.create_native_chat_scheduler(
            "nvidia/Cosmos3-Super", stage_name="reasoner", max_concurrency=concurrency
        )
    assert not native.reasoner


@pytest.mark.parametrize("kind", ["generation", "reasoner"])
def test_adapter_construction_failure_releases_runtime_and_allows_retry(
    native, snapshot, tmp_path, monkeypatch, kind
):
    if kind == "generation":
        owner = stages
        name = "NativeGenerationScheduler"
        factory = stages.create_generation_scheduler
        kwargs = {"output_dir": str(tmp_path / "outputs")}
    else:
        owner = native_chat_scheduler
        name = "NativeChatScheduler"
        factory = native_chat_scheduler.create_native_chat_scheduler
        kwargs = {"stage_name": "reasoner"}

    def fail(*args, **kwargs):
        raise RuntimeError("adapter initialization failed")

    with monkeypatch.context() as patch:
        patch.setattr(owner, name, fail)
        with pytest.raises(RuntimeError, match="adapter initialization failed"):
            factory(str(snapshot), runtime_gpu_ids=[0, 1], **kwargs)
    assert native.shutdown == [kind]
    monkeypatch.setattr(owner, name, lambda *args, **kwargs: args[0])
    runtime = factory(str(snapshot), runtime_gpu_ids=[0, 1], **kwargs)
    runtime.shutdown()
    assert native.shutdown == [kind, kind]


@pytest.mark.parametrize("stage_override", [False, True])
def test_http_and_sdk_share_one_snapshot_and_native_owner(
    native, snapshot, tmp_path, monkeypatch, stage_override
):
    resolutions = []

    def resolve(value):
        resolutions.append(value)
        assert len(resolutions) == 1, "Hub revision was resolved a second time"
        return str(snapshot)

    monkeypatch.setattr(checkpoint, "resolve_checkpoint", resolve)
    config = Cosmos3PipelineConfig(model_path="nvidia/Cosmos3-Super@campaign")
    stage = config.stages[0]
    stage.runtime_gpu_ids = [0, 1, 2, 3]
    stage.factory.output_dir = str(tmp_path / "outputs")
    if stage_override:
        stage.factory.model_path = "nvidia/Cosmos3-Super@stage-pin"
        stage.factory.server_args_overrides = {"served_model_name": "super-local"}
    kwargs = resolve_stage_typed_kwargs(stage)
    kwargs.setdefault("model_path", config.model_path)
    scheduler = stages.create_generation_scheduler(
        **kwargs, runtime_gpu_ids=stage.runtime_gpu_ids
    )
    try:
        frontend = scheduler.http_app()
        worker = native.generation[0]
        assert frontend.server_args is worker
        assert worker.model_path == str(snapshot)
        assert worker.served_model_name == (
            "super-local" if stage_override else "nvidia/Cosmos3-Super"
        )
        assert worker.gpu_ids == [0, 1, 2, 3]
        assert resolutions == [
            "nvidia/Cosmos3-Super@stage-pin" if stage_override else config.model_path
        ]
        assert config.model_path == "nvidia/Cosmos3-Super@campaign"
    finally:
        scheduler.stop()


def test_chat_factory_resolves_pin_and_preserves_native_defaults(
    native, snapshot, monkeypatch
):
    monkeypatch.setattr(checkpoint, "resolve_checkpoint", lambda model: str(snapshot))
    monkeypatch.setattr(
        native_chat_scheduler, "NativeChatScheduler", lambda *args, **kwargs: args[0]
    )
    runtime = native_chat_scheduler.create_native_chat_scheduler(
        "nvidia/Cosmos3-Super@campaign", stage_name="reasoner", runtime_gpu_ids=[0, 1]
    )
    assert native.reasoner[0]["model_path"] == str(snapshot)
    assert native.reasoner[0]["served_model_name"] == "nvidia/Cosmos3-Super"
    assert "context_length" not in native.reasoner[0]
    assert "mem_fraction_static" not in native.reasoner[0]
    runtime.shutdown()
