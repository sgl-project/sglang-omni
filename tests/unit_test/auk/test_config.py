# SPDX-License-Identifier: Apache-2.0
"""Pipeline config and registry wiring for AuK."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import huggingface_hub
import pytest

from sglang_omni.config import manager
from sglang_omni.config.path import ConfigPath
from sglang_omni.config.schema import ProcessConfig
from sglang_omni.models.auk.config import (
    CONDITIONING_STAGE,
    DECODE_STAGE,
    ENGINE_STAGE,
    PREPROCESSING_STAGE,
    AuKPipelineConfig,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY


@pytest.mark.parametrize(
    "architecture", ["AuKForConditionalGeneration", "AuK", "AuK-Flash"]
)
def test_registered_pipeline_is_connected(architecture):
    config_cls = PIPELINE_CONFIG_REGISTRY.get_config(architecture)
    assert config_cls is AuKPipelineConfig
    config = config_cls(model_path="tencent/AuK")
    preprocessing, conditioning, engine, decode = config.stages
    assert config.resolved_entry_stage == preprocessing.name == PREPROCESSING_STAGE
    assert preprocessing.next == conditioning.name == CONDITIONING_STAGE
    assert conditioning.next == engine.name == ENGINE_STAGE
    assert engine.next == decode.name == DECODE_STAGE
    assert engine.factory.enable_dit_fused_qk_norm_rope is True
    assert config.terminal_stages == [decode.name]
    for stage in config.stages:
        module, name = stage.factory_path.rsplit(".", 1)
        assert callable(getattr(importlib.import_module(module), name))


@pytest.mark.parametrize("name", ["AuK", "AuK-Flash"])
def test_local_checkpoint_yaml_resolves_pipeline(tmp_path, name):
    from sglang_omni.models.auk.hf_config import load_auk_config

    (tmp_path / "config.yaml").write_text(
        f"model:\n  name: {name}\n"
        "  vae:\n    latent_dim: 64\n    model_init_kwargs:\n"
        "      latent_dim: ${model.vae.latent_dim}\n"
    )
    assert manager.resolve_config_cls_for_model_path(str(tmp_path)) is AuKPipelineConfig
    assert load_auk_config(str(tmp_path)).vae_init_kwargs["latent_dim"] == 64


def test_unrelated_omegaconf_yaml_is_not_auk(tmp_path):
    (tmp_path / "config.yaml").write_text("model:\n  name: OtherModel\n")
    with pytest.raises(ValueError, match="Could not resolve model architecture"):
        manager.resolve_config_cls_for_model_path(str(tmp_path))


def test_local_weight_marker_resolves_without_yaml(tmp_path):
    (tmp_path / "auk_base.safetensors").write_bytes(b"")
    assert manager.resolve_config_cls_for_model_path(str(tmp_path)) is AuKPipelineConfig


def test_hub_config_yaml_resolves_without_snapshot(monkeypatch, tmp_path):
    def fail_auto_config(*args, **kwargs):
        raise OSError("no config.json")

    def fail_snapshot(*args, **kwargs):
        raise AssertionError("discovery must not download weights")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail_snapshot)
    monkeypatch.setattr(
        manager, "AutoConfig", SimpleNamespace(from_pretrained=fail_auto_config)
    )

    path = tmp_path / "config.yaml"
    path.write_text("model:\n  name: AuK\n")

    def fake_hub_download(repo_id, filename, **kwargs):
        assert repo_id == "tencent/AuK"
        assert filename == "config.yaml"
        return str(path)

    monkeypatch.setattr("sglang_omni.utils.hf.hf_hub_download", fake_hub_download)
    assert manager.resolve_config_cls_for_model_path("tencent/AuK") is AuKPipelineConfig


@pytest.mark.parametrize(
    "invalid",
    ["gpu", "process", "device", "replica", "concurrency", "order", "wait_for"],
)
def test_offload_rejects_incompatible_placement(invalid: str) -> None:
    """Offload must reject nonserial placement and cyclic fan-in dependencies."""
    config = AuKPipelineConfig(model_path="test")
    for stage in config.stages[1:]:
        stage.factory.device = "cuda"
    if invalid == "gpu":
        config.stage_named(ENGINE_STAGE).gpu = 1
    elif invalid == "process":
        config.stage_named(ENGINE_STAGE).process = "other"
    elif invalid == "device":
        config.stage_named(ENGINE_STAGE).factory.device = "cpu"
    elif invalid == "replica":
        config.processes["pipeline"] = ProcessConfig(num_replicas=2)
    elif invalid == "concurrency":
        config.stage_named(ENGINE_STAGE).factory.max_concurrency = 2
    elif invalid == "wait_for":
        config.stage_named(ENGINE_STAGE).wait_for = [CONDITIONING_STAGE, DECODE_STAGE]
        config.stage_named(ENGINE_STAGE).merge_fn = "test.merge"
    else:
        config.stages[1], config.stages[2] = config.stages[2], config.stages[1]
    with pytest.raises(ValueError, match="AuK offload requires"):
        config.with_serial_offload(frozenset({"conditioning", "dit", "vae"}))


def test_resident_supports_custom_stage_names() -> None:
    """Renaming resident stages must not trigger offload validation or environment setup."""
    environment = {"OMP_NUM_THREADS": "2"}
    configuration = AuKPipelineConfig(
        model_path="test", env_defaults=environment
    ).model_dump()
    for stage in configuration["stages"]:
        stage["name"] = f"custom_{stage['name']}"
        if stage["next"] is not None:
            stage["next"] = f"custom_{stage['next']}"
        else:
            pass
    pipeline = AuKPipelineConfig(**configuration)
    assert pipeline.resolved_env_defaults() == environment


@pytest.mark.parametrize(
    "enabled_stage", [CONDITIONING_STAGE, ENGINE_STAGE, DECODE_STAGE]
)
def test_offload_requires_all_gpu_stages(enabled_stage: str) -> None:
    """Partially enabled offload must fail validation rather than use resident settings."""
    configuration = AuKPipelineConfig(model_path="test")
    configuration.stage_named(enabled_stage).factory.enable_serial_offload = True
    with pytest.raises(ValueError, match="must be enabled on .* together"):
        AuKPipelineConfig.model_validate(
            configuration.model_dump(serialize_as_any=True)
        )


def test_capture_shapes_cli_text_becomes_a_list_of_shapes() -> None:
    """A dotted CLI value arrives as text; a raw string is not a shape."""
    path = ConfigPath.parse(
        "stages.auk_engine.factory.dit_cuda_graph_capture_shapes",
        AuKPipelineConfig,
    )
    assert path.coerce("[[1,192,0,192],[1,320,320,384]]") == [
        [1, 192, 0, 192],
        [1, 320, 320, 384],
    ]
