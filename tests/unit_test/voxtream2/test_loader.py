# SPDX-License-Identifier: Apache-2.0
"""Local-asset validation, instance-only cache adaptation, and config wiring."""

import json
import subprocess
import sys
from unittest.mock import Mock

import pytest
import torch
import yaml

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.voxtream2 import stages
from sglang_omni.models.voxtream2.assets import resolve_assets
from sglang_omni.models.voxtream2.config import VoXtream2PipelineConfig


def config_values(tmp_path):
    return dict(
        model_path=str(tmp_path),
        reference_audio=str(tmp_path / "reference.wav"),
        mimi_checkpoint=str(tmp_path / "mimi.safetensors"),
        speaker_checkpoint=str(tmp_path / "speaker.pt"),
        generator_config_path=str(tmp_path / "generator.json"),
        speaking_rate_config_path=str(tmp_path / "rate.json"),
    )


def test_registered_config_file_and_factory_parameters(tmp_path):
    values = config_values(tmp_path)
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        yaml.safe_dump(dict(config_cls="VoXtream2PipelineConfig", **values))
    )
    config = ConfigManager.from_file(str(path)).config
    assert isinstance(config, VoXtream2PipelineConfig)
    assert config.entry_stage == "tts_engine"
    assert config.stages[0].terminal
    assert config.stages[0].gpu == 0
    assert (
        config.stage_factory_kwargs("tts_engine")["reference_audio"]
        == values["reference_audio"]
    )
    assert config.resolve_custom_voice_config().speakers == ("reference",)
    restored = VoXtream2PipelineConfig(**config.model_dump())
    assert restored.model_dump() == config.model_dump()


@pytest.mark.parametrize(
    "field,value", [("dtype", "int8"), ("max_pending", 0), ("output_buffer_chunks", 0)]
)
def test_config_rejects_invalid_limits(tmp_path, field, value):
    with pytest.raises(ValueError):
        VoXtream2PipelineConfig(**config_values(tmp_path), **{field: value})


def loader_values(tmp_path):
    data = {
        "model_config_name": "config.json",
        "phoneme_dict_name": "phones.json",
        "model_name": "model.safetensors",
    }
    (tmp_path / "generator.json").write_text(json.dumps(data))
    (tmp_path / "rate.json").write_text("{}")
    values = config_values(tmp_path)
    del values["reference_audio"]
    return dict(**values, device="cpu", dtype="float32", use_cuda_graph=False)


def test_missing_asset_fails_before_importing_or_allocating_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="model config"):
        resolve_assets(**loader_values(tmp_path))


@pytest.mark.parametrize(
    "changes",
    [
        {"dtype": "int8"},
        {"dtype": "float16"},
        {"use_cuda_graph": True},
        {"device": "meta"},
    ],
)
def test_loader_rejects_unsupported_device_configuration(tmp_path, changes):
    values = loader_values(tmp_path)
    values.update(changes)
    with pytest.raises(ValueError):
        resolve_assets(**values)


def test_cache_adaptation_is_instance_local_and_idempotent():
    # Optional model dependency: pure framework test installations can skip.
    cache_module = pytest.importorskip("sglang_omni.models.voxtream2.cache")
    KVCache = cache_module.KVCache

    original_update, original_reset = KVCache.update, KVCache.reset

    def make_cache():
        return KVCache(
            batch_size=1, max_seq_len=8, num_heads=1, head_dim=2, dtype=torch.float32
        )

    model = torch.nn.ModuleList([make_cache(), make_cache()])
    unrelated = make_cache()
    cache_module.bind_graph_safe_caches(model)
    bound_update = model[0].update
    cache_module.bind_graph_safe_caches(model)
    assert model[0].update is bound_update
    assert KVCache.update is original_update and KVCache.reset is original_reset
    assert unrelated.update.__func__ is original_update
    keys = torch.ones((1, 1, 2, 2))
    model[0].update(keys, keys * 2)
    assert model[0].size == 2
    assert model[1].size == unrelated.size == 0
    model[0].reset()
    assert model[0].size == 0
    assert not model[0].k_cache.any()


def test_stage_factory_passes_device_and_local_paths(tmp_path, monkeypatch):
    values = config_values(tmp_path)
    backend = Mock(sample_rate=24000, max_samples=480000)
    factory = Mock(return_value=backend)
    monkeypatch.setattr(stages, "RemoteVoXtreamBackend", factory)
    scheduler = stages.create_executor(
        **values, gpu_id=1, max_pending=2, output_buffer_chunks=3
    )
    options = factory.call_args.kwargs
    assert options["python_executable"] == sys.executable
    assert options["model_config"]["device"] == "cuda:1"
    assert options["model_config"]["model_path"] == str(tmp_path)
    assert scheduler.backend is backend
    assert scheduler.inbox.maxsize == 2 and scheduler.outbox.capacity == 3


def test_stage_factory_selects_isolated_worker_without_loading_model(
    tmp_path, monkeypatch
):

    values = config_values(tmp_path)
    backend = Mock(sample_rate=24000, max_samples=480000)
    factory = Mock(return_value=backend)
    monkeypatch.setattr(stages, "RemoteVoXtreamBackend", factory)
    scheduler = stages.create_executor(
        **values,
        gpu_id=1,
        model_python="/model/bin/python",
        model_source_paths=["/redimnet"],
        worker_cancel_timeout=7,
    )
    assert scheduler.backend is backend
    options = factory.call_args.kwargs
    assert options["python_executable"] == "/model/bin/python"
    assert options["source_paths"] == ["/redimnet"]
    assert options["cancel_timeout"] == 7
    assert options["model_config"]["device"] == "cuda:1"
    assert options["model_config"]["reference_audio"] == values["reference_audio"]


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_worker_timeout_config_is_finite_and_positive(tmp_path, value):
    with pytest.raises(ValueError):
        VoXtream2PipelineConfig(
            **config_values(tmp_path), worker_response_timeout=value
        )


def test_stage_import_does_not_import_model_backend() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import sglang_omni.models.voxtream2.stages; "
            "assert 'sglang_omni.models.voxtream2.backend' not in sys.modules; "
            "assert 'sglang_omni.models.voxtream2.loader' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("invalid_config", [[], None, {"model_config_name": 12}])
def test_asset_config_rejects_malformed_objects(tmp_path, invalid_config):
    values = loader_values(tmp_path)
    (tmp_path / "generator.json").write_text(json.dumps(invalid_config))
    with pytest.raises(ValueError):
        resolve_assets(**values)
