# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pytest

from sglang_omni.models.ming_tts.mlx.loading import (
    checkpoint_files,
    load_component_weights,
    read_config,
)


def test_checkpoint_single_file(tmp_path: Path) -> None:
    weights = tmp_path / "model.safetensors"
    weights.touch()
    assert checkpoint_files(tmp_path) == [weights]


def test_checkpoint_index_deduplicates_shards(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"a": "part-1.safetensors", "b": "part-1.safetensors", "c": "part-2.safetensors"}
    }))
    assert checkpoint_files(tmp_path) == [tmp_path / "part-1.safetensors", tmp_path / "part-2.safetensors"]


def test_checkpoint_index_rejects_escape(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"a": "../outside.safetensors"}
    }))
    with pytest.raises(ValueError, match="within the model directory"):
        checkpoint_files(tmp_path)


def test_checkpoint_index_allows_huggingface_snapshot_symlinks(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    blob = tmp_path / "blob"
    blob.touch()
    shard = snapshot / "part.safetensors"
    shard.symlink_to(blob)
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"weight": "part.safetensors"}
    }))
    assert checkpoint_files(snapshot) == [shard]


def test_missing_checkpoint_is_not_silently_random(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        checkpoint_files(tmp_path)


def test_read_nested_config(tmp_path: Path) -> None:
    expected = {"llm_config": {"model_type": "bailing_moe"}, "audio_tokenizer_config": {}}
    (tmp_path / "config.json").write_text(json.dumps(expected))
    assert read_config(tmp_path) == expected


@pytest.mark.parametrize("component,expected", [
    ("ar", {"model.model.norm.weight": "norm", "stop_head.weight": "head"}),
    ("audio", {"encoder.fc1.weight": "encoder", "decoder.fc1.weight": "decoder"}),
])
def test_weight_ownership_before_tensor_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    component: str, expected: dict[str, str],
) -> None:
    parent = ModuleType("mlx")
    core = ModuleType("mlx.core")
    weights = {
        "model.model.norm.weight": "norm", "stop_head.weight": "head",
        "audio.encoder.fc1.weight": "encoder", "audio.decoder.fc1.weight": "decoder",
    }

    def load(path: str) -> dict[str, Any]:
        assert path.endswith("model.safetensors")
        return weights

    core.load = load
    parent.core = core
    monkeypatch.setitem(sys.modules, "mlx", parent)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    (tmp_path / "model.safetensors").touch()
    assert load_component_weights(tmp_path, component=component) == expected


def test_duplicate_shard_keys_are_not_silently_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"}
    }))
    parent = ModuleType("mlx")
    core = ModuleType("mlx.core")
    core.load = lambda path: {"stop_head.weight": object()}
    parent.core = core
    monkeypatch.setitem(sys.modules, "mlx", parent)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    with pytest.raises(ValueError, match="Duplicate checkpoint tensor"):
        load_component_weights(tmp_path, component="ar")
