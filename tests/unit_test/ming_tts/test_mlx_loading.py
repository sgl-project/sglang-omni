# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import JsonValue

from sglang_omni.models.ming_tts.mlx.config import (
    AcousticConfig,
    ModelConfig,
    TextConfig,
)
from sglang_omni.models.ming_tts.mlx.loading import (
    checkpoint_files,
    load_component_weights,
)


def text_config_dict() -> dict[str, JsonValue]:
    return dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        moe_intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        first_k_dense_replace=1,
        multi_gate=True,
        rope_scaling={"type": "3D", "factor": None, "mrope_section": [1, 1, 2]},
    )


def test_composite_config_accepts_tiny_a3b_structure_without_importing_mlx() -> None:
    config = ModelConfig.from_dict(
        dict(
            llm_config=text_config_dict(),
            ditar_config=dict(
                hidden_size=16, depth=2, num_heads=2, patch_size=2, history_patch_size=4
            ),
            aggregator_config=dict(hidden_size=16, depth=1, num_heads=2),
            audio_tokenizer_config={"enc_kwargs": {"latent_dim": 4}},
            architectures=["BailingMMNativeForConditionalGeneration"],
        )
    )
    assert config.llm_config.mrope_section == (1, 1, 2)
    assert (config.patch_size, config.history_patch_size, config.latent_dim) == (
        2,
        4,
        4,
    )


@pytest.mark.parametrize(
    "change",
    [
        {"model_type": "qwen2"},
        {"num_experts": 0},
        {"num_experts_per_tok": 5},
        {"use_qk_norm": True},
        {"use_sliding_window": True},
        {"score_function": "sigmoid"},
        {"hidden_act": "gelu"},
        {"router_dtype": "float32"},
        {"n_group": 2},
        {"moe_router_enable_expert_bias": True},
        {"moe_shared_expert_intermediate_size": 99},
        {"rope_scaling": {"type": "linear", "factor": 2}},
        {"rope_scaling": {"type": "3D", "factor": None}},
    ],
)
def test_reject_unsupported_text_variants(change: dict[str, JsonValue]) -> None:
    with pytest.raises(ValueError):
        TextConfig.from_dict({**text_config_dict(), **change})


def test_official_mrope_sections() -> None:
    config = TextConfig.from_dict(
        {
            **text_config_dict(),
            "head_dim": 128,
            "rope_scaling": {"type": "3D", "factor": None},
        }
    )
    assert config.mrope_section == (16, 24, 24)


@pytest.mark.parametrize(
    "change", [{"qk_norm": "rms_norm"}, {"pe_attn_head": 1}, {"spk_dim": 192}]
)
def test_reject_unsupported_acoustic_variants(change: dict[str, JsonValue]) -> None:
    with pytest.raises(ValueError):
        AcousticConfig.from_dict(dict(hidden_size=16, depth=1, num_heads=2, **change))


def test_checkpoint_single_file(tmp_path: Path) -> None:
    weights = tmp_path / "model.safetensors"
    weights.touch()
    assert checkpoint_files(tmp_path) == [weights]


def test_checkpoint_index_deduplicates_shards(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "part-1.safetensors",
                    "b": "part-1.safetensors",
                    "c": "part-2.safetensors",
                }
            }
        )
    )
    assert checkpoint_files(tmp_path) == [
        tmp_path / "part-1.safetensors",
        tmp_path / "part-2.safetensors",
    ]


def test_checkpoint_index_rejects_escape(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "../outside.safetensors"}})
    )
    with pytest.raises(ValueError, match="within the model directory"):
        checkpoint_files(tmp_path)


def test_checkpoint_index_allows_huggingface_snapshot_symlinks(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    blob = tmp_path / "blob"
    blob.touch()
    shard = snapshot / "part.safetensors"
    shard.symlink_to(blob)
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "part.safetensors"}})
    )
    assert checkpoint_files(snapshot) == [shard]


def test_missing_checkpoint_is_not_silently_random(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        checkpoint_files(tmp_path)


@pytest.mark.parametrize(
    "component,expected",
    [
        ("ar", {"model.model.norm.weight": "norm", "stop_head.weight": "head"}),
        ("audio", {"encoder.fc1.weight": "encoder", "decoder.fc1.weight": "decoder"}),
    ],
)
def test_weight_ownership_before_tensor_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    component: str,
    expected: dict[str, str],
) -> None:
    parent = ModuleType("mlx")
    core = ModuleType("mlx.core")
    weights = {
        "model.model.norm.weight": "norm",
        "stop_head.weight": "head",
        "audio.encoder.fc1.weight": "encoder",
        "audio.decoder.fc1.weight": "decoder",
    }

    def load(path: str) -> dict[str, str]:
        assert path.endswith("model.safetensors")
        return weights

    core.load = load
    parent.core = core
    monkeypatch.setitem(sys.modules, "mlx", parent)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    (tmp_path / "model.safetensors").touch()
    assert load_component_weights(tmp_path, component=component) == expected


def test_duplicate_shard_keys_are_not_silently_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"}}
        )
    )
    parent = ModuleType("mlx")
    core = ModuleType("mlx.core")
    core.load = lambda path: {"stop_head.weight": object()}
    parent.core = core
    monkeypatch.setitem(sys.modules, "mlx", parent)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    with pytest.raises(ValueError, match="Duplicate checkpoint tensor"):
        load_component_weights(tmp_path, component="ar")
