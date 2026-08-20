# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from sglang_omni.models.kimi_audio.checkpoint import (
    _is_text_output_weight,
    _make_filtered_view,
    resolve_kimi_audio_text_checkpoint,
)


def test_text_checkpoint_keeps_only_executed_main_layers_and_text_branch() -> None:
    assert _is_text_output_weight("model.embed_tokens.weight")
    assert _is_text_output_weight("model.layers.21.self_attn.q_proj.weight")
    assert _is_text_output_weight("model.layers.27.self_attn.q_proj.weight")
    assert not _is_text_output_weight("model.mimo_layers.5.mlp.down_proj.weight")
    assert not _is_text_output_weight("model.mimo_norm.weight")
    assert _is_text_output_weight("model.vq_adaptor.layers.0.weight")
    assert not _is_text_output_weight("mimo_output.weight")
    assert _is_text_output_weight("lm_head.weight")
    assert _is_text_output_weight("model.norm.weight")


def test_filtered_view_removes_remote_config_mapping(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "moonshot_kimia", "auto_map": {"AutoConfig": "x"}})
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"auto_map": {"AutoTokenizer": "y"}})
    )
    view = _make_filtered_view(
        tmp_path,
        {
            "metadata": {},
            "weight_map": {"model.embed_tokens.weight": "model.safetensors"},
        },
    )

    filtered_config = json.loads((Path(view) / "config.json").read_text())
    assert "auto_map" not in filtered_config
    assert filtered_config["model_type"] == "moonshot_kimia"
    tokenizer_config = json.loads((Path(view) / "tokenizer_config.json").read_text())
    assert "auto_map" in tokenizer_config


def test_local_checkpoint_is_also_exposed_as_a_filtered_view(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"auto_map": {"AutoConfig": "x"}}))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "model.embed_tokens.weight": "kept.safetensors",
                    "model.mimo_layers.0.mlp.down_proj.weight": "unused.safetensors",
                },
            }
        )
    )
    (tmp_path / "kept.safetensors").touch()
    (tmp_path / "unused.safetensors").touch()

    view = Path(resolve_kimi_audio_text_checkpoint(str(tmp_path)))
    index = json.loads((view / "model.safetensors.index.json").read_text())

    assert index["weight_map"] == {"model.embed_tokens.weight": "kept.safetensors"}
    assert (view / "kept.safetensors").is_symlink()
    assert (view / "config.json").is_file()
