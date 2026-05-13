# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sglang_omni_v1.model_runner.checkpoint_filter import (
    CheckpointFilterConfig,
    filter_and_remap_weight_iterator,
    filter_safetensor_files,
    install_checkpoint_filter,
)
from sglang_omni_v1.models.qwen3_omni.bootstrap import (
    QWEN_TALKER_CHECKPOINT_FILTER,
    QWEN_THINKER_CHECKPOINT_FILTER,
)


def _write_index(model_dir: Path, weight_map: dict[str, str]) -> None:
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )


def test_index_filter_selects_matching_shards_and_preserves_file_order(
    tmp_path: Path,
) -> None:
    _write_index(
        tmp_path,
        {
            "thinker.model.layers.0.weight": "a.safetensors",
            "talker.model.layers.0.weight": "b.safetensors",
            "code2wav.decoder.weight": "c.safetensors",
        },
    )
    files = [
        str(tmp_path / "c.safetensors"),
        str(tmp_path / "b.safetensors"),
        str(tmp_path / "a.safetensors"),
    ]
    profile = CheckpointFilterConfig(
        name="talker",
        accept_prefixes=("talker.",),
    )

    selected = filter_safetensor_files(
        model_dir=tmp_path,
        files=files,
        profile=profile,
    )

    assert selected == [str(tmp_path / "b.safetensors")]


def test_index_filter_rejects_missing_required_prefix(tmp_path: Path) -> None:
    _write_index(tmp_path, {"thinker.model.embed_tokens.weight": "a.safetensors"})
    profile = CheckpointFilterConfig(name="talker", accept_prefixes=("talker.",))

    with pytest.raises(ValueError, match="matched no tensors"):
        filter_safetensor_files(
            model_dir=tmp_path,
            files=[str(tmp_path / "a.safetensors")],
            profile=profile,
        )


def test_index_filter_rejects_unavailable_matching_shard(tmp_path: Path) -> None:
    _write_index(tmp_path, {"talker.model.embed_tokens.weight": "missing.safetensors"})
    profile = CheckpointFilterConfig(name="talker", accept_prefixes=("talker.",))

    with pytest.raises(ValueError, match="shard files are available"):
        filter_safetensor_files(
            model_dir=tmp_path,
            files=[str(tmp_path / "other.safetensors")],
            profile=profile,
        )


def test_missing_index_selects_matching_safetensors_by_header(tmp_path: Path) -> None:
    import numpy as np
    from safetensors.numpy import save_file

    thinker_file = tmp_path / "thinker.safetensors"
    talker_file = tmp_path / "talker.safetensors"
    save_file(
        {"thinker.model.embed_tokens.weight": np.zeros((1,), dtype=np.float32)},
        thinker_file,
    )
    save_file(
        {"talker.model.embed_tokens.weight": np.zeros((1,), dtype=np.float32)},
        talker_file,
    )
    files = [str(thinker_file), str(talker_file)]
    profile = CheckpointFilterConfig(name="talker", accept_prefixes=("talker.",))

    selected = filter_safetensor_files(
        model_dir=tmp_path,
        files=files,
        profile=profile,
    )

    assert selected == [str(talker_file)]


def test_weight_iterator_filters_and_remaps_talker_prefix() -> None:
    weights = [
        ("thinker.model.weight", object()),
        ("talker.layers.0.weight", object()),
        ("code2wav.decoder.weight", object()),
    ]

    filtered = list(
        filter_and_remap_weight_iterator(weights, QWEN_TALKER_CHECKPOINT_FILTER)
    )

    assert [name for name, _ in filtered] == ["layers.0.weight"]


def test_weight_iterator_preserves_and_remaps_thinker_text_names() -> None:
    weights = [
        ("thinker.model.layers.0.weight", object()),
        ("model.layers.1.weight", object()),
        ("lm_head.weight", object()),
        ("thinker.audio_tower.layers.0.weight", object()),
        ("talker.layers.0.weight", object()),
    ]

    filtered = list(
        filter_and_remap_weight_iterator(weights, QWEN_THINKER_CHECKPOINT_FILTER)
    )

    assert [name for name, _ in filtered] == [
        "model.layers.0.weight",
        "model.layers.1.weight",
        "lm_head.weight",
    ]


def test_loader_instance_filter_applies_shard_and_iterator_rules(
    tmp_path: Path,
) -> None:
    class FakeLoader:
        def _prepare_weights(self, model_name_or_path, revision, fall_back_to_pt):
            return (
                tmp_path,
                [
                    str(tmp_path / "thinker.safetensors"),
                    str(tmp_path / "talker.safetensors"),
                ],
                True,
            )

        def _get_weights_iterator(self, source):
            return iter(
                [
                    ("thinker.model.layers.0.weight", object()),
                    ("talker.layers.0.weight", object()),
                ]
            )

    _write_index(
        tmp_path,
        {
            "thinker.model.layers.0.weight": "thinker.safetensors",
            "talker.layers.0.weight": "talker.safetensors",
        },
    )
    loader = FakeLoader()

    install_checkpoint_filter(loader, QWEN_TALKER_CHECKPOINT_FILTER)

    _, files, use_safetensors = loader._prepare_weights("model", None, True)
    assert files == [str(tmp_path / "talker.safetensors")]
    assert use_safetensors is True
    assert [name for name, _ in loader._get_weights_iterator(None)] == [
        "layers.0.weight"
    ]
