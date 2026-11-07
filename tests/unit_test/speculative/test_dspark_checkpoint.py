# SPDX-License-Identifier: Apache-2.0
"""DSpark checkpoint metadata and temporary checkpoint lifetime."""

import json
from pathlib import Path

import pytest
from transformers import Qwen3Config

from sglang_omni.speculative.dspark_checkpoint import materialize_draft_checkpoint


def test_ming_draft_preserves_anchor_alignment_and_trained_heads(
    tmp_path: Path,
) -> None:
    """An eight-position anchor-sampling draft needs eight proposals, not seven."""
    source_config = {
        "speculators_model_type": "dspark",
        "sample_from_anchor": True,
        "sliding_window_non_causal": False,
        "dtype": "bfloat16",
        "draft_vocab_size": 48000,
        "aux_hidden_state_layer_ids": [2, 7, 14, 18, 21],
        "mask_token_id": 156892,
        "block_size": 8,
        "markov_rank": 256,
        "markov_head_type": "vanilla",
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "transformer_layer_config": {
            "hidden_size": 1536,
            "vocab_size": 157184,
            "num_hidden_layers": 5,
            "num_attention_heads": 12,
            "num_key_value_heads": 12,
            "head_dim": 128,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(source_config))
    weight_names = (
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )
    for name in weight_names:
        (tmp_path / name).touch()

    with materialize_draft_checkpoint(str(tmp_path), None) as checkpoint:
        config = Qwen3Config.from_pretrained(checkpoint)
        assert config.architectures == ["ReducedVocabularyDSparkDraftModel"]
        assert config.dflash_config == {
            "target_layer_ids": [1, 6, 13, 17, 20],
            "mask_token_id": 156892,
            "block_size": 8,
        }
        assert config.sample_from_anchor is True
        assert (config.vocab_size, config.draft_vocab_size) == (157184, 48000)
        assert (config.markov_head_type, config.markov_rank) == ("vanilla", 256)
        assert config.enable_confidence_head is True
        assert config.confidence_head_with_markov is True
        assert config.rope_parameters["rope_theta"] == 10000.0
        for name in weight_names:
            assert (Path(checkpoint) / name).resolve() == tmp_path / name

    assert not Path(checkpoint).exists()
    assert all((tmp_path / name).exists() for name in weight_names)


def test_dflash_checkpoint_is_rejected_before_creating_dspark_config(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"speculators_model_type": "dflash"})
    )
    with pytest.raises(ValueError, match="not a speculators DSpark"):
        materialize_draft_checkpoint(str(tmp_path), None)
