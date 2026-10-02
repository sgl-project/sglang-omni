# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.easymagpie_tts.hf_config import (
    EasyMagpieTTSConfig,
    adapt_backbone_config,
    partition_weights,
)


def test_special_audio_ids_follow_the_codebook_size(tiny_raw_config) -> None:
    raw = dict(tiny_raw_config, num_audio_codebooks=8, codebook_size=1024)
    config = EasyMagpieTTSConfig.from_dict(raw)
    assert config.num_stacked_codebooks == 16
    assert config.codebook_vocab_size == 1032
    assert (config.audio_bos_id, config.audio_eos_id) == (1024, 1025)


def test_forced_audio_ids_override_the_offsets(tiny_raw_config) -> None:
    raw = dict(tiny_raw_config, forced_audio_bos_id=3, forced_audio_eos_id=0)
    config = EasyMagpieTTSConfig.from_dict(raw)
    assert (config.audio_bos_id, config.audio_eos_id) == (3, 0)


def test_config_rejects_missing_keys_and_mismatched_widths(tiny_raw_config) -> None:
    raw = dict(tiny_raw_config)
    raw.pop("codebook_size")
    with pytest.raises(ValueError, match="codebook_size"):
        EasyMagpieTTSConfig.from_dict(raw)
    with pytest.raises(ValueError, match="embedding widths"):
        EasyMagpieTTSConfig.from_dict(dict(tiny_raw_config, audio_embedding_dim=4))


def test_backbone_config_legacy_mamba_keys_override_defaults() -> None:
    adapted = adapt_backbone_config(
        {"n_groups": 4, "conv_kernel": 2, "use_conv_bias": True, "mamba_d_conv": 4}
    )
    assert adapted["mamba_n_groups"] == 4
    assert adapted["mamba_conv_bias"] is True
    assert adapted["mamba_d_conv"] == 2
    assert adapted["n_shared_experts"] == 1


def test_sglang_nemotron_h_config_reads_the_adapted_keys(tiny_raw_config) -> None:
    from sglang.srt.configs.nemotron_h import NemotronHConfig

    raw = {
        **tiny_raw_config,
        "vocab_size": 2,
        "hybrid_override_pattern": "M*E",
        "n_groups": 4,
        "conv_kernel": 2,
        "use_conv_bias": False,
    }
    loaded = NemotronHConfig(**raw)
    config = NemotronHConfig(**adapt_backbone_config(loaded.to_dict()))
    assert (config.mamba_n_groups, config.conv_kernel) == (4, 2)
    assert config.use_conv_bias is False
    assert config.eos_token_id == 1
    assert EasyMagpieTTSConfig.from_dict(config.to_dict()).num_stacked_codebooks == 4


@pytest.mark.parametrize(
    "eos_token_id,expected", [(2, 1), (None, 1), (-1, 1), (0, 0), (1, 1)]
)
def test_backbone_eos_must_be_inside_the_dummy_vocabulary(
    eos_token_id: int | None, expected: int
) -> None:
    adapted = adapt_backbone_config({"vocab_size": 2, "eos_token_id": eos_token_id})
    assert adapted["eos_token_id"] == expected


def test_partition_keeps_head_embeddings_away_from_the_backbone_loader() -> None:
    conv = torch.zeros(4, 2, 1)
    head_weights: list[tuple[str, torch.Tensor]] = []
    backbone = list(
        partition_weights(
            [
                ("decoder.embeddings.weight", torch.zeros(1)),
                ("audio_embeddings.0.weight", torch.zeros(1)),
                ("audio_decoder.pre_conv.conv.weight", torch.zeros(1)),
                ("local_transformer.layers.0.pos_ff.proj.conv.weight", conv),
            ],
            head_weights,
        )
    )
    assert [name for name, _ in backbone] == ["backbone.embeddings.weight"]
    heads = dict(head_weights)
    assert set(heads) == {
        "audio_embeddings.0.weight",
        "local_transformer.layers.0.pos_ff.proj.conv.weight",
    }
    assert heads["local_transformer.layers.0.pos_ff.proj.conv.weight"].shape == (4, 2)


def test_partition_streams_backbone_weights_lazily() -> None:
    events = []

    def weights():
        for name in ("decoder.a", "phoneme_embeddings.0.weight", "decoder.b"):
            events.append(("yielded", name))
            yield name, torch.zeros(1)

    head_weights: list[tuple[str, torch.Tensor]] = []
    for name, _ in partition_weights(weights(), head_weights):
        events.append(("loaded", name))
    assert events == [
        ("yielded", "decoder.a"),
        ("loaded", "backbone.a"),
        ("yielded", "phoneme_embeddings.0.weight"),
        ("yielded", "decoder.b"),
        ("loaded", "backbone.b"),
    ]
    assert [name for name, _ in head_weights] == ["phoneme_embeddings.0.weight"]
