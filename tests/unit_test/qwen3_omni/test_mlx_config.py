# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the native MLX Qwen3-Omni configuration parser."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from sglang_omni.models.qwen3_omni.mlx.config import (
    QuantizationConfig,
    Qwen3OmniMlxConfig,
)
from tests.utils.build_tiny_qwen3_omni_checkpoint import build_tiny_config


def tiny_root_config_dict() -> dict[str, Any]:
    """Return the complete nested Omni config with production IDs/vocabularies."""

    return build_tiny_config().to_dict()


def _to_published_legacy_rope(text_cfg: dict[str, Any]) -> None:
    """Rewrite a text config's normalized ``rope_parameters`` in place into the"""

    rope = text_cfg.pop("rope_parameters")
    text_cfg["rope_theta"] = rope["rope_theta"]
    scaling: dict[str, Any] = {"rope_type": rope.get("rope_type", "default")}
    if "mrope_section" in rope:
        scaling["mrope_section"] = rope["mrope_section"]
    text_cfg["rope_scaling"] = scaling


def test_mlx_config_reads_nested_thinker_and_talker_sections() -> None:
    config = Qwen3OmniMlxConfig.from_dict(tiny_root_config_dict())
    assert config.thinker.text_config.num_experts == 4
    assert config.thinker.text_config.num_experts_per_tok == 2
    assert config.talker.text_config.num_experts == 4
    assert config.talker.code_predictor_config.num_code_groups == 16


def test_mlx_config_parses_vision_audio_and_code2wav() -> None:
    config = Qwen3OmniMlxConfig.from_dict(tiny_root_config_dict())

    assert config.vision.spatial_merge_size == 2
    assert config.vision.deepstack_visual_indexes == (0, 1)
    assert config.audio.num_mel_bins == 128
    assert config.audio.output_dim == config.thinker.text_config.hidden_size
    assert config.code2wav.num_quantizers == config.talker.num_code_groups
    assert config.code2wav.upsampling_ratios
    assert config.code2wav.upsample_rates


def test_mlx_config_uses_published_vision_position_default() -> None:
    raw = copy.deepcopy(tiny_root_config_dict())
    raw["thinker_config"]["vision_config"].pop("num_position_embeddings")

    config = Qwen3OmniMlxConfig.from_dict(raw)

    assert config.vision.num_position_embeddings == 2304


def test_mlx_config_rejects_code2wav_quantizer_mismatch() -> None:
    raw = tiny_root_config_dict()
    raw["code2wav_config"]["num_quantizers"] = 8

    with pytest.raises(ValueError, match="num_quantizers"):
        Qwen3OmniMlxConfig.from_dict(raw)


def test_mlx_config_normalizes_talker_num_local_experts_alias() -> None:
    raw = tiny_root_config_dict()
    # Transformers stores the talker expert count under the ``num_local_experts``
    # alias (attribute_map) while the thinker uses ``num_experts`` directly.
    assert "num_local_experts" in raw["talker_config"]["text_config"]
    assert "num_experts" not in raw["talker_config"]["text_config"]

    config = Qwen3OmniMlxConfig.from_dict(raw)
    assert config.talker.text_config.num_experts == 4


def test_mlx_config_reads_mrope_section_and_theta() -> None:
    config = Qwen3OmniMlxConfig.from_dict(tiny_root_config_dict())
    assert config.thinker.text_config.mrope_section == (2, 1, 1)
    assert config.thinker.text_config.rope_theta == 1000000.0
    assert config.talker.text_config.mrope_section == (2, 1, 1)


def test_mlx_config_reads_talker_shared_expert_intermediate_size() -> None:
    config = Qwen3OmniMlxConfig.from_dict(tiny_root_config_dict())
    assert config.talker.text_config.shared_expert_intermediate_size == 64
    # The thinker MoE has no shared expert branch.
    assert config.thinker.text_config.shared_expert_intermediate_size in (None, 0)


def test_mlx_config_missing_field_raises_with_dotted_path() -> None:
    raw = copy.deepcopy(tiny_root_config_dict())
    # Remove both the alias and the canonical name so the field is truly absent.
    raw["talker_config"]["text_config"].pop("num_local_experts", None)
    raw["talker_config"]["text_config"].pop("num_experts", None)

    with pytest.raises(KeyError) as excinfo:
        Qwen3OmniMlxConfig.from_dict(raw)

    assert "talker.text_config.num_experts" in str(excinfo.value)


def test_mlx_config_missing_code_predictor_field_raises_with_dotted_path() -> None:
    raw = copy.deepcopy(tiny_root_config_dict())
    raw["talker_config"]["code_predictor_config"].pop("num_code_groups", None)

    with pytest.raises(KeyError) as excinfo:
        Qwen3OmniMlxConfig.from_dict(raw)

    assert "talker.code_predictor_config.num_code_groups" in str(excinfo.value)


def test_mlx_config_missing_thinker_section_raises_with_dotted_path() -> None:
    raw = copy.deepcopy(tiny_root_config_dict())
    raw.pop("thinker_config", None)

    with pytest.raises(KeyError) as excinfo:
        Qwen3OmniMlxConfig.from_dict(raw)

    assert "thinker_config" in str(excinfo.value)


def test_mlx_config_reads_published_legacy_rope_layout() -> None:
    # Published Qwen3-Omni checkpoints store rope as a ``rope_scaling`` dict
    # (rope_type + mrope_section) plus a sibling ``rope_theta``. Direct JSON
    raw = copy.deepcopy(tiny_root_config_dict())
    _to_published_legacy_rope(raw["thinker_config"]["text_config"])
    _to_published_legacy_rope(raw["talker_config"]["text_config"])
    _to_published_legacy_rope(raw["talker_config"]["code_predictor_config"])

    assert "rope_parameters" not in raw["thinker_config"]["text_config"]
    assert "rope_scaling" in raw["thinker_config"]["text_config"]
    assert raw["thinker_config"]["text_config"]["rope_theta"] == 1000000.0

    config = Qwen3OmniMlxConfig.from_dict(raw)
    assert config.thinker.text_config.mrope_section == (2, 1, 1)
    assert config.thinker.text_config.rope_theta == 1000000.0
    assert config.talker.text_config.mrope_section == (2, 1, 1)
    assert config.talker.text_config.rope_theta == 1000000.0
    assert config.talker.code_predictor_config.rope_theta == 1000000.0


def test_mlx_config_normalized_and_legacy_rope_agree() -> None:
    normalized = Qwen3OmniMlxConfig.from_dict(tiny_root_config_dict())

    raw = copy.deepcopy(tiny_root_config_dict())
    _to_published_legacy_rope(raw["thinker_config"]["text_config"])
    _to_published_legacy_rope(raw["talker_config"]["text_config"])
    _to_published_legacy_rope(raw["talker_config"]["code_predictor_config"])
    legacy = Qwen3OmniMlxConfig.from_dict(raw)

    assert (
        legacy.thinker.text_config.mrope_section
        == normalized.thinker.text_config.mrope_section
    )
    assert (
        legacy.thinker.text_config.rope_theta
        == normalized.thinker.text_config.rope_theta
    )


def test_mlx_config_missing_mrope_section_raises_with_dotted_path() -> None:
    # ``mrope_section`` must never silently become ``None``; a config missing it
    # under both rope layouts must raise naming the full dotted path.
    raw = copy.deepcopy(tiny_root_config_dict())
    raw["thinker_config"]["text_config"]["rope_parameters"].pop("mrope_section", None)

    with pytest.raises(KeyError) as excinfo:
        Qwen3OmniMlxConfig.from_dict(raw)

    assert "thinker.text_config.mrope_section" in str(excinfo.value)


def test_mlx_config_missing_rope_theta_raises_with_dotted_path() -> None:
    raw = copy.deepcopy(tiny_root_config_dict())
    _to_published_legacy_rope(raw["thinker_config"]["text_config"])
    raw["thinker_config"]["text_config"].pop("rope_theta", None)

    with pytest.raises(KeyError) as excinfo:
        Qwen3OmniMlxConfig.from_dict(raw)

    assert "thinker.text_config.rope_theta" in str(excinfo.value)


def test_quantization_config_from_dict_parses_bits_and_group_size() -> None:
    quant = QuantizationConfig.from_dict({"bits": 4, "group_size": 32})
    assert quant.bits == 4
    assert quant.group_size == 32
    assert quant.mode == "affine"


def test_quantization_config_missing_field_raises_with_dotted_path() -> None:
    with pytest.raises(KeyError) as excinfo:
        QuantizationConfig.from_dict({"bits": 4})
    assert "quantization.group_size" in str(excinfo.value)


def test_mlx_config_parses_optional_quantization_metadata() -> None:
    raw = copy.deepcopy(tiny_root_config_dict())
    raw["quantization"] = {"bits": 4, "group_size": 32}

    config = Qwen3OmniMlxConfig.from_dict(raw)
    assert config.quantization is not None
    assert config.quantization.bits == 4
    assert config.quantization.group_size == 32


def test_mlx_config_quantization_defaults_to_none_when_absent() -> None:
    config = Qwen3OmniMlxConfig.from_dict(tiny_root_config_dict())
    assert config.quantization is None


# ---------------------------------------------------------------------------
# Task 7 review follow-ups (authorized in Task 8)


def test_mlx_config_parses_talker_thinker_hidden_size() -> None:
    """The talker's resize MLPs are sized from a declared field, not inferred."""

    raw = tiny_root_config_dict()
    assert "thinker_hidden_size" in raw["talker_config"]

    config = Qwen3OmniMlxConfig.from_dict(raw)

    assert config.talker.thinker_hidden_size == (
        raw["talker_config"]["thinker_hidden_size"]
    )
    assert config.talker.thinker_hidden_size == (config.thinker.text_config.hidden_size)


def test_mlx_config_talker_thinker_hidden_size_is_optional() -> None:
    """A config predating the field still parses; the caller then falls back."""

    raw = tiny_root_config_dict()
    raw["talker_config"].pop("thinker_hidden_size")

    config = Qwen3OmniMlxConfig.from_dict(raw)

    assert config.talker.thinker_hidden_size is None


def test_mlx_config_rejects_disagreeing_num_code_groups() -> None:
    """A talker that emits more groups than its predictor can expand is refused."""

    raw = tiny_root_config_dict()
    raw["talker_config"]["num_code_groups"] = 8
    assert raw["talker_config"]["code_predictor_config"]["num_code_groups"] == 16

    with pytest.raises(ValueError, match="num_code_groups"):
        Qwen3OmniMlxConfig.from_dict(raw)


def test_mlx_config_accepts_agreeing_num_code_groups() -> None:
    raw = tiny_root_config_dict()
    raw["talker_config"]["num_code_groups"] = 8
    raw["talker_config"]["code_predictor_config"]["num_code_groups"] = 8
    raw["code2wav_config"]["num_quantizers"] = 8

    config = Qwen3OmniMlxConfig.from_dict(raw)

    assert config.talker.num_code_groups == 8
    assert config.talker.code_predictor_config.num_code_groups == 8
