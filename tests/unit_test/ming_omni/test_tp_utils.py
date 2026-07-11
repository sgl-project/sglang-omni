# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from sglang_omni.models.ming_omni.tp_utils import (
    validate_attention_tp_config,
    validate_stage_tp_support,
)


def test_validate_attention_tp_config_accepts_even_attention_and_kv_sharding():
    validate_attention_tp_config(
        num_attention_heads=32,
        num_key_value_heads=4,
        tp_size=4,
        context="ming thinker",
    )


def test_validate_attention_tp_config_accepts_kv_replication():
    validate_attention_tp_config(
        num_attention_heads=32,
        num_key_value_heads=4,
        tp_size=8,
        context="ming thinker",
    )


def test_validate_attention_tp_config_rejects_invalid_attention_heads():
    with pytest.raises(ValueError, match="num_attention_heads=32.*tp_size=3"):
        validate_attention_tp_config(
            num_attention_heads=32,
            num_key_value_heads=4,
            tp_size=3,
            context="ming thinker",
        )


def test_validate_attention_tp_config_rejects_invalid_kv_sharding():
    with pytest.raises(ValueError, match="num_key_value_heads=6.*tp_size=4"):
        validate_attention_tp_config(
            num_attention_heads=32,
            num_key_value_heads=6,
            tp_size=4,
            context="ming thinker",
        )


def test_validate_attention_tp_config_rejects_invalid_kv_replication():
    with pytest.raises(ValueError, match="tp_size=8.*num_key_value_heads=3"):
        validate_attention_tp_config(
            num_attention_heads=24,
            num_key_value_heads=3,
            tp_size=8,
            context="ming thinker",
        )


def test_validate_stage_tp_support_allows_thinker_tp():
    validate_stage_tp_support(stage_name="thinker", tp_size=4)


def test_validate_stage_tp_support_allows_image_encoder_tp():
    validate_stage_tp_support(stage_name="image_encoder", tp_size=2)


def test_validate_stage_tp_support_rejects_unsupported_stage_tp():
    with pytest.raises(ValueError, match="Stage 'audio_encoder'.*does not support TP"):
        validate_stage_tp_support(stage_name="audio_encoder", tp_size=2)
