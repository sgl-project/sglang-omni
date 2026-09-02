# SPDX-License-Identifier: Apache-2.0
"""The arch override rewrites every field SGLang sizes the KV pool from.

SGLang builds the engine's ModelConfig from the root checkpoint config and
picks the thinker text config for a model that has one, so a sub model
with fewer layers (the Qwen3-Omni talker) starts with the thinker's layer
count in both num_hidden_layers and num_attention_layers. The pool takes
the larger of the two, so the override has to rewrite both.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("sglang")

from sglang_omni.model_runner.model_worker import ModelWorker  # noqa: E402


def _qwen3_omni_engine_config():
    thinker_text = SimpleNamespace(
        num_hidden_layers=48,
        num_attention_heads=32,
        num_key_value_heads=4,
        hidden_size=2048,
        head_dim=128,
        vocab_size=152064,
    )
    talker_text = SimpleNamespace(
        num_hidden_layers=20,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_size=1024,
        head_dim=128,
        vocab_size=3072,
    )
    hf_config = SimpleNamespace(
        architectures=["Qwen3OmniMoeForConditionalGeneration"],
        thinker_config=SimpleNamespace(text_config=thinker_text),
        talker_config=SimpleNamespace(text_config=talker_text),
    )
    return SimpleNamespace(
        hf_config=hf_config,
        hf_text_config=thinker_text,
        num_attention_heads=thinker_text.num_attention_heads,
        num_key_value_heads=thinker_text.num_key_value_heads,
        hidden_size=thinker_text.hidden_size,
        num_hidden_layers=thinker_text.num_hidden_layers,
        num_attention_layers=thinker_text.num_hidden_layers,
        head_dim=thinker_text.head_dim,
        vocab_size=thinker_text.vocab_size,
    )


def test_talker_override_sizes_the_pool_from_the_talker_layers() -> None:
    config = _qwen3_omni_engine_config()
    ModelWorker._apply_arch_override(config, "Qwen3OmniTalker")
    assert config.hf_config.architectures == ["Qwen3OmniTalker"]
    assert config.hf_text_config is config.hf_config.talker_config.text_config
    assert config.num_hidden_layers == 20
    assert config.num_attention_layers == 20
    assert config.num_key_value_heads == 2
    assert config.num_attention_heads == 16
    assert config.hidden_size == 1024


def test_thinker_override_keeps_the_thinker_layers() -> None:
    config = _qwen3_omni_engine_config()
    ModelWorker._apply_arch_override(config, "Qwen3OmniThinkerForCausalLM")
    assert config.num_hidden_layers == 48
    assert config.num_attention_layers == 48
    assert config.num_key_value_heads == 4
