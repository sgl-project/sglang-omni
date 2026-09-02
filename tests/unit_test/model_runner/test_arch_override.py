# SPDX-License-Identifier: Apache-2.0
"""The KV pool of a sub model engine is sized from the sub model's layers.

SGLang builds the engine's ModelConfig from the root checkpoint config and,
for a config with a thinker_config, takes the thinker's text config, so
the Qwen3-Omni talker engine starts with the thinker's 48 layers in both
num_hidden_layers and num_attention_layers. SGLang then sizes the pool
from the larger of the two through resolve_layer_indices. The override
has to leave the pool at the talker's 20 layers and the thinker's at 48.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("sglang")

from sglang.srt.model_executor.model_runner_components.layer_setup import (  # noqa: E402
    resolve_layer_indices,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402

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
        num_nextn_predict_layers=None,
        head_dim=thinker_text.head_dim,
        vocab_size=thinker_text.vocab_size,
    )


def _pool_layers(config) -> int:
    return resolve_layer_indices(
        model=object(),
        model_config=config,
        is_draft_worker=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    ).num_effective_layers


def test_talker_pool_is_sized_from_the_talker_layers() -> None:
    config = _qwen3_omni_engine_config()
    assert _pool_layers(config) == 48
    ModelWorker._apply_arch_override(config, "Qwen3OmniTalker")
    assert _pool_layers(config) == 20
    assert config.num_key_value_heads == 2


def test_thinker_pool_keeps_the_thinker_layers() -> None:
    config = _qwen3_omni_engine_config()
    ModelWorker._apply_arch_override(config, "Qwen3OmniThinkerForCausalLM")
    assert _pool_layers(config) == 48
    assert config.num_key_value_heads == 4
