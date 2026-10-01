# SPDX-License-Identifier: Apache-2.0
"""The LM stage loads a shim holding only the LM weights, under the engine policy it needs."""

import json
import shutil
from pathlib import Path

import pytest

from sglang_omni.models.personaplex.architecture import TEXT_CARD
from sglang_omni.models.personaplex.engine_builder import (
    PersonaPlexEngineBuilder,
    PersonaPlexRealtimeEngineBuilder,
    shim_checkpoint_dir,
)
from sglang_omni.models.personaplex.hf_config import DEFAULT_CONTEXT_LENGTH
from sglang_omni.models.personaplex.lm_session import PersonaPlexSessionAdapter


def write_checkpoint(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.safetensors").write_text("lm")
    (root / "tokenizer-e351c8d8-checkpoint125.safetensors").write_text("mimi")
    (root / "tokenizer_spm_32k_3.model").write_text("spm")
    return root


@pytest.mark.parametrize("context_length", [None, 2048])
def test_builder_writes_backbone_config_and_links_only_lm_weights(
    tmp_path: Path, context_length: int | None
) -> None:
    source = write_checkpoint(tmp_path / "checkpoint")
    builder = (
        PersonaPlexEngineBuilder()
        if context_length is None
        else PersonaPlexEngineBuilder(context_length=context_length)
    )
    expected_context = (
        DEFAULT_CONTEXT_LENGTH if context_length is None else context_length
    )
    assert builder.context_length == expected_context
    shim = Path(builder.resolve_checkpoint(str(source)))
    try:
        assert sorted(p.name for p in shim.iterdir()) == [
            "config.json",
            "model.safetensors",
        ]
        weights = shim / "model.safetensors"
        assert weights.is_symlink()
        assert weights.resolve() == (source / "model.safetensors").resolve()
        config = json.loads((shim / "config.json").read_text())
        assert config["architectures"] == ["PersonaPlexForCausalLM"]
        assert config["max_position_embeddings"] == expected_context
        assert config["model_type"] == "llama"
        assert config["rope_is_neox_style"] is False
        assert config["rms_norm_eps"] == 1e-8
        assert config["intermediate_size"] == 11264
        assert config["vocab_size"] == 32000
    finally:
        shutil.rmtree(shim, ignore_errors=True)


def test_shim_requires_the_lm_weights(tmp_path):
    with pytest.raises(FileNotFoundError, match="LM weights missing"):
        shim_checkpoint_dir(tmp_path, context_length=4096)


def test_generation_defaults_keep_the_runner_assumptions():
    defaults = PersonaPlexEngineBuilder().generation_defaults(dtype="bfloat16")
    assert defaults["max_running_requests"] == 1
    assert defaults["chunked_prefill_size"] == -1
    assert defaults["disable_overlap_schedule"] is True
    assert defaults["disable_cuda_graph"] is True
    assert defaults["sampling_backend"] == "pytorch"


def test_realtime_engine_serves_calls_through_streaming_sessions():
    builder = PersonaPlexRealtimeEngineBuilder(
        max_running_requests=2, context_length=4096
    )
    defaults = builder.generation_defaults(dtype="bfloat16")
    assert defaults["enable_streaming_session"] is True
    assert defaults["max_running_requests"] == 2
    assert defaults["disable_radix_cache"] is True
    adapter = builder.extra_scheduler_kwargs()["session_adapter"]
    assert isinstance(adapter, PersonaPlexSessionAdapter)
    assert (adapter.vocab_size, adapter.context_length) == (TEXT_CARD, 4096)
