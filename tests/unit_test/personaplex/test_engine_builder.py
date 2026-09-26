# SPDX-License-Identifier: Apache-2.0
"""The LM stage loads a shim holding only the LM weights, under the engine policy it needs."""

import json
import shutil
from pathlib import Path

import pytest

from sglang_omni.models.personaplex.engine_builder import (
    PersonaPlexEngineBuilder,
    shim_checkpoint_dir,
)
from sglang_omni.models.personaplex.hf_config import DEFAULT_CONTEXT_LENGTH


def write_checkpoint(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.safetensors").write_text("lm")
    (root / "tokenizer-e351c8d8-checkpoint125.safetensors").write_text("mimi")
    (root / "tokenizer_spm_32k_3.model").write_text("spm")
    return root


def test_shim_links_only_the_lm_weights(tmp_path):
    source = write_checkpoint(tmp_path / "checkpoint")
    shim = shim_checkpoint_dir(source, context_length=4096)
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
        assert config["max_position_embeddings"] == 4096
    finally:
        shutil.rmtree(shim, ignore_errors=True)


def test_shim_requires_the_lm_weights(tmp_path):
    with pytest.raises(FileNotFoundError, match="LM weights missing"):
        shim_checkpoint_dir(tmp_path, context_length=4096)


class StopAfterCheckpoint(Exception):
    pass


def test_engine_context_length_reaches_the_shim(tmp_path):
    builder = PersonaPlexEngineBuilder()
    assert builder.context_length == DEFAULT_CONTEXT_LENGTH
    shims = []
    make_shim = builder.resolve_checkpoint

    def resolve_checkpoint(model_path):
        shims.append(Path(make_shim(model_path)))
        raise StopAfterCheckpoint

    builder.resolve_checkpoint = resolve_checkpoint
    with pytest.raises(StopAfterCheckpoint):
        builder.build(
            str(write_checkpoint(tmp_path)),
            device="cpu",
            server_args_overrides={"context_length": 2048},
        )
    try:
        config = json.loads((shims[0] / "config.json").read_text())
        assert config["max_position_embeddings"] == 2048
    finally:
        shutil.rmtree(shims[0], ignore_errors=True)


def test_generation_defaults_keep_the_runner_assumptions():
    defaults = PersonaPlexEngineBuilder().generation_defaults(dtype="bfloat16")
    assert defaults["max_running_requests"] == 1
    assert defaults["chunked_prefill_size"] == -1
    assert defaults["disable_overlap_schedule"] is True
    assert defaults["disable_cuda_graph"] is True
    assert defaults["sampling_backend"] == "pytorch"
