# SPDX-License-Identifier: Apache-2.0
"""The LM stage loads a shim holding only the LM weights, under the engine policy it needs."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
from sglang.srt.configs.model_config import get_hybrid_layer_ids, is_hybrid_swa_model
from transformers import PretrainedConfig

from sglang_omni.models.personaplex.architecture import TEMPORAL_TRANSFORMER
from sglang_omni.models.personaplex.engine_builder import (
    PersonaPlexEngineBuilder,
    shim_checkpoint_dir,
)
from sglang_omni.models.personaplex.hf_config import (
    DEFAULT_CONTEXT_LENGTH,
    build_backbone_config,
)
from sglang_omni.platforms.cpu import CPUOmniPlatform
from sglang_omni.platforms.cuda import CUDAOmniPlatform


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


def test_backbone_config_selects_only_windowed_layers() -> None:
    config = PretrainedConfig.from_dict(build_backbone_config())
    assert is_hybrid_swa_model(config.architectures, config)
    window_layers, full_layers = get_hybrid_layer_ids(config.architectures, config)
    assert window_layers == list(range(TEMPORAL_TRANSFORMER.num_layers))
    assert full_layers == []


@pytest.mark.parametrize(
    "is_cuda_host,device,page_size,disable_radix_cache",
    [
        (True, "cuda", 1, True),
        (True, "cpu", 1, True),
        (True, "cuda", 64, True),
        (False, "cpu", 1, True),
        (True, "cuda", 1, False),
    ],
)
def test_window_kv_uses_resolved_device_and_cache_settings(
    monkeypatch: pytest.MonkeyPatch,
    is_cuda_host: bool,
    device: Literal["cuda", "cpu"],
    page_size: int,
    disable_radix_cache: bool,
) -> None:
    raw_server_args = SimpleNamespace(device="cuda", page_size=128)
    declarations: list[tuple[str, bool]] = []

    def record_override(
        server_args: SimpleNamespace, source: str, *, disable_hybrid_swa_memory: bool
    ) -> None:
        assert server_args is raw_server_args
        declarations.append((source, disable_hybrid_swa_memory))

    def resolved_configuration(server_args: SimpleNamespace) -> SimpleNamespace:
        assert server_args is raw_server_args
        return SimpleNamespace(
            device=device, page_size=page_size, disable_radix_cache=disable_radix_cache
        )

    platform = CUDAOmniPlatform() if is_cuda_host else CPUOmniPlatform()
    module = "sglang_omni.models.personaplex.engine_builder"
    monkeypatch.setattr(f"{module}.current_platform", platform)
    monkeypatch.setattr(
        f"{module}.resolved_view",
        resolved_configuration,
    )
    monkeypatch.setattr(f"{module}.override_server_args", record_override)
    PersonaPlexEngineBuilder().customize_server_args(raw_server_args)
    if is_cuda_host and device == "cuda" and page_size == 1 and disable_radix_cache:
        assert declarations == []
    else:
        assert declarations == [("sglang_omni.personaplex.window_kv", True)]
