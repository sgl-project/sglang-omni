# SPDX-License-Identifier: Apache-2.0
"""The LM stage loads a shim holding only the LM weights, under the engine policy it needs."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import Mock

import pytest
from sglang.srt.configs.model_config import get_hybrid_layer_ids, is_hybrid_swa_model
from transformers import PretrainedConfig

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import (
    resolve_stage_factory_args,
    resolve_stage_typed_kwargs,
)
from sglang_omni.models.personaplex.architecture import (
    DEFAULT_AUDIO_TEMPERATURE,
    DEFAULT_AUDIO_TOP_K,
    TEMPORAL_TRANSFORMER,
    TEXT_CARD,
)
from sglang_omni.models.personaplex.config import (
    PersonaPlexPipelineConfig,
    PersonaPlexSessionPipelineConfig,
)
from sglang_omni.models.personaplex.engine_builder import (
    PersonaPlexEngineBuilder,
    PersonaPlexSessionEngineBuilder,
    shim_checkpoint_dir,
)
from sglang_omni.models.personaplex.hf_config import (
    DEFAULT_CONTEXT_LENGTH,
    build_backbone_config,
)
from sglang_omni.models.personaplex.lm_session import PersonaPlexSessionAdapter
from sglang_omni.models.personaplex.sampling import AudioSampling
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.cpu import CPUOmniPlatform
from sglang_omni.platforms.cuda import CUDAOmniPlatform

DEFAULT_DEPFORMER_SAMPLING = AudioSampling(
    DEFAULT_AUDIO_TEMPERATURE, DEFAULT_AUDIO_TOP_K
)


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
        PersonaPlexEngineBuilder(
            enable_async_decode=False,
            depformer_cuda_graph_batch_sizes=[],
            depformer_cuda_graph_sampling=DEFAULT_DEPFORMER_SAMPLING,
        )
        if context_length is None
        else PersonaPlexEngineBuilder(
            enable_async_decode=False,
            depformer_cuda_graph_batch_sizes=[],
            depformer_cuda_graph_sampling=DEFAULT_DEPFORMER_SAMPLING,
            context_length=context_length,
        )
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


@pytest.mark.parametrize("is_cuda", [True, False])
def test_generation_defaults_keep_the_runner_assumptions(monkeypatch, is_cuda):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: is_cuda)
    defaults = PersonaPlexEngineBuilder(
        enable_async_decode=False,
        depformer_cuda_graph_batch_sizes=[],
        depformer_cuda_graph_sampling=DEFAULT_DEPFORMER_SAMPLING,
    ).generation_defaults(dtype="bfloat16")
    assert defaults["max_running_requests"] == 1
    assert defaults["chunked_prefill_size"] == -1
    assert defaults["disable_overlap_schedule"] is True
    assert defaults["disable_cuda_graph"] is not is_cuda
    assert defaults["sampling_backend"] == "pytorch"


@pytest.mark.parametrize(
    "is_cuda,disable_cuda_graph,batch_sizes,should_capture",
    [
        (True, False, [1, 2, 4, 8], True),
        (False, False, [1, 2, 4, 8], False),
        (True, True, [1, 2, 4, 8], False),
        (True, False, [], False),
    ],
)
def test_depformer_graphs_capture_at_runner_build(
    monkeypatch: pytest.MonkeyPatch,
    is_cuda: bool,
    disable_cuda_graph: bool,
    batch_sizes: list[int],
    should_capture: bool,
) -> None:
    runner = SimpleNamespace(
        model=SimpleNamespace(depformer="depformer"), depformer_cuda_graph=None
    )
    graph_runner = SimpleNamespace(capture_graphs=Mock())
    graph_runner_factory = Mock(return_value=graph_runner)
    module = "sglang_omni.models.personaplex.engine_builder"
    monkeypatch.setattr(f"{module}.PersonaPlexModelRunner", Mock(return_value=runner))
    monkeypatch.setattr(f"{module}.DepformerCudaGraphRunner", graph_runner_factory)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: is_cuda)
    model_worker = SimpleNamespace(
        server_args=SimpleNamespace(
            disable_cuda_graph=disable_cuda_graph, max_running_requests=4
        )
    )
    builder = PersonaPlexEngineBuilder(
        enable_async_decode=False,
        depformer_cuda_graph_batch_sizes=batch_sizes,
        depformer_cuda_graph_sampling=DEFAULT_DEPFORMER_SAMPLING,
    )

    assert builder.make_model_runner(model_worker, Mock()) is runner
    if should_capture:
        graph_runner_factory.assert_called_once_with("depformer", tuple(batch_sizes))
        sampling = graph_runner.capture_graphs.call_args.args[0]
        assert (sampling.temperature, sampling.top_k) == (
            DEFAULT_AUDIO_TEMPERATURE,
            DEFAULT_AUDIO_TOP_K,
        )
        assert graph_runner.capture_graphs.call_args.kwargs == {"max_batch_size": 4}
        assert runner.depformer_cuda_graph is graph_runner
    else:
        graph_runner_factory.assert_not_called()
        assert runner.depformer_cuda_graph is None


def test_backbone_config_selects_only_windowed_layers() -> None:
    config = PretrainedConfig.from_dict(build_backbone_config())
    assert is_hybrid_swa_model(config.architectures, config)
    window_layers, full_layers = get_hybrid_layer_ids(config.architectures, config)
    assert window_layers == list(range(TEMPORAL_TRANSFORMER.num_layers))
    assert full_layers == []


@pytest.mark.parametrize(
    "is_cuda_host,device,page_size",
    [
        (True, "cuda", 1),
        (True, "cpu", 1),
        (True, "cuda", 64),
        (False, "cpu", 1),
    ],
)
def test_window_kv_uses_resolved_device_and_page_size(
    monkeypatch: pytest.MonkeyPatch,
    is_cuda_host: bool,
    device: Literal["cuda", "cpu"],
    page_size: int,
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
        return SimpleNamespace(device=device, page_size=page_size)

    platform = CUDAOmniPlatform() if is_cuda_host else CPUOmniPlatform()
    module = "sglang_omni.models.personaplex.engine_builder"
    monkeypatch.setattr(f"{module}.current_platform", platform)
    monkeypatch.setattr(
        f"{module}.resolved_view",
        resolved_configuration,
    )
    monkeypatch.setattr(f"{module}.override_server_args", record_override)
    PersonaPlexEngineBuilder(
        enable_async_decode=False,
        depformer_cuda_graph_batch_sizes=[],
        depformer_cuda_graph_sampling=DEFAULT_DEPFORMER_SAMPLING,
    ).customize_server_args(raw_server_args)
    if is_cuda_host and device == "cuda" and page_size == 1:
        assert declarations == []
    else:
        assert declarations == [("sglang_omni.personaplex.window_kv", True)]


def test_realtime_engine_serves_calls_through_streaming_sessions():
    tokenizer = object()
    builder = PersonaPlexSessionEngineBuilder(
        tokenizer=tokenizer,
        context_length=4096,
        enable_async_decode=False,
        depformer_cuda_graph_batch_sizes=[1],
        depformer_cuda_graph_sampling=AudioSampling(0.5, 7),
    )
    defaults = builder.generation_defaults(dtype="bfloat16")
    assert defaults["enable_streaming_session"] is True
    assert defaults["disable_radix_cache"] is True
    adapter = builder.extra_scheduler_kwargs()["session_adapter"]
    assert isinstance(adapter, PersonaPlexSessionAdapter)
    assert (adapter.vocab_size, adapter.context_length) == (TEXT_CARD, 4096)
    assert adapter.tokenizer is tokenizer


@pytest.mark.parametrize("batch_sizes", [[], [1, 4]])
def test_depformer_cuda_graph_buckets_accept_dotted_cli(batch_sizes: list[int]) -> None:
    config = ConfigManager(
        PersonaPlexPipelineConfig(model_path="personaplex")
    ).merge_config(
        {"lm.factory.depformer_cuda_graph_batch_sizes": json.dumps(batch_sizes)}
    )
    assert (
        resolve_stage_typed_kwargs(config.stage_named("lm"))[
            "depformer_cuda_graph_batch_sizes"
        ]
        == batch_sizes
    )


def test_radix_caching_is_refused_at_startup() -> None:
    """With radix caching on, a request could reuse another voice's prompt KV."""
    builder = PersonaPlexEngineBuilder(
        enable_async_decode=False,
        depformer_cuda_graph_batch_sizes=[],
        depformer_cuda_graph_sampling=DEFAULT_DEPFORMER_SAMPLING,
    )
    with pytest.raises(ValueError, match="disable_radix_cache=True"):
        builder.validate_before_infrastructure(
            SimpleNamespace(disable_radix_cache=False)
        )
    builder.validate_before_infrastructure(SimpleNamespace(disable_radix_cache=True))


@pytest.mark.parametrize("batch_sizes", [None, [], [1, 4]])
def test_session_graph_sizes_follow_concurrency_and_cli(
    batch_sizes: list[int] | None,
) -> None:
    """Resolved session factories follow concurrency changes and explicit buckets."""
    config = PersonaPlexSessionPipelineConfig(model_path="personaplex")
    if batch_sizes is not None:
        config = ConfigManager(config).merge_config(
            {"lm.factory.depformer_cuda_graph_batch_sizes": json.dumps(batch_sizes)}
        )
    else:
        pass
    for max_sessions in (4, 32, 1):
        config = ConfigManager(config).merge_config({"max_sessions": max_sessions})
        factory_kwargs = resolve_stage_factory_args(config.stage_named("lm"), config)
        expected_batch_sizes = (
            list(range(1, max_sessions + 2)) if batch_sizes is None else batch_sizes
        )
        assert (
            factory_kwargs["depformer_cuda_graph_batch_sizes"] == expected_batch_sizes
        )
        assert factory_kwargs["server_args_overrides"]["max_running_requests"] == (
            max_sessions + 1
        )
