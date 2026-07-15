# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect

import httpx
import pytest
import torch
from huggingface_hub.errors import RepositoryNotFoundError

from sglang_omni.models.moss_transcribe_diarize.config import (
    MossTranscribeDiarizePipelineConfig,
)
from sglang_omni.models.moss_transcribe_diarize.stages import (
    _missing_additional_chat_templates_compat,
    create_sglang_moss_transcribe_diarize_executor,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY


def test_moss_transcribe_diarize_config_uses_single_batched_stage() -> None:
    config = MossTranscribeDiarizePipelineConfig(
        model_path="OpenMOSS-Team/MOSS-Transcribe-Diarize"
    )

    assert config.entry_stage == "asr"
    assert [stage.name for stage in config.stages] == ["asr"]
    assert config.terminal_stages == ["asr"]
    assert config.gpu_placement == {"asr": 0}
    assert config.stages[0].factory.endswith(
        "create_sglang_moss_transcribe_diarize_executor"
    )
    assert config.stages[0].factory_args["device"] == "cuda:0"
    assert config.stages[0].factory_args["max_running_requests"] == 16
    assert config.stages[0].factory_args["request_build_max_workers"] == 2
    assert config.stages[0].factory_args["request_build_max_pending"] == 16
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config(
            "MossTranscribeDiarizeForConditionalGeneration"
        )
        is MossTranscribeDiarizePipelineConfig
    )
    assert MossTranscribeDiarizePipelineConfig.mem_fraction_role_to_stage() == {
        "asr": "asr"
    }
    assert MossTranscribeDiarizePipelineConfig.generation_sglang_role_to_stage() == {
        "generation": "asr"
    }


def test_moss_transcribe_diarize_stage_reserves_encoder_headroom() -> None:
    signature = inspect.signature(create_sglang_moss_transcribe_diarize_executor)

    assert signature.parameters["max_running_requests"].default == 16
    assert signature.parameters["mem_fraction_static"].default == 0.80
    assert signature.parameters["request_build_max_workers"].default == 2
    assert signature.parameters["request_build_max_pending"].default == 16
    assert signature.parameters["mm_embedding_cache_size_bytes"].default == 0
    assert signature.parameters["encoder_chunk_buckets"].default is None
    assert signature.parameters["encoder_torch_compile"].default is False


def test_compile_encoder_sets_runner_and_warms_each_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from sglang_omni.models.moss_transcribe_diarize.sglang_model import (
        MossTranscribeDiarizeForConditionalGeneration as Model,
    )

    monkeypatch.setattr(
        "sglang_omni.models.moss_transcribe_diarize.sglang_model.set_torch_compile_config",
        lambda: None,
    )
    warmups: list[tuple[int, ...]] = []
    runner = lambda feats, pos, forward_batch: warmups.append(tuple(feats.shape))
    monkeypatch.setattr(torch, "compile", lambda module, **kwargs: runner)

    encoder = torch.nn.Linear(4, 4)
    model = SimpleNamespace(
        whisper_encoder=encoder,
        _compiled_encoder=None,
        _compiled_chunk_buckets=frozenset(),
        config=SimpleNamespace(audio_config=SimpleNamespace(num_mel_bins=4)),
    )

    Model.compile_encoder(model, [2, 1, 1], input_feature_len=6)

    assert model._compiled_encoder is runner
    assert model._compiled_chunk_buckets == frozenset({1, 2})
    assert model._compiled_input_feature_len == 6
    assert len(warmups) == 6
    assert {shape[0] for shape in warmups} == {1, 2}
    assert all(shape[1:] == (4, 6) for shape in warmups)


def test_compile_encoder_drops_bucket_whose_warmup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from sglang_omni.models.moss_transcribe_diarize.sglang_model import (
        MossTranscribeDiarizeForConditionalGeneration as Model,
    )

    monkeypatch.setattr(
        "sglang_omni.models.moss_transcribe_diarize.sglang_model.set_torch_compile_config",
        lambda: None,
    )

    def runner(feats, pos, forward_batch):
        if feats.shape[0] == 2:
            raise RuntimeError("simulated OOM during warmup")

    monkeypatch.setattr(torch, "compile", lambda module, **kwargs: runner)

    model = SimpleNamespace(
        whisper_encoder=torch.nn.Linear(4, 4),
        _compiled_encoder=None,
        _compiled_chunk_buckets=frozenset(),
        _compiled_input_feature_len=0,
        config=SimpleNamespace(audio_config=SimpleNamespace(num_mel_bins=4)),
    )

    Model.compile_encoder(model, [1, 2], input_feature_len=6)

    assert model._compiled_chunk_buckets == frozenset({1})


def _stub_factory_env(monkeypatch: pytest.MonkeyPatch, *, want_cuda_graph: bool):
    from types import SimpleNamespace

    from sglang_omni.models.moss_transcribe_diarize import stages

    calls = {
        "init_device_graphs": 0,
        "compile_encoder": [],
        "init_encoder_graphs": [],
    }
    model = SimpleNamespace(
        compile_encoder=lambda buckets, feat_len: calls["compile_encoder"].append(
            (list(buckets), feat_len)
        ),
        init_encoder_graphs=lambda buckets, feat_len: calls[
            "init_encoder_graphs"
        ].append((list(buckets), feat_len)),
        init_encoder_cache=lambda n: None,
    )

    def _bump_init_device_graphs() -> None:
        calls["init_device_graphs"] += 1

    model_runner = SimpleNamespace(
        model=model, init_device_graphs=_bump_init_device_graphs
    )
    model_worker = SimpleNamespace(model_runner=model_runner)
    infra = (want_cuda_graph, (model_worker, None, None, None, None, None, None))

    processor = SimpleNamespace(
        tokenizer=object(),
        feature_extractor=SimpleNamespace(nb_max_frames=3000),
    )

    monkeypatch.setattr(
        stages,
        "AutoProcessor",
        SimpleNamespace(from_pretrained=lambda *a, **k: processor),
    )
    monkeypatch.setattr(stages, "_default_max_new_tokens", lambda path: 100)
    monkeypatch.setattr(stages, "_default_context_length", lambda path: 4096)
    monkeypatch.setattr(stages, "build_generation_batch_overrides", lambda **k: {})
    monkeypatch.setattr(stages, "build_sglang_server_args", lambda *a, **k: object())
    monkeypatch.setattr(stages, "validate_generation_batch_policy", lambda **k: None)
    monkeypatch.setattr(
        stages, "create_sglang_infrastructure_defer_cuda_graph", lambda *a, **k: infra
    )
    monkeypatch.setattr(stages, "init_mm_embedding_cache", lambda n: None)
    monkeypatch.setattr(
        stages,
        "make_moss_transcribe_diarize_scheduler_adapters",
        lambda **k: (object(), object()),
    )
    monkeypatch.setattr(
        stages,
        "make_moss_transcribe_diarize_stream_output_builder",
        lambda **k: object(),
    )
    monkeypatch.setattr(stages, "SGLangOutputProcessor", lambda **k: object())
    monkeypatch.setattr(stages, "ModelRunner", lambda *a, **k: object())
    monkeypatch.setattr(stages, "OmniScheduler", lambda **k: SimpleNamespace())
    return calls


def test_factory_compiles_encoder_and_skips_cuda_graph_when_flag_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_factory_env(monkeypatch, want_cuda_graph=True)

    create_sglang_moss_transcribe_diarize_executor(
        "OpenMOSS-Team/MOSS-Transcribe-Diarize", encoder_torch_compile=True
    )

    assert len(calls["compile_encoder"]) == 1
    assert calls["init_encoder_graphs"] == []
    assert calls["init_device_graphs"] == 1


def _repo_not_found(url: str) -> RepositoryNotFoundError:
    response = httpx.Response(404, request=httpx.Request("GET", url))
    return RepositoryNotFoundError(f"missing: {url}", response=response)


def test_processor_compat_ignores_missing_additional_chat_templates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers.processing_utils as processing_utils
    import transformers.utils.hub as hub_utils

    def missing_templates(*_args: object, **_kwargs: object) -> list[str]:
        raise _repo_not_found(
            "https://huggingface.co/api/models/repo/tree/main/"
            "additional_chat_templates"
        )

    monkeypatch.setattr(processing_utils, "list_repo_templates", missing_templates)
    monkeypatch.setattr(hub_utils, "list_repo_templates", missing_templates)

    with _missing_additional_chat_templates_compat():
        assert (
            processing_utils.list_repo_templates("repo", local_files_only=False) == []
        )
        assert hub_utils.list_repo_templates("repo", local_files_only=False) == []


def test_processor_compat_preserves_non_template_repo_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers.processing_utils as processing_utils

    def missing_repo(*_args: object, **_kwargs: object) -> list[str]:
        raise _repo_not_found("https://huggingface.co/api/models/missing-repo")

    monkeypatch.setattr(processing_utils, "list_repo_templates", missing_repo)

    with _missing_additional_chat_templates_compat():
        with pytest.raises(RepositoryNotFoundError, match="missing-repo"):
            processing_utils.list_repo_templates("missing-repo", local_files_only=False)
