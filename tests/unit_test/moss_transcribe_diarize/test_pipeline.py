# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect

import httpx
import pytest
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
