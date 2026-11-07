# SPDX-License-Identifier: Apache-2.0
"""Pipeline registration and backbone configuration contracts."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest
import torch
from sglang.srt.configs.model_config import ModelConfig
from transformers import Qwen3Config

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.models.ming3_asr import stages
from sglang_omni.models.ming3_asr.config import (
    Ming3ASRFactoryArgs,
    Ming3ASRPipelineConfig,
)
from sglang_omni.models.ming3_asr.stages import Ming3ASRConfig, create_thinker_executor
from sglang_omni.models.model_capabilities import get_model_capabilities
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.proto.request import OmniRequest, StagePayload

CHECKPOINT_ARCHITECTURE = "MingV3ASRForConditionalGeneration"
HIDDEN_SIZE = 64
NATIVE_CLIP_SECONDS = 30.0


def test_checkpoint_architecture_resolves_to_ming3_asr_pipeline() -> None:
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config(CHECKPOINT_ARCHITECTURE)
        is Ming3ASRPipelineConfig
    )
    capabilities = get_model_capabilities(CHECKPOINT_ARCHITECTURE)
    assert capabilities is not None
    assert capabilities.supports_reference_audio is False


def test_default_pipeline_ends_at_thinker() -> None:
    config = Ming3ASRPipelineConfig(model_path="inclusionAI/Ming-3.0-asr-tiny")
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "thinker",
    ]
    assert config.terminal_stages == ["thinker"]
    thinker = config.stages[1]
    assert isinstance(thinker.factory, Ming3ASRFactoryArgs)
    assert thinker.factory.pre_lm_max_batch_size == 8
    assert thinker.factory.pre_lm_max_batch_wait_ms == 2
    factory_arguments = resolve_stage_factory_args(thinker, config)
    assert (
        factory_arguments["server_args_overrides"]["speculative_algorithm"] == "DSPARK"
    )
    manager = ConfigManager(config)
    ordinary_config = manager.merge_config(
        manager.parse_extra_args(
            ["--thinker.engine.disable_speculative_decoding", "true"]
        )
    )
    ordinary_arguments = resolve_stage_factory_args(
        ordinary_config.stages[1], ordinary_config
    )
    assert not any(
        name.startswith("speculative_")
        for name in ordinary_arguments["server_args_overrides"]
    )


def two_stage_document(
    *,
    pre_lm_max_batch_size: int = 4,
    pre_lm_max_batch_wait_ms: int = 0,
) -> dict[str, object]:
    return {
        "model_path": "inclusionAI/Ming-3.0-asr-tiny",
        "stages": [
            {
                "name": "preprocessing",
                "process": "preprocessing",
                "factory_path": (
                    "sglang_omni.models.ming3_asr.stages.create_preprocessing_executor"
                ),
                "next": "thinker",
            },
            {
                "name": "thinker",
                "process": "thinker",
                "gpu": 0,
                "terminal": True,
                "factory_path": (
                    "sglang_omni.models.ming3_asr.stages.create_thinker_executor"
                ),
                "factory": {
                    "pre_lm_max_batch_size": pre_lm_max_batch_size,
                    "pre_lm_max_batch_wait_ms": pre_lm_max_batch_wait_ms,
                },
            },
        ],
    }


def test_two_stage_configuration_loads_pre_lm_knobs() -> None:
    config = Ming3ASRPipelineConfig.model_validate(two_stage_document())
    assert [stage.name for stage in config.stages] == ["preprocessing", "thinker"]
    thinker = config.stages[1]
    assert thinker.factory.pre_lm_max_batch_size == 4
    assert thinker.factory.pre_lm_max_batch_wait_ms == 0


@pytest.mark.parametrize(
    ("batch_size", "wait_ms", "match"),
    [
        (0, 2, "pre_lm_max_batch_size"),
        (8, -1, "pre_lm_max_batch_wait_ms"),
    ],
)
def test_configuration_rejects_invalid_pre_lm_knobs(
    batch_size: int,
    wait_ms: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        Ming3ASRPipelineConfig.model_validate(
            two_stage_document(
                pre_lm_max_batch_size=batch_size,
                pre_lm_max_batch_wait_ms=wait_ms,
            )
        )


@pytest.mark.parametrize(
    ("batch_size", "wait_ms", "match"),
    [
        (0, 2, "pre_lm_max_batch_size"),
        (8, -1, "pre_lm_max_batch_wait_ms"),
    ],
)
def test_thinker_factory_rejects_invalid_pre_lm_knobs_before_load(
    batch_size: int,
    wait_ms: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        create_thinker_executor(
            "missing-checkpoint",
            max_seq_len=8,
            max_new_tokens=4,
            stream_emit_interval_seconds=0.05,
            pre_lm_max_batch_size=batch_size,
            pre_lm_max_batch_wait_ms=wait_ms,
            request_build_max_workers=1,
            request_build_max_pending=1,
        )


def test_thinker_factory_rejects_tensor_parallelism_before_load() -> None:
    with pytest.raises(ValueError, match="tp_size"):
        create_thinker_executor(
            "missing-checkpoint",
            max_seq_len=8,
            max_new_tokens=4,
            stream_emit_interval_seconds=0.05,
            pre_lm_max_batch_size=8,
            pre_lm_max_batch_wait_ms=2,
            request_build_max_workers=1,
            request_build_max_pending=1,
            tp_size=2,
        )


def test_long_uploads_split_into_native_clips() -> None:
    config = Ming3ASRPipelineConfig(model_path="inclusionAI/Ming-3.0-asr-tiny")
    chunking = config.resolved_audio_chunking
    assert chunking.allow_audio_chunking is True
    assert chunking.max_native_clip_s == NATIVE_CLIP_SECONDS


def test_backbone_config_flattens_text_config() -> None:
    config = Ming3ASRConfig(
        text_config={
            "hidden_size": HIDDEN_SIZE,
            "model_type": "bailing_moe_v3",
        },
        audio_token_id=7,
    )
    assert config.model_type == "ming_v3_asr"
    assert config.hidden_size == HIDDEN_SIZE
    assert config.architectures == ["BailingMoeV3ForCausalLM"]
    assert config.norm_topk_prob is True
    explicit_config = Ming3ASRConfig(
        text_config={"hidden_size": HIDDEN_SIZE, "norm_topk_prob": False}
    )
    assert explicit_config.norm_topk_prob is False


def test_dspark_checkpoint_lives_until_stage_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DSpark greedy requests must survive architecture and sampling adaptation."""
    checkpoint = TemporaryDirectory(prefix="test-ming-dspark-")
    checkpoint_path = Path(checkpoint.name)
    Qwen3Config(
        architectures=["ReducedVocabularyDSparkDraftModel"],
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
    ).save_pretrained(checkpoint_path)
    monkeypatch.setattr(
        stages,
        "resolve_concrete_device",
        MagicMock(return_value=torch.device("cuda", 0)),
    )
    monkeypatch.setattr(
        stages, "materialize_draft_checkpoint", MagicMock(return_value=checkpoint)
    )
    monkeypatch.setattr(
        stages,
        "load_checkpoint_config",
        MagicMock(
            return_value={
                "audio_config": {},
                "text_config": {"hidden_size": 64, "eos_token_id": 7},
            }
        ),
    )
    for name in (
        "AudioEncoderConfig",
        "Ming3AudioEncoder",
        "AutoConfig",
        "ModelRegistry",
        "PreTrainedTokenizerFast",
        "Ming3AudioModelRunner",
        "Ming3ASRStreamOutputBuilder",
        "OmniScheduler",
    ):
        monkeypatch.setattr(stages, name, MagicMock())
    encoder_service = MagicMock()
    monkeypatch.setattr(stages, "Ming3ASRPreLMEncoderService", encoder_service)
    server_arguments = MagicMock()
    monkeypatch.setattr(stages, "build_sglang_server_args", server_arguments)
    infrastructure = MagicMock(return_value=tuple(MagicMock() for _ in range(5)))
    infrastructure.return_value[-1].vocab_size = 16
    monkeypatch.setattr(stages, "create_sglang_infrastructure", infrastructure)

    scheduler = create_thinker_executor(
        "target-checkpoint",
        max_seq_len=8192,
        max_new_tokens=256,
        stream_emit_interval_seconds=0.05,
        pre_lm_max_batch_size=8,
        pre_lm_max_batch_wait_ms=2,
        request_build_max_workers=1,
        request_build_max_pending=1,
        device="cuda",
        gpu_id=0,
        server_args_overrides={
            "speculative_algorithm": "DSPARK",
            "speculative_draft_model_path": "draft-checkpoint",
        },
    )
    assert checkpoint_path.is_dir()
    assert infrastructure.call_args.kwargs["supports_speculative_decoding"] is True
    overrides = server_arguments.call_args.kwargs
    assert overrides["speculative_draft_model_path"] == str(checkpoint_path)
    assert overrides["speculative_num_draft_tokens"] == 9
    assert overrides["speculative_dspark_block_size"] == 8
    assert overrides["disable_overlap_schedule"] is True
    draft_config = ModelConfig(
        str(checkpoint_path),
        model_override_args=overrides.get("json_model_override_args", "{}"),
        is_draft_model=True,
        speculative_algorithm="DSPARK",
    )
    assert draft_config.hf_config.architectures == ["ReducedVocabularyDSparkDraftModel"]

    payload = StagePayload(
        request_id="greedy-request",
        request=OmniRequest(inputs=None, params={"temperature": 0.7}),
        data={
            "features": [],
            "input_ids": torch.tensor([1, 7]),
            "audio_positions": torch.tensor([1]),
            "audio_embeddings": torch.zeros(1, HIDDEN_SIZE),
        },
    )
    with pytest.raises(ValueError, match="requires temperature=0"):
        stages.OmniScheduler.call_args.kwargs["request_builder"](payload)
    payload.request.params = {}
    greedy_request = stages.OmniScheduler.call_args.kwargs["request_builder"](payload)
    assert greedy_request.req.sampling_params.top_k == 1

    scheduler.shutdown_callback()
    assert not checkpoint_path.exists()
    encoder_service.return_value.close.assert_called_once_with()


def test_dspark_startup_failure_releases_adapted_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = TemporaryDirectory(prefix="test-ming-dspark-")
    checkpoint_path = Path(checkpoint.name)
    monkeypatch.setattr(
        stages,
        "resolve_concrete_device",
        MagicMock(return_value=torch.device("cuda", 0)),
    )
    monkeypatch.setattr(
        stages, "materialize_draft_checkpoint", MagicMock(return_value=checkpoint)
    )
    with pytest.raises(FileNotFoundError):
        create_thinker_executor(
            "missing-target-checkpoint",
            max_seq_len=8192,
            max_new_tokens=256,
            stream_emit_interval_seconds=0.05,
            pre_lm_max_batch_size=8,
            pre_lm_max_batch_wait_ms=2,
            request_build_max_workers=1,
            request_build_max_pending=1,
            device="cuda",
            gpu_id=0,
            server_args_overrides={
                "speculative_algorithm": "DSPARK",
                "speculative_draft_model_path": "draft-checkpoint",
            },
        )
    assert not checkpoint_path.exists()
