# SPDX-License-Identifier: Apache-2.0
"""Pipeline registration and backbone configuration contracts."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.arg_groups import memory_hook
from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from sglang_omni.client.client import Client
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.models.ming3_asr import stages
from sglang_omni.models.ming3_asr.config import (
    Ming3ASRFactoryArgs,
    Ming3ASRPipelineConfig,
)
from sglang_omni.models.ming3_asr.stages import Ming3ASRConfig, create_thinker_executor
from sglang_omni.models.ming3_asr.text_streaming import Ming3ASRRequestData
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


@pytest.mark.parametrize("disable_mtp", [False, True])
def test_native_mtp_configuration_and_result_routing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, disable_mtp: bool
) -> None:
    """Bundled MTP must keep the target depth and return acceptance statistics."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "ming_v3_asr",
                "architectures": [CHECKPOINT_ARCHITECTURE],
                "text_config": {
                    "hidden_size": 1536,
                    "num_hidden_layers": 24,
                    "num_nextn_predict_layers": 1,
                    "layer_group_size": 4,
                    "short_conv_kernel_size": 4,
                    "eos_token_id": 156892,
                },
                "audio_config": {
                    "n_mels": 128,
                    "n_audio_ctx": 3000,
                    "n_audio_state": 1280,
                    "n_audio_head": 20,
                    "n_audio_layer": 32,
                    "downsampler_kernel_size_list": [3],
                    "downsampler_stride_list": [2],
                },
            }
        )
    )
    monkeypatch.setattr(
        stages.ModelRegistry, "models", dict(stages.ModelRegistry.models)
    )
    monkeypatch.delitem(
        stages.ModelRegistry.models, "BailingMoeForCausalLMNextN", raising=False
    )
    monkeypatch.setattr(
        memory_hook, "get_device_memory_capacity", lambda device: 80 * 1024
    )
    monkeypatch.setattr(stages, "Ming3AudioEncoder", Mock())
    monkeypatch.setattr(stages, "Ming3ASRPreLMEncoderService", Mock())
    monkeypatch.setattr(stages, "Ming3AudioModelRunner", Mock())
    infrastructure = Mock(return_value=(None, None, None, None, None))
    monkeypatch.setattr(stages, "create_sglang_infrastructure", infrastructure)
    tokenizer = Mock(decode=Mock(return_value="English\thello"))
    monkeypatch.setattr(
        stages.PreTrainedTokenizerFast, "from_pretrained", Mock(return_value=tokenizer)
    )
    scheduler = Mock()
    monkeypatch.setattr(stages, "OmniScheduler", scheduler)
    manager = ConfigManager(Ming3ASRPipelineConfig(model_path=str(tmp_path)))
    pipeline = manager.merge_config(
        manager.parse_extra_args(
            [
                "--thinker.factory.device",
                "cpu",
                "--thinker.engine.device",
                "cpu",
                "--thinker.engine.speculative_algorithm",
                "NEXTN",
                "--thinker.engine.enable_mixed_chunk",
                "true",
                "--thinker.engine.revision",
                "snapshot",
                "--thinker.engine.disable_speculative_decoding",
                str(disable_mtp).lower(),
            ]
        )
    )
    create_thinker_executor(
        **resolve_stage_factory_args(pipeline.stage_named("thinker"), pipeline)
    )
    server_args = infrastructure.call_args.args[0]
    options = resolved_view(server_args)
    target = ModelConfig.from_server_args(server_args, is_draft_model=False)
    assert target.hf_config.num_hidden_layers == 24
    assert target.hf_config.architectures == ["BailingMoeV3ForCausalLM"]
    if disable_mtp:
        assert options.speculative_algorithm is None
        assert "BailingMoeForCausalLMNextN" not in stages.ModelRegistry.models
    else:
        assert infrastructure.call_args.kwargs["supports_speculative_decoding"]
        assert (
            SpeculativeAlgorithm.from_string(
                options.speculative_algorithm
            ).create_worker(server_args)
            is EAGLEWorkerV2
        )
        assert options.enable_mixed_chunk is False
        assert options.speculative_draft_model_path == str(tmp_path)
        assert options.speculative_draft_model_revision == "snapshot"
        draft = ModelConfig.from_server_args(server_args, is_draft_model=True)
        assert draft.hf_config.architectures == ["BailingMoeForCausalLMNextN"]
        assert draft.hf_config.model_type == "bailing_hybrid"
        assert draft.hf_config.use_kda
        draft.hf_config.for_nextn_model = True
        assert draft.hf_config.layers_block_type == ["attention"]
        assert len(target.hf_config.layers_block_type) == 24
        assert (
            stages.ModelRegistry.models["BailingMoeForCausalLMNextN"]
            is stages.Ming3ASRMTP
        )
    statistics = {
        "spec_accept_rate": 0.5,
        "spec_accept_length": 2.0,
        "spec_num_correct_drafts": 1,
        "spec_num_proposed_drafts": 2,
        "spec_verify_ct": 1,
        "spec_correct_drafts_histogram": [0, 1, 0],
    }
    request = Ming3ASRRequestData(
        input_ids=torch.tensor([1, 2]),
        output_ids=[3, 4],
        spec_tokens_details=None if disable_mtp else statistics,
        stage_payload=StagePayload(
            request_id="transcript", request=OmniRequest(inputs="audio"), data={}
        ),
    )
    payload = scheduler.call_args.kwargs["result_adapter"](request)
    chunk = Client.default_result_builder(payload.request_id, payload.data)
    assert chunk.text == "hello"
    assert chunk.spec_tokens_details == request.spec_tokens_details


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("speculative_eagle_topk", "2", "topk=1"),
        ("speculative_draft_model_path", "other", "checkpoint"),
    ],
)
def test_invalid_mtp_options_fail_before_loading(
    option: str,
    value: str,
    message: str,
) -> None:
    manager = ConfigManager(Ming3ASRPipelineConfig(model_path="missing-checkpoint"))
    pipeline = manager.merge_config(
        manager.parse_extra_args(
            [
                "--thinker.engine.speculative_algorithm",
                "NEXTN",
                f"--thinker.engine.{option}",
                value,
            ]
        )
    )
    with pytest.raises(ValueError, match=message):
        create_thinker_executor(
            **resolve_stage_factory_args(pipeline.stage_named("thinker"), pipeline)
        )
