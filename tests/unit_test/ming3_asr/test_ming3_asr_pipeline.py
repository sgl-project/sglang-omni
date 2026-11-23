# SPDX-License-Identifier: Apache-2.0
"""Pipeline registration and backbone configuration contracts."""

import json
from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch
from safetensors.torch import save_file
from sglang.srt.configs.model_config import ModelConfig
from torch import nn
from transformers import AutoConfig, Qwen3Config

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.models.ming3_asr import mtp, stages
from sglang_omni.models.ming3_asr.config import (
    Ming3ASRFactoryArgs,
    Ming3ASRPipelineConfig,
)
from sglang_omni.models.ming3_asr.mtp import Ming3ASRMTP
from sglang_omni.models.ming3_asr.stages import Ming3ASRConfig, create_thinker_executor
from sglang_omni.models.model_capabilities import get_model_capabilities
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.modules.ming_audio.thinker import Ming3AudioMTP
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.serve.openai_errors import generation_error_status_code

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
    assert "SGLANG_DSPARK_USE_GREEDY_DRAFT" not in config.resolved_stage_env_defaults(
        "thinker"
    )
    factory_arguments = resolve_stage_factory_args(thinker, config)
    assert not any(
        name.startswith("speculative_")
        for name in factory_arguments["server_args_overrides"]
    )


def test_dspark_cli_options_reach_the_worker_environment() -> None:
    manager = ConfigManager(Ming3ASRPipelineConfig(model_path="unused-target"))
    config = manager.merge_config(
        manager.parse_extra_args(
            [
                "--thinker.engine.speculative_algorithm",
                "DSPARK",
                "--dspark-use-greedy-draft",
                "false",
                "--ragged-verify-mode",
                "compact",
            ]
        )
    )
    environment = config.resolved_stage_env_defaults("thinker")
    assert environment["SGLANG_DSPARK_USE_GREEDY_DRAFT"] == "0"
    assert environment["SGLANG_RAGGED_VERIFY_MODE"] == "compact"


def test_draft_path_requires_an_algorithm() -> None:
    manager = ConfigManager(Ming3ASRPipelineConfig(model_path="unused-target"))
    config = manager.merge_config(
        manager.parse_extra_args(
            ["--thinker.engine.speculative_draft_model_path", "example/another-draft"]
        )
    )
    factory_arguments = resolve_stage_factory_args(config.stages[1], config)
    assert (
        "speculative_draft_model_revision"
        not in factory_arguments["server_args_overrides"]
    )
    with pytest.raises(ValueError, match="needs speculative_algorithm"):
        create_thinker_executor(**factory_arguments)


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


def test_checkpoint_config_loads_from_a_hub_model_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "config.json").write_text('{"text_config": {"hidden_size": 64}}')
    hub_lookup = MagicMock(return_value=str(tmp_path / "config.json"))
    monkeypatch.setattr(stages, "cached_file", hub_lookup)

    configuration = stages.load_checkpoint_config("inclusionAI/Ming-3.0-asr-tiny")

    assert configuration == {"text_config": {"hidden_size": 64}}
    hub_lookup.assert_called_once_with("inclusionAI/Ming-3.0-asr-tiny", "config.json")


def test_checkpoint_config_loads_from_a_local_directory(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"text_config": {"hidden_size": 64}}')

    assert stages.load_checkpoint_config(str(tmp_path)) == {
        "text_config": {"hidden_size": 64}
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


@pytest.fixture
def mock_thinker_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
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
        "Ming3ASRPreLMEncoderService",
        "build_sglang_server_args",
        "AutoConfig",
        "PreTrainedTokenizerFast",
        "Ming3AudioModelRunner",
        "Ming3ASRStreamOutputBuilder",
        "OmniScheduler",
    ):
        monkeypatch.setattr(stages, name, MagicMock())
    monkeypatch.setattr(stages.ModelRegistry, "models", {})
    infrastructure = MagicMock(return_value=tuple(MagicMock() for _ in range(5)))
    infrastructure.return_value[-1].vocab_size = 16
    monkeypatch.setattr(stages, "create_sglang_infrastructure", infrastructure)


@pytest.mark.usefixtures("mock_thinker_runtime")
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
    manager = ConfigManager(Ming3ASRPipelineConfig(model_path="target-checkpoint"))
    config = manager.merge_config(
        manager.parse_extra_args(
            [
                "--thinker.engine.speculative_algorithm",
                "DSPARK",
                "--thinker.engine.speculative_draft_model_path",
                "wirybeaver/Ming-3.0-asr-tiny-dspark",
            ]
        )
    )
    factory_arguments = resolve_stage_factory_args(config.stages[1], config)
    scheduler = create_thinker_executor(**factory_arguments)
    overrides = stages.build_sglang_server_args.call_args.kwargs
    draft_config = ModelConfig(
        overrides["speculative_draft_model_path"],
        model_override_args=overrides.get("json_model_override_args", "{}"),
        is_draft_model=True,
        speculative_algorithm="DSPARK",
    )
    assert draft_config.hf_config.architectures == ["ReducedVocabularyDSparkDraftModel"]

    payload = StagePayload(
        request_id="greedy-request",
        request=OmniRequest(inputs=None, params={"temperature": 0.7}),
        data={
            "features": [torch.zeros(1)],
            "input_ids": torch.tensor([1, 7]),
            "audio_positions": torch.tensor([1]),
            "audio_embeddings": torch.zeros(1, HIDDEN_SIZE),
        },
    )
    with pytest.raises(ValueError, match="requires temperature=0") as rejected:
        stages.OmniScheduler.call_args.kwargs["request_builder"](payload)
    assert generation_error_status_code(RuntimeError(str(rejected.value))) == 400
    stages.Ming3ASRPreLMEncoderService.return_value.encode.assert_not_called()
    payload.data["features"] = []
    payload.request.params = {}
    greedy_request = stages.OmniScheduler.call_args.kwargs["request_builder"](payload)
    assert greedy_request.req.sampling_params.top_k == 1

    scheduler.shutdown_callback()
    assert not checkpoint_path.exists()
    stages.Ming3ASRPreLMEncoderService.return_value.close.assert_called_once_with()


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
    monkeypatch.setattr(
        stages, "cached_file", MagicMock(side_effect=FileNotFoundError("config.json"))
    )
    manager = ConfigManager(
        Ming3ASRPipelineConfig(model_path="missing-target-checkpoint")
    )
    config = manager.merge_config(
        manager.parse_extra_args(
            [
                "--thinker.engine.speculative_algorithm",
                "DSPARK",
                "--thinker.engine.speculative_draft_model_path",
                "wirybeaver/Ming-3.0-asr-tiny-dspark",
            ]
        )
    )
    factory_arguments = resolve_stage_factory_args(config.stages[1], config)
    with pytest.raises(FileNotFoundError):
        create_thinker_executor(**factory_arguments)
    assert not checkpoint_path.exists()


@pytest.mark.usefixtures("mock_thinker_runtime")
@pytest.mark.parametrize("draft_path", [None, "target-checkpoint", "another/asr-mtp"])
def test_nextn_uses_target_checkpoint_with_asr_draft(
    draft_path: str | None, tmp_path: Path
) -> None:
    pipeline = Ming3ASRPipelineConfig(model_path="target-checkpoint")
    arguments = resolve_stage_factory_args(pipeline.stage_named("thinker"), pipeline)
    engine = {"speculative_algorithm": "NEXTN", "revision": "target-revision"}
    if draft_path is not None:
        engine["speculative_draft_model_path"] = draft_path
    if draft_path == "another/asr-mtp":
        engine["speculative_draft_model_revision"] = "draft-revision"
    arguments.update(device="cpu", server_args_overrides=engine)
    create_thinker_executor(**arguments)
    resolved = stages.build_sglang_server_args.call_args.kwargs
    assert resolved["speculative_draft_model_path"] == pipeline.model_path
    assert resolved["speculative_draft_model_revision"] == "target-revision"
    draft_class = stages.ModelRegistry.models["BailingMoeForCausalLMNextN"]
    if draft_path == "another/asr-mtp":
        assert draft_class is Ming3ASRMTP
        AutoConfig.register(Ming3ASRConfig.model_type, Ming3ASRConfig, exist_ok=True)
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": Ming3ASRConfig.model_type,
                    "text_config": {
                        "short_conv_kernel_size": 4,
                        "num_nextn_predict_layers": 1,
                    },
                }
            )
        )
        config = ModelConfig(
            str(tmp_path),
            is_draft_model=True,
            speculative_algorithm="EAGLE",
            model_override_args=resolved["json_model_override_args"],
        ).hf_config
        assert config.architectures == ["BailingMoeForCausalLMNextN"]
        assert config.mtp_weights_path == draft_path
        assert config.mtp_weights_revision == "draft-revision"
    else:
        assert draft_class is Ming3AudioMTP
    assert stages.create_sglang_infrastructure.call_args.kwargs[
        "supports_speculative_decoding"
    ]


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        (
            {"speculative_algorithm": "NEXTN", "speculative_eagle_topk": 4},
            "speculative_eagle_topk=4",
        ),
        (
            {
                "speculative_algorithm": "NEXTN",
                "json_model_override_args": '{"architectures": ["OtherModel"]}',
            },
            "architectures=.*OtherModel",
        ),
        ({"speculative_algorithm": "EAGLE3"}, "DSPARK and NEXTN"),
    ],
)
def test_unsupported_speculative_configuration_is_rejected(
    overrides: dict[str, str | int], error: str
) -> None:
    pipeline = Ming3ASRPipelineConfig(model_path="unused-target")
    arguments = resolve_stage_factory_args(pipeline.stage_named("thinker"), pipeline)
    arguments.update(device="cpu", server_args_overrides=overrides)
    with pytest.raises(ValueError, match=error):
        create_thinker_executor(**arguments)


@pytest.mark.usefixtures("mock_thinker_runtime")
def test_invalid_sampling_parameters_return_bad_request() -> None:
    pipeline = Ming3ASRPipelineConfig(model_path="target-checkpoint")
    arguments = resolve_stage_factory_args(pipeline.stage_named("thinker"), pipeline)
    arguments["device"] = "cpu"
    create_thinker_executor(**arguments)
    payload = StagePayload(
        request_id="invalid-temperature",
        request=OmniRequest(inputs=None, params={"temperature": float("inf")}),
        data={"features": []},
    )

    with pytest.raises(ValueError, match="temperature") as rejected:
        stages.OmniScheduler.call_args.kwargs["request_builder"](payload)
    assert generation_error_status_code(RuntimeError(str(rejected.value))) == 400


def test_asr_mtp_loads_fine_tuned_prediction_weights_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ASR draft replaces the prediction block and retains shared verifier weights."""
    prediction_name = "model.layers.24.eh_proj.weight"
    fine_tuned = torch.full((2, 4), 3.0)
    checkpoint = tmp_path / "model.safetensors"
    save_file({prediction_name: fine_tuned}, checkpoint)
    download = Mock(return_value=str(checkpoint))
    monkeypatch.setattr(mtp, "cached_file", download)
    shared = {
        "model.word_embeddings.weight": torch.ones(4, 2),
        "lm_head.weight": torch.ones(4, 2),
        "model.layers.23.input_layernorm.weight": torch.ones(2),
    }
    loaded: dict[str, torch.Tensor] = {}

    def load_native(
        self: Ming3AudioMTP, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> None:
        loaded.update(weights)

    monkeypatch.setattr(Ming3AudioMTP, "load_weights", load_native)
    draft = Ming3ASRMTP.__new__(Ming3ASRMTP)
    nn.Module.__init__(draft)
    draft.config = SimpleNamespace(
        num_hidden_layers=24,
        mtp_weights_path="another/asr-mtp",
        mtp_weights_revision="draft-revision",
    )
    draft.load_weights(iter([*shared.items(), (prediction_name, torch.zeros(2, 4))]))
    assert set(loaded) == {*shared, prediction_name}
    assert all(loaded[name] is weight for name, weight in shared.items())
    torch.testing.assert_close(loaded[prediction_name], fine_tuned)
    download.assert_called_once_with(
        "another/asr-mtp",
        "model.safetensors",
        revision="draft-revision",
    )


def test_asr_mtp_rejects_unused_local_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    projection = "model.layers.24.eh_proj.weight"
    save_file(
        {projection: torch.ones(1), "model.layers.24.extra.weight": torch.ones(1)},
        tmp_path / "model.safetensors",
    )
    monkeypatch.setattr(
        Ming3AudioMTP, "load_weights", lambda self, weights: dict(weights)
    )
    draft = Ming3ASRMTP.__new__(Ming3ASRMTP)
    nn.Module.__init__(draft)
    draft.config = SimpleNamespace(
        num_hidden_layers=24, mtp_weights_path=str(tmp_path), mtp_weights_revision=None
    )
    with pytest.raises(ValueError, match="model.layers.24.extra.weight"):
        draft.load_weights([(projection, torch.zeros(1))])
