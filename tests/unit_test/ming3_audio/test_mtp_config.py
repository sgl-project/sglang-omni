# SPDX-License-Identifier: Apache-2.0
"""Ming MTP configuration and native worker selection."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import JsonValue
from sglang.srt.arg_groups import memory_hook
from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.models.ming3_audio import stages
from sglang_omni.models.ming3_audio.config import Ming3AudioPipelineConfig


@pytest.fixture(autouse=True)
def isolate_model_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        stages.ModelRegistry, "models", dict(stages.ModelRegistry.models)
    )


@pytest.mark.parametrize(
    ("cli_flags", "expected_algorithm", "expected_async_decode"),
    [
        ([], None, False),
        (["--thinker.engine.speculative_algorithm", "NEXTN"], "EAGLE", False),
        (
            [
                "--thinker.factory.enable_async_decode",
                "true",
            ],
            None,
            True,
        ),
        (
            [
                "--thinker.engine.speculative_algorithm",
                "NEXTN",
                "--thinker.engine.disable_speculative_decoding",
                "true",
                "--thinker.factory.enable_async_decode",
                "true",
            ],
            None,
            True,
        ),
    ],
    ids=["plain", "mtp", "plain-async", "disabled"],
)
def test_factory_builds_isolated_native_mtp_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cli_flags: list[str],
    expected_algorithm: str | None,
    expected_async_decode: bool,
) -> None:
    """The public flags must select the V3 MTP without changing the target depth."""

    checkpoint = Path(__file__).parent / "fixtures" / "config.json"
    (tmp_path / "config.json").write_text(checkpoint.read_text())
    monkeypatch.setattr(
        memory_hook, "get_device_memory_capacity", lambda device: 80 * 1024
    )
    monkeypatch.delitem(
        stages.ModelRegistry.models, "BailingMoeForCausalLMNextN", raising=False
    )

    class InfrastructureReached(Exception):
        pass

    infrastructure = Mock(side_effect=InfrastructureReached)
    monkeypatch.setattr(stages, "create_sglang_infrastructure", infrastructure)
    manager = ConfigManager(Ming3AudioPipelineConfig(model_path=str(tmp_path)))
    pipeline = manager.merge_config(
        manager.parse_extra_args(
            [
                "--thinker.factory.device",
                "cpu",
                "--thinker.engine.device",
                "cpu",
                "--thinker.engine.revision",
                "test-revision",
                *cli_flags,
            ]
        )
    )
    with pytest.raises(InfrastructureReached):
        stages.create_thinker_executor(
            **resolve_stage_factory_args(pipeline.stage_named("thinker"), pipeline)
        )

    server_args = infrastructure.call_args.args[0]
    configuration = resolved_view(server_args)
    target = ModelConfig.from_server_args(server_args, is_draft_model=False)
    assert target.hf_config.architectures == ["BailingMoeV3ForCausalLM"]
    assert target.hf_config.num_hidden_layers == 42
    assert target.hf_config.num_nextn_predict_layers == 1
    assert (
        infrastructure.call_args.kwargs["enable_async_decode"] is expected_async_decode
    )
    if expected_algorithm is None:
        assert configuration.speculative_algorithm is None
        assert target.hf_config.model_type == "bailing_moe_v3_audio"
        assert "BailingMoeForCausalLMNextN" not in stages.ModelRegistry.models
    else:
        assert infrastructure.call_args.kwargs["supports_speculative_decoding"]
        assert configuration.speculative_algorithm == expected_algorithm
        assert (
            SpeculativeAlgorithm.from_string(expected_algorithm).create_worker(
                server_args
            )
            is EAGLEWorkerV2
        )
        assert configuration.speculative_draft_model_path == str(tmp_path)
        assert configuration.speculative_draft_model_revision == "test-revision"
        assert configuration.speculative_eagle_topk == 1
        draft = ModelConfig.from_server_args(server_args, is_draft_model=True)
        assert draft.hf_config.architectures == ["BailingMoeForCausalLMNextN"]
        assert draft.hf_config.model_type == "bailing_hybrid"
        assert draft.hf_config.use_kda
        assert draft.hf_config is not target.hf_config
        draft.hf_config.for_nextn_model = True
        assert draft.hf_config.layers_block_type == ["attention"]
        assert len(target.hf_config.layers_block_type) == 42
        assert target.hf_config.layers_block_type.count("linear_attention") == 35
        assert target.hf_config.architectures == ["BailingMoeV3ForCausalLM"]
        assert (
            stages.ModelRegistry.models["BailingMoeForCausalLMNextN"]
            is stages.Ming3AudioMTP
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"speculative_algorithm": "DFLASH"}, "bundled MTP"),
        ({"speculative_algorithm": "EAGLE3"}, "bundled MTP"),
        (
            {"speculative_algorithm": None, "speculative_draft_model_path": "other"},
            "needs speculative_algorithm",
        ),
        ({"speculative_draft_model_path": "other"}, "checkpoint"),
        ({"speculative_draft_model_revision": "other"}, "revision"),
        ({"speculative_eagle_topk": 2}, "topk=1"),
        ({"enable_multi_layer_eagle": True}, "one MTP layer"),
        ({"enable_mixed_chunk": True}, "mixed_chunk off"),
    ],
)
def test_invalid_mtp_options_fail_before_reading_weights(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, JsonValue],
    message: str,
) -> None:
    monkeypatch.setattr(
        stages.Ming3AudioConfig,
        "from_pretrained",
        Mock(side_effect=AssertionError("Invalid options reached checkpoint loading")),
    )
    with pytest.raises(ValueError, match=message):
        stages.create_thinker_executor(
            "missing-checkpoint",
            max_seq_len=8192,
            enable_async_decode=False,
            device="cpu",
            server_args_overrides={"speculative_algorithm": "NEXTN", **overrides},
        )


@pytest.mark.parametrize(
    "changed_fields",
    [
        {"num_nextn_predict_layers": 0},
        {"num_nextn_predict_layers": 2},
        {"mtp_use_kda": True},
        {"short_conv_kernel_size": None},
    ],
)
def test_checkpoint_without_supported_mtp_fails_before_weight_loading(
    tmp_path: Path, changed_fields: dict[str, JsonValue]
) -> None:
    checkpoint = json.loads(
        (Path(__file__).parent / "fixtures" / "config.json").read_text()
    )
    checkpoint["text_config"].update(changed_fields)
    (tmp_path / "config.json").write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match="one MLA MTP layer"):
        stages.create_thinker_executor(
            str(tmp_path),
            max_seq_len=8192,
            enable_async_decode=False,
            device="cpu",
            server_args_overrides={"speculative_algorithm": "NEXTN"},
        )
