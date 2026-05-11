# SPDX-License-Identifier: Apache-2.0
"""V1 SGLang AR memory override and autosizing tests."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import typer

import sglang_omni_v1.models.qwen3_omni.stages as qwen_stages
from sglang_omni_v1.cli.serve import (
    apply_encoder_mem_reserve_cli_override,
    apply_mem_fraction_cli_overrides,
)
from sglang_omni_v1.config import PipelineConfig, StageConfig
from sglang_omni_v1.models.qwen3_omni.config import (
    Qwen3OmniPipelineConfig,
    Qwen3OmniSpeechPipelineConfig,
)
from sglang_omni_v1.scheduling.sglang_backend.server_args_builder import (
    apply_encoder_mem_reserve,
    build_sglang_server_args,
)


def _stage(config, name: str):
    return next(stage for stage in config.stages if stage.name == name)


def _server_args_overrides(config, name: str) -> dict:
    return _stage(config, name).factory_args.get("server_args_overrides", {})


def test_builder_omits_mem_fraction_static_by_default() -> None:
    server_args = build_sglang_server_args(
        "dummy",
        context_length=8192,
        tp_size=2,
        random_seed=777,
    )

    assert server_args.mem_fraction_static is None
    assert server_args.context_length == 8192
    assert server_args.tp_size == 2
    assert server_args.random_seed == 777


def test_builder_forwards_explicit_mem_fraction_static() -> None:
    server_args = build_sglang_server_args(
        "dummy",
        context_length=4096,
        mem_fraction_static=0.82,
        dtype="bfloat16",
    )

    assert server_args.mem_fraction_static == 0.82
    assert server_args.dtype == "bfloat16"


def test_encoder_mem_reserve_applies_only_to_valid_auto_values() -> None:
    server_args = SimpleNamespace(mem_fraction_static=0.929)

    apply_encoder_mem_reserve(server_args, 0.05)

    assert server_args.mem_fraction_static == 0.879

    apply_encoder_mem_reserve(server_args, 0.0)
    assert server_args.mem_fraction_static == 0.879

    with pytest.raises(ValueError, match="below the safe floor"):
        apply_encoder_mem_reserve(SimpleNamespace(mem_fraction_static=0.15), 0.10)


def test_cli_global_and_specific_mem_fraction_target_only_qwen_ar_stages() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    apply_mem_fraction_cli_overrides(
        config,
        mem_fraction_static=0.80,
        thinker_mem_fraction_static=0.70,
        talker_mem_fraction_static=0.65,
    )

    assert _server_args_overrides(config, "thinker")["mem_fraction_static"] == 0.70
    assert _server_args_overrides(config, "talker_ar")["mem_fraction_static"] == 0.65
    for non_ar_stage in ("image_encoder", "audio_encoder", "code2wav"):
        assert "server_args_overrides" not in _stage(config, non_ar_stage).factory_args


def test_cli_per_role_mem_fraction_overrides_global_when_all_three_passed() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    apply_mem_fraction_cli_overrides(
        config,
        mem_fraction_static=0.80,
        thinker_mem_fraction_static=0.70,
        talker_mem_fraction_static=0.65,
    )

    assert _server_args_overrides(config, "thinker")["mem_fraction_static"] == 0.70
    assert _server_args_overrides(config, "talker_ar")["mem_fraction_static"] == 0.65


def test_cli_global_mem_fraction_applies_when_no_per_role_override() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    apply_mem_fraction_cli_overrides(
        config,
        mem_fraction_static=0.80,
        thinker_mem_fraction_static=None,
        talker_mem_fraction_static=None,
    )

    assert _server_args_overrides(config, "thinker")["mem_fraction_static"] == 0.80
    assert _server_args_overrides(config, "talker_ar")["mem_fraction_static"] == 0.80


def test_cli_partial_per_role_falls_back_to_global_for_unspecified_role() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    apply_mem_fraction_cli_overrides(
        config,
        mem_fraction_static=0.80,
        thinker_mem_fraction_static=0.70,
        talker_mem_fraction_static=None,
    )

    assert _server_args_overrides(config, "thinker")["mem_fraction_static"] == 0.70
    assert _server_args_overrides(config, "talker_ar")["mem_fraction_static"] == 0.80


def test_cli_talker_per_role_overrides_global_thinker_falls_back() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    apply_mem_fraction_cli_overrides(
        config,
        mem_fraction_static=0.80,
        thinker_mem_fraction_static=None,
        talker_mem_fraction_static=0.65,
    )

    assert _server_args_overrides(config, "thinker")["mem_fraction_static"] == 0.80
    assert _server_args_overrides(config, "talker_ar")["mem_fraction_static"] == 0.65


def test_cli_mem_fraction_static_survives_runtime_overrides_overlay() -> None:
    from sglang_omni_v1.config.compiler import _resolve_factory_args

    config = Qwen3OmniSpeechPipelineConfig(
        model_path="dummy",
        runtime_overrides={
            "thinker": {"server_args_overrides": {"disable_cuda_graph": True}}
        },
    )

    apply_mem_fraction_cli_overrides(
        config,
        mem_fraction_static=0.80,
        thinker_mem_fraction_static=None,
        talker_mem_fraction_static=None,
    )

    resolved = _resolve_factory_args(_stage(config, "thinker"), config)

    assert resolved["server_args_overrides"]["mem_fraction_static"] == 0.80
    assert resolved["server_args_overrides"]["disable_cuda_graph"] is True


def test_cli_rejects_talker_override_on_text_only_qwen_without_partial_write() -> None:
    config = Qwen3OmniPipelineConfig(model_path="dummy")
    original = config.model_dump()

    with pytest.raises(typer.BadParameter, match="talker"):
        apply_mem_fraction_cli_overrides(
            config,
            mem_fraction_static=None,
            thinker_mem_fraction_static=None,
            talker_mem_fraction_static=0.65,
        )

    assert config.model_dump() == original


def test_cli_rejects_invalid_mem_fraction_without_partial_write() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    original = config.model_dump()

    with pytest.raises(typer.BadParameter, match="must be > 0 and < 1"):
        apply_mem_fraction_cli_overrides(
            config,
            mem_fraction_static=1.0,
            thinker_mem_fraction_static=None,
            talker_mem_fraction_static=None,
        )

    assert config.model_dump() == original


def test_cli_rejects_global_mem_fraction_when_pipeline_has_no_supported_roles() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(
                name="preprocessing",
                factory=(
                    "sglang_omni_v1.models.qwen3_omni.stages."
                    "create_preprocessing_executor"
                ),
                terminal=True,
            )
        ],
    )

    with pytest.raises(typer.BadParameter, match="supported"):
        apply_mem_fraction_cli_overrides(
            config,
            mem_fraction_static=0.80,
            thinker_mem_fraction_static=None,
            talker_mem_fraction_static=None,
        )


def test_cli_encoder_mem_reserve_routes_as_thinker_factory_arg() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    apply_encoder_mem_reserve_cli_override(
        config,
        encoder_mem_reserve=0.15,
        mem_fraction_static=None,
        thinker_mem_fraction_static=None,
    )

    thinker_args = _stage(config, "thinker").factory_args
    assert thinker_args["encoder_mem_reserve"] == 0.15
    assert "encoder_mem_reserve" not in thinker_args.get("server_args_overrides", {})
    assert "encoder_mem_reserve" not in _stage(config, "talker_ar").factory_args


def test_cli_encoder_mem_reserve_is_exclusive_with_thinker_auto_path_pins() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    with pytest.raises(typer.BadParameter, match="mutually exclusive"):
        apply_encoder_mem_reserve_cli_override(
            config,
            encoder_mem_reserve=0.15,
            mem_fraction_static=0.80,
            thinker_mem_fraction_static=None,
        )

    with pytest.raises(typer.BadParameter, match="mutually exclusive"):
        apply_encoder_mem_reserve_cli_override(
            config,
            encoder_mem_reserve=0.15,
            mem_fraction_static=None,
            thinker_mem_fraction_static=0.70,
        )


def test_cli_encoder_mem_reserve_rejects_config_pinned_thinker_mem_fraction() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    thinker_args = _stage(config, "thinker").factory_args
    thinker_args["server_args_overrides"] = {"mem_fraction_static": 0.70}

    with pytest.raises(typer.BadParameter, match="not explicitly pinned"):
        apply_encoder_mem_reserve_cli_override(
            config,
            encoder_mem_reserve=0.15,
            mem_fraction_static=None,
            thinker_mem_fraction_static=None,
        )


def test_cli_encoder_mem_reserve_rejects_runtime_pinned_thinker_mem_fraction() -> None:
    config = Qwen3OmniSpeechPipelineConfig(
        model_path="dummy",
        runtime_overrides={
            "thinker": {"server_args_overrides": {"mem_fraction_static": 0.70}}
        },
    )

    with pytest.raises(typer.BadParameter, match="not explicitly pinned"):
        apply_encoder_mem_reserve_cli_override(
            config,
            encoder_mem_reserve=0.15,
            mem_fraction_static=None,
            thinker_mem_fraction_static=None,
        )


def test_cli_encoder_mem_reserve_survives_runtime_overrides_overlay() -> None:
    from sglang_omni_v1.config.compiler import _resolve_factory_args

    config = Qwen3OmniSpeechPipelineConfig(
        model_path="dummy",
        runtime_overrides={"thinker": {"encoder_mem_reserve": 0.10}},
    )

    apply_encoder_mem_reserve_cli_override(
        config,
        encoder_mem_reserve=0.15,
        mem_fraction_static=None,
        thinker_mem_fraction_static=None,
    )

    resolved = _resolve_factory_args(_stage(config, "thinker"), config)

    assert resolved["encoder_mem_reserve"] == 0.15


def test_qwen_thinker_auto_path_applies_encoder_reserve() -> None:
    server_args = SimpleNamespace(mem_fraction_static=0.929)

    applied = qwen_stages._apply_qwen_thinker_encoder_reserve(
        server_args,
        has_explicit_mem_fraction_static=False,
        encoder_mem_reserve=0.05,
    )

    assert applied is True
    assert server_args.mem_fraction_static == 0.879


def test_qwen_thinker_explicit_pin_bypasses_encoder_reserve() -> None:
    server_args = SimpleNamespace(mem_fraction_static=0.70)

    applied = qwen_stages._apply_qwen_thinker_encoder_reserve(
        server_args,
        has_explicit_mem_fraction_static=True,
        encoder_mem_reserve=0.20,
    )

    assert applied is False
    assert server_args.mem_fraction_static == 0.70


def test_qwen_factory_signatures_keep_reserve_thinker_only() -> None:
    thinker_sig = inspect.signature(
        qwen_stages.create_sglang_thinker_executor_from_config
    )
    talker_sig = inspect.signature(qwen_stages.create_talker_ar_executor_from_config)

    assert thinker_sig.parameters["encoder_mem_reserve"].default == 0.05
    assert "encoder_mem_reserve" not in talker_sig.parameters
