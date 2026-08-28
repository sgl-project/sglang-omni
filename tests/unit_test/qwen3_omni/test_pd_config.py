# SPDX-License-Identifier: Apache-2.0
"""Qwen PD topology is declared directly, not synthesized by the compiler."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.config import resolve_stage_factory_args
from sglang_omni.models.qwen3_omni import bootstrap
from sglang_omni.models.qwen3_omni.config import (
    Qwen3OmniPDPipelineConfig,
    Qwen3OmniSpeechPDPipelineConfig,
)
from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime


def _stages(config):
    return {stage.name: stage for stage in config.stages}


@pytest.mark.parametrize(
    ("config_cls", "expected_names"),
    [
        (
            Qwen3OmniPDPipelineConfig,
            [
                "preprocessing",
                "image_encoder",
                "audio_encoder",
                "mm_aggregate",
                "thinker_prefill",
                "thinker_decode",
                "decode",
            ],
        ),
        (
            Qwen3OmniSpeechPDPipelineConfig,
            [
                "preprocessing",
                "image_encoder",
                "audio_encoder",
                "thinker_prefill",
                "thinker_decode",
                "decode",
                "talker_ar",
                "code2wav",
            ],
        ),
    ],
)
def test_pd_configs_declare_both_engine_stages(config_cls, expected_names) -> None:
    config = config_cls(model_path="dummy")
    stages = _stages(config)

    assert [stage.name for stage in config.stages] == expected_names
    assert "thinker" not in stages

    prefill = stages["thinker_prefill"]
    decode = stages["thinker_decode"]
    assert prefill.next == "thinker_decode"
    assert decode.next == "decode"
    assert prefill.gpu == 0
    assert decode.gpu == 1
    assert prefill.factory.model_extra["scheduler_role"] == "prefill"
    assert decode.factory.model_extra["scheduler_role"] == "decode"
    assert prefill.factory.enable_async_decode is False
    assert decode.factory.enable_async_decode is True
    assert prefill.engine.disable_radix_cache is True
    assert decode.engine.disable_radix_cache is True
    assert prefill.engine.model_extra["page_size"] == 1
    assert decode.engine.model_extra["page_size"] == 1


def test_pd_stage_ownership_is_explicit_in_the_graph() -> None:
    config = Qwen3OmniSpeechPDPipelineConfig(model_path="dummy")
    stages = _stages(config)
    prefill = stages["thinker_prefill"]
    decode = stages["thinker_decode"]

    assert prefill.wait_for == ["preprocessing", "image_encoder", "audio_encoder"]
    assert prefill.route_fn is None
    assert prefill.project_payload == {}
    assert prefill.stream_done_to_fn is None

    assert decode.wait_for is None
    assert decode.route_fn.endswith(".resolve_thinker_next_stages")
    assert decode.project_payload == {
        "decode": (
            "sglang_omni.models.qwen3_omni.request_builders.project_thinker_to_decode"
        )
    }
    assert decode.stream_done_to_fn.endswith(".resolve_thinker_stream_done_targets")

    for upstream in ("preprocessing", "image_encoder", "audio_encoder"):
        assert "thinker_prefill" in stages[upstream].next
        assert "thinker" not in stages[upstream].next


def test_ordinary_compiler_preserves_explicit_pd_stage_names() -> None:
    config = Qwen3OmniPDPipelineConfig(model_path="dummy")
    prep = prepare_pipeline_runtime(config)
    try:
        assert [stage.name for stage in prep.stages_cfg] == [
            stage.name for stage in config.stages
        ]
        assert set(prep.endpoints) >= {
            "stage_thinker_prefill",
            "stage_thinker_decode",
        }
    finally:
        prep.runtime_dir.close()


def test_pd_factory_role_is_a_strict_constructor_argument() -> None:
    config = Qwen3OmniPDPipelineConfig(model_path="dummy")
    prefill = _stages(config)["thinker_prefill"]

    args = resolve_stage_factory_args(prefill, config)

    assert args["scheduler_role"] == "prefill"


@pytest.mark.parametrize(
    ("stage_name", "path", "value", "message"),
    [
        ("thinker_prefill", "engine.page_size", 2, "page_size=1"),
        (
            "thinker_decode",
            "engine.disable_radix_cache",
            False,
            "disable_radix_cache=true",
        ),
        ("thinker_decode", "factory.scheduler_role", "prefill", "scheduler_role"),
        ("thinker_decode", "gpu", 0, "separate GPUs"),
    ],
)
def test_pd_config_rejects_unsupported_runtime_overrides(
    stage_name, path, value, message
) -> None:
    config = Qwen3OmniPDPipelineConfig(model_path="dummy")
    data = config.model_dump()
    stage = next(entry for entry in data["stages"] if entry["name"] == stage_name)
    target = stage
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value

    with pytest.raises(ValueError, match=message):
        Qwen3OmniPDPipelineConfig(**data)


def test_thinker_factory_selects_scheduler_class_at_construction(monkeypatch) -> None:
    import sglang_omni.scheduling as scheduling

    classes = SimpleNamespace(
        OmniScheduler=type("Regular", (), {}),
    )
    pd_classes = SimpleNamespace(
        OmniPrefillScheduler=type("Prefill", (), {}),
        OmniDecodeScheduler=type("Decode", (), {}),
    )
    monkeypatch.setattr(scheduling, "omni_scheduler", classes, raising=False)
    monkeypatch.setattr(scheduling, "pd_scheduler", pd_classes, raising=False)

    assert bootstrap._thinker_scheduler_cls(None) is classes.OmniScheduler
    assert (
        bootstrap._thinker_scheduler_cls("prefill") is pd_classes.OmniPrefillScheduler
    )
    assert bootstrap._thinker_scheduler_cls("decode") is pd_classes.OmniDecodeScheduler
