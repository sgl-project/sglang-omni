# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from sglang_omni.config import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    PlacementConfig,
    StageConfig,
)

_FACTORY = "tests.unit_test.fixtures.pipeline_fakes.dummy_factory"


def _stage(**kwargs) -> StageConfig:
    data = {
        "name": "stage",
        "process": "pipeline",
        "factory_path": _FACTORY,
        "terminal": True,
    }
    data.update(kwargs)
    cls = kwargs.pop("cls", None)
    if cls is not None:
        data.pop("cls", None)
        return cls(**data)
    return StageConfig(**data)


def _engine_stage(**kwargs) -> EngineStageConfig:
    data = {
        "name": "stage",
        "process": "pipeline",
        "factory_path": _FACTORY,
        "terminal": True,
    }
    data.update(kwargs)
    return EngineStageConfig(**data)


def test_stage_accepts_typed_values_in_every_consumer_group() -> None:
    stage = _engine_stage(
        gpu_memory_fraction=0.25,
        engine={"mem_fraction_static": 0.7},
        factory={"max_concurrency": 4, "max_seq_len": 8192, "video_fps": 2.0},
    )

    assert stage.gpu_memory_fraction == 0.25
    assert stage.engine.mem_fraction_static == 0.7
    assert stage.factory.max_concurrency == 4
    assert stage.factory.max_seq_len == 8192
    assert stage.factory.video_fps == 2.0


def test_invalid_gpu_memory_fraction_raises() -> None:
    with pytest.raises(ValueError, match="gpu_memory_fraction"):
        _stage(gpu_memory_fraction=0.0)


def test_invalid_engine_mem_fraction_static_raises() -> None:
    with pytest.raises(ValueError, match="mem_fraction_static"):
        EngineArgs(mem_fraction_static=1.0)


def test_invalid_model_group_values_raise() -> None:
    with pytest.raises(ValueError, match="max_seq_len"):
        FactoryArgs(max_seq_len=0)
    with pytest.raises(ValueError, match="video_fps"):
        FactoryArgs(video_fps=-1.0)


def test_prefill_coalesce_range_is_enforced_at_validation() -> None:
    assert FactoryArgs(prefill_coalesce_requests=32).prefill_coalesce_requests == 32
    assert FactoryArgs(prefill_coalesce_wait_ms=300).prefill_coalesce_wait_ms == 300.0

    with pytest.raises(ValueError, match="prefill_coalesce_requests"):
        FactoryArgs(prefill_coalesce_requests=-1)
    with pytest.raises(ValueError, match="prefill_coalesce_wait_ms"):
        FactoryArgs(prefill_coalesce_wait_ms=0.0)


def test_prefill_coalesce_requests_of_one_warns(caplog) -> None:
    with caplog.at_level("WARNING", logger="sglang_omni.config.schema"):
        FactoryArgs(prefill_coalesce_requests=1)
    assert "disables coalescing" in caplog.text


def test_the_engine_block_is_refused_off_engine_stages() -> None:
    with pytest.raises(ValueError, match="not an engine stage"):
        _stage(engine={"mem_fraction_static": 0.7})


def test_undeclared_group_keys_pass_through_to_the_consumer() -> None:
    """The group vocabularies belong to the modules that read them; the
    schema keeps unknown keys instead of guessing at their legality."""
    stage = _engine_stage(
        engine={"disable_radix_cache": True},
        factory={"made_up_knob": 7, "lookahead": 9},
    )

    assert stage.engine.overrides()["disable_radix_cache"] is True
    assert stage.factory.model_extra["made_up_knob"] == 7
    assert stage.factory.model_extra["lookahead"] == 9


def test_stage_rejects_terminal_with_next() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        PipelineConfig(
            model_path="dummy",
            stages=[
                _stage(name="source", next="sink", terminal=True),
                _stage(name="sink"),
            ],
        )


def test_tp_size_accepts_a_matching_gpu_list() -> None:
    stage = _stage(tp_size=2, gpu=[0, 1])

    assert stage.tp_size == 2


def test_tp_size_below_one_raises() -> None:
    with pytest.raises(ValueError, match="tp_size"):
        _stage(tp_size=0)


def test_pipeline_accepts_placement_config() -> None:
    config = PipelineConfig(
        model_path="dummy",
        placement=PlacementConfig(max_total_gpu_memory_fraction_per_gpu=0.95),
        stages=[_stage()],
    )

    assert config.placement.max_total_gpu_memory_fraction_per_gpu == 0.95


def test_invalid_placement_limit_raises() -> None:
    with pytest.raises(ValueError, match="max_total_gpu_memory_fraction_per_gpu"):
        PlacementConfig(max_total_gpu_memory_fraction_per_gpu=1.1)
