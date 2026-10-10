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

FACTORY = "tests.unit_test.fixtures.pipeline_fakes.dummy_factory"


def make_stage(**kwargs) -> StageConfig:
    data = {
        "name": "stage",
        "process": "pipeline",
        "factory_path": FACTORY,
        "terminal": True,
    }
    data.update(kwargs)
    cls = kwargs.pop("cls", None)
    if cls is not None:
        data.pop("cls", None)
        return cls(**data)
    return StageConfig(**data)


def engine_stage(**kwargs) -> EngineStageConfig:
    data = {
        "name": "stage",
        "process": "pipeline",
        "factory_path": FACTORY,
        "terminal": True,
    }
    data.update(kwargs)
    return EngineStageConfig(**data)


def test_stage_accepts_typed_values_in_every_consumer_group() -> None:
    stage = engine_stage(
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
        make_stage(gpu_memory_fraction=0.0)


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
        make_stage(engine={"mem_fraction_static": 0.7})


def test_undeclared_group_keys_pass_through_to_the_consumer() -> None:
    """The group vocabularies belong to the modules that read them; the
    schema keeps unknown keys instead of guessing at their legality."""
    stage = engine_stage(
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
                make_stage(name="source", next="sink", terminal=True),
                make_stage(name="sink"),
            ],
        )


def test_tp_size_accepts_a_matching_gpu_list() -> None:
    stage = make_stage(tp_size=2, gpu=[0, 1])

    assert stage.tp_size == 2


def test_tp_size_below_one_raises() -> None:
    with pytest.raises(ValueError, match="tp_size"):
        make_stage(tp_size=0)


def test_pipeline_accepts_placement_config() -> None:
    config = PipelineConfig(
        model_path="dummy",
        placement=PlacementConfig(max_total_gpu_memory_fraction_per_gpu=0.95),
        stages=[make_stage()],
    )

    assert config.placement.max_total_gpu_memory_fraction_per_gpu == 0.95


def test_invalid_placement_limit_raises() -> None:
    with pytest.raises(ValueError, match="max_total_gpu_memory_fraction_per_gpu"):
        PlacementConfig(max_total_gpu_memory_fraction_per_gpu=1.1)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (2 * 1024**3, 2 * 1024**3),
        ("512KiB", 512 * 1024),
        ("128MiB", 128 * 1024**2),
        ("2GiB", 2 * 1024**3),
        ("1TiB", 1024**4),
    ],
)
def test_engine_kv_cache_bytes_parses_sizes(value, expected) -> None:
    from sglang_omni.config import EngineArgs

    assert EngineArgs(kv_cache_bytes=value).kv_cache_bytes == expected


@pytest.mark.parametrize("value", [0, -1, True, "2GB", "1.5GiB", "GiB", " 2GiB"])
def test_engine_kv_cache_bytes_rejects_bad_values(value) -> None:
    from sglang_omni.config import EngineArgs

    with pytest.raises(ValueError):
        EngineArgs(kv_cache_bytes=value)


def test_engine_kv_cache_bytes_rejects_mem_fraction_static() -> None:
    from sglang_omni.config import EngineArgs

    with pytest.raises(ValueError, match="keep exactly one|cannot be set together"):
        EngineArgs(kv_cache_bytes="2GiB", mem_fraction_static=0.7)


def test_stage_total_reserve_rejects_gpu_memory_fraction() -> None:
    from sglang_omni.config import StageConfig

    with pytest.raises(ValueError, match="same budget in two units"):
        StageConfig(
            name="a",
            factory_path="x.y",
            terminal=True,
            gpu_memory_fraction=0.5,
            total_reserve_bytes="8GiB",
        )


def test_stage_rejects_kv_above_total_reserve() -> None:
    from sglang_omni.config import EngineArgs, EngineStageConfig

    with pytest.raises(ValueError, match="must not exceed"):
        EngineStageConfig(
            name="a",
            factory_path="x.y",
            terminal=True,
            total_reserve_bytes="2GiB",
            engine=EngineArgs(kv_cache_bytes="4GiB"),
        )


def test_engine_kv_cache_bytes_rejects_max_total_tokens() -> None:
    from sglang_omni.config import EngineArgs

    with pytest.raises(ValueError, match="cannot be set together"):
        EngineArgs(kv_cache_bytes="2GiB", max_total_tokens=4096)


ADMISSION_ENV = EngineArgs.ADMISSION_NEW_TOKENS_ESTIMATE_ENV


def test_engine_admission_estimate_is_derived_into_env_not_server_args() -> None:
    engine = EngineArgs(admission_new_tokens_estimate=256, max_running_requests=64)

    assert ADMISSION_ENV == "SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION"
    assert engine.overrides() == {"max_running_requests": 64}
    assert engine.derived_env_defaults() == {ADMISSION_ENV: "256"}


def test_engine_without_an_admission_estimate_derives_nothing() -> None:
    assert EngineArgs().derived_env_defaults() == {}
    assert EngineArgs(max_running_requests=8).derived_env_defaults() == {}


@pytest.mark.parametrize("estimate", [0, -1])
def test_engine_args_reject_a_non_positive_admission_estimate(estimate: int) -> None:
    with pytest.raises(ValueError):
        EngineArgs(admission_new_tokens_estimate=estimate)


def test_stage_lays_written_env_over_the_derived_admission_estimate() -> None:
    stage = engine_stage(
        engine=EngineArgs(admission_new_tokens_estimate=256),
        env={"SGLANG_OTHER": "1"},
    )

    assert stage.resolved_env_defaults() == {ADMISSION_ENV: "256", "SGLANG_OTHER": "1"}
    assert stage.env == {"SGLANG_OTHER": "1"}


def test_stage_accepts_env_that_agrees_with_the_derived_admission_estimate() -> None:
    stage = engine_stage(
        engine=EngineArgs(admission_new_tokens_estimate=256),
        env={ADMISSION_ENV: "256"},
    )

    assert stage.resolved_env_defaults() == {ADMISSION_ENV: "256"}


def test_stage_rejects_env_that_disagrees_with_the_derived_admission_estimate() -> None:
    with pytest.raises(
        ValueError, match="disagrees with engine.admission_new_tokens_estimate"
    ):
        engine_stage(
            engine=EngineArgs(admission_new_tokens_estimate=256),
            env={ADMISSION_ENV: "512"},
        )


def test_stage_without_engine_block_resolves_written_env_only() -> None:
    stage = make_stage(name="code2wav", env={ADMISSION_ENV: "64"})

    assert stage.resolved_env_defaults() == {ADMISSION_ENV: "64"}


def test_engine_stage_default_block_derives_nothing() -> None:
    assert engine_stage().resolved_env_defaults() == {}


def test_pipeline_rejects_shared_process_stages_with_different_estimates() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"stages 'talker' and 'code2wav' resolve different "
            r"SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION defaults \('256' derived vs "
            r"'64' derived\)"
        ),
    ):
        PipelineConfig(
            model_path="model",
            stages=[
                engine_stage(
                    name="talker",
                    next="code2wav",
                    terminal=False,
                    engine=EngineArgs(admission_new_tokens_estimate=256),
                ),
                engine_stage(
                    name="code2wav",
                    engine=EngineArgs(admission_new_tokens_estimate=64),
                ),
            ],
        )


def test_pipeline_rejects_mixed_written_and_derived_estimates_in_one_process() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"stages 'talker' and 'code2wav' resolve different "
            r"SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION defaults \('256' derived vs "
            r"'64' written\)"
        ),
    ):
        PipelineConfig(
            model_path="model",
            stages=[
                engine_stage(
                    name="talker",
                    next="code2wav",
                    terminal=False,
                    engine=EngineArgs(admission_new_tokens_estimate=256),
                ),
                make_stage(name="code2wav", env={ADMISSION_ENV: "64"}),
            ],
        )
