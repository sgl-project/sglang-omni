# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the stage factory kwargs resolution.

Factory kwargs come from exactly two channels: the pipeline author's
``stage_factory_kwargs`` hook (code), and the stage's typed consumer groups
(configuration). These tests pin the overlay order, the pass-through of
free-form group keys, and the placement kwargs that only the launch planner
may supply.
"""

from __future__ import annotations

import pytest

from sglang_omni.config import (
    EngineStageConfig,
    PipelineConfig,
    StageConfig,
    resolve_stage_factory_args,
)

_FACTORY = "tests.unit_test.fixtures.pipeline_fakes.runtime_factory"
_FACTORY_WITHOUT_TOTAL_BUDGET = (
    "tests.unit_test.fixtures.pipeline_fakes.runtime_factory_without_total_budget"
)
_OPEN_FACTORY = "tests.unit_test.fixtures.pipeline_fakes.dummy_factory"


def _stage(**kwargs) -> EngineStageConfig:
    data = {
        "name": "thinker",
        "process": "pipeline",
        "factory_path": _FACTORY,
        "terminal": True,
        "gpu": 1,
    }
    data.update(kwargs)
    return EngineStageConfig(**data)


class _HookedPipelineConfig(PipelineConfig):
    """A pipeline whose author seeds constructor kwargs for the thinker."""

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, object]:
        if stage_name == "thinker":
            return {
                "video_fps": 1.0,
                "server_args_overrides": {"disable_cuda_graph": True},
            }
        return {}


class _PlacementFightingPipelineConfig(PipelineConfig):
    def stage_factory_kwargs(self, stage_name: str) -> dict[str, object]:
        del stage_name
        return {"gpu_id": 0}


def test_typed_groups_map_to_factory_kwargs() -> None:
    stage = _stage(
        gpu_memory_fraction=0.25,
        engine={"mem_fraction_static": 0.72},
        factory={"video_fps": 2.0},
    )
    config = PipelineConfig(model_path="dummy-model", stages=[stage])

    args = resolve_stage_factory_args(stage, config)

    assert args["model_path"] == "dummy-model"
    assert args["gpu_id"] == 1
    assert args["video_fps"] == 2.0
    assert args["server_args_overrides"] == {"mem_fraction_static": 0.72}
    assert args["total_gpu_memory_fraction"] == 0.25


def test_total_gpu_memory_fraction_is_not_injected_into_unrelated_factories() -> None:
    stage = _stage(
        factory_path=_FACTORY_WITHOUT_TOTAL_BUDGET,
        gpu_memory_fraction=0.25,
        engine={"mem_fraction_static": 0.72},
    )
    config = PipelineConfig(model_path="dummy-model", stages=[stage])

    args = resolve_stage_factory_args(stage, config)

    assert "total_gpu_memory_fraction" not in args
    assert args["server_args_overrides"] == {"mem_fraction_static": 0.72}


def test_a_user_group_value_overrides_the_author_kwarg() -> None:
    stage = _stage(factory={"video_fps": 2.0})
    config = _HookedPipelineConfig(model_path="dummy-model", stages=[stage])

    args = resolve_stage_factory_args(stage, config)

    assert args["video_fps"] == 2.0


def test_the_author_kwarg_holds_when_the_user_says_nothing() -> None:
    stage = _stage()
    config = _HookedPipelineConfig(model_path="dummy-model", stages=[stage])

    args = resolve_stage_factory_args(stage, config)

    assert args["video_fps"] == 1.0


def test_engine_keys_merge_over_the_authors_server_args() -> None:
    """Both channels feed ``server_args_overrides``; per-key, config wins."""
    stage = _stage(engine={"mem_fraction_static": 0.72})
    config = _HookedPipelineConfig(model_path="dummy-model", stages=[stage])

    args = resolve_stage_factory_args(stage, config)

    assert args["server_args_overrides"] == {
        "disable_cuda_graph": True,
        "mem_fraction_static": 0.72,
    }


def test_placement_owned_kwargs_are_refused_from_the_hook() -> None:
    stage = _stage()
    config = _PlacementFightingPipelineConfig(model_path="dummy-model", stages=[stage])

    with pytest.raises(ValueError, match="owned by placement"):
        resolve_stage_factory_args(stage, config)


def test_a_set_key_the_factory_does_not_accept_is_refused() -> None:
    """Silently dropping a set path would turn configuration into a no-op."""
    stage = _stage(factory={"lookahead": 9})
    config = PipelineConfig(model_path="dummy-model", stages=[stage])

    with pytest.raises(ValueError, match="does not accept a 'lookahead'"):
        resolve_stage_factory_args(stage, config)


def test_free_form_keys_reach_a_factory_that_takes_kwargs() -> None:
    stage = _stage(factory_path=_OPEN_FACTORY, factory={"lookahead": 9})
    config = PipelineConfig(model_path="dummy-model", stages=[stage])

    args = resolve_stage_factory_args(stage, config)

    assert args["lookahead"] == 9


def test_scheduler_keys_pass_under_their_own_names() -> None:
    stage = _stage(factory_path=_FACTORY, factory={"encoder_mem_reserve": 0.1})
    config = PipelineConfig(model_path="dummy-model", stages=[stage])

    args = resolve_stage_factory_args(stage, config)

    assert args["encoder_mem_reserve"] == 0.1


def test_rank_gpu_id_can_be_supplied_by_launch_planner() -> None:
    stage = _stage(gpu=0)
    config = PipelineConfig(model_path="dummy-model", stages=[stage])

    args = resolve_stage_factory_args(stage, config, gpu_id=3)

    assert args["gpu_id"] == 3


def test_a_plain_stage_carries_no_server_args() -> None:
    stage = StageConfig(
        name="front",
        process="pipeline",
        factory_path=_OPEN_FACTORY,
        terminal=True,
    )
    config = PipelineConfig(model_path="dummy-model", stages=[stage])

    args = resolve_stage_factory_args(stage, config)

    assert "server_args_overrides" not in args
