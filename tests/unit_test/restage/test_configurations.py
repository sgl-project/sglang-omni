"""Search configuration changes use the same typed patches as serving."""

from sglang_omni.restage.configurations import enumerate_configurations
from tests.unit_test.restage.conftest_seed import pipeline


def test_tp_and_memory_choices_form_independent_dimensions():
    source = pipeline()
    before = source.model_dump()
    variants = list(
        enumerate_configurations(
            source,
            {
                "tp": [
                    {"engine.tp_size": 1, "engine.gpu": [0]},
                    {"engine.tp_size": 2, "engine.gpu": [0, 1]},
                ],
                "memory": [
                    {"engine.gpu_memory_fraction": 0.4},
                    {"engine.gpu_memory_fraction": 0.6},
                ],
            },
        )
    )
    assert len(variants) == 4
    assert all(v.config is not None for v in variants)
    assert {
        (
            v.config.stage_named("engine").tp_size,
            v.config.stage_named("engine").gpu_memory_fraction,
        )
        for v in variants
    } == {(1, 0.4), (1, 0.6), (2, 0.4), (2, 0.6)}
    assert source.model_dump() == before


def test_process_grouping_is_a_search_dimension():
    variants = list(
        enumerate_configurations(
            pipeline(),
            {
                "group": [{}, {"tail.process": "engine"}],
            },
        )
    )
    assert [v.config.stage_named("tail").process for v in variants] == [
        "tail",
        "engine",
    ]


def test_conflicting_choices_are_reported_not_silently_overwritten():
    variants = list(
        enumerate_configurations(
            pipeline(),
            {
                "a": [{"engine.tp_size": 1}],
                "b": [{"engine.tp_size": 2}],
            },
        )
    )
    assert variants[0].config is None
    assert "set twice" in variants[0].rejection


def test_internal_topology_edits_stay_forbidden():
    variant = next(
        enumerate_configurations(
            pipeline(),
            {
                "bad": [{"engine.factory_path": "untrusted.factory"}],
            },
        )
    )
    assert variant.config is None
    assert "internal" in variant.rejection
