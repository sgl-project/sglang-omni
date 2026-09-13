# SPDX-License-Identifier: Apache-2.0
"""Tests for MPS fact extraction from resolved process specs."""

from __future__ import annotations

from sglang_omni.mps.decision import collect_mps_facts
from sglang_omni.pipeline.stage_workers import StageLaunchConfig, StageWorkerProcessSpec

_FACTORY = f"{__name__}.unused_factory"


def proc(name, gpu_id, tp_size=1):
    return StageWorkerProcessSpec(
        process_name=name,
        stage_specs=[
            StageLaunchConfig(
                stage_name=name,
                factory=_FACTORY,
                gpu_id=gpu_id,
                placement_gpu_id=gpu_id,
                tp_size=tp_size,
            )
        ],
    )


def test_extracts_resolved_process_facts_without_deciding_physical_identity():
    placed = proc("placed", 0)
    placed.stage_specs[0].placement_gpu_id = 3
    placed.stage_specs[0].factory_kwargs = {"nested": [{"device": "cuda:1"}]}
    placed.stage_specs[0].typed_kwargs = {"configured": "cuda:2"}
    placed.stage_specs[0].factory_arg_defaults = {"fallback": "cuda:4"}
    tp = proc("tp", 4, tp_size=2)

    facts = collect_mps_facts([placed, tp])

    assert facts[0].process_name == "placed"
    assert facts[0].placement_gpu_ids == (3,)
    assert facts[0].explicit_cuda_gpu_ids == (1, 2, 4)
    assert not facts[0].contains_tp
    assert facts[1].contains_tp


def test_duplicate_process_specs_preserve_first_seen_order_and_merge_gpu_facts():
    first = proc("b", 1)
    first.stage_specs[0].factory_kwargs = {"device": "cuda:1"}
    later = proc("b", 2)
    later.stage_specs[0].factory_kwargs = {"device": "cuda:2"}
    facts = collect_mps_facts(spec for spec in (first, proc("a", 0), later))

    assert [fact.process_name for fact in facts] == ["b", "a"]
    assert facts[0].placement_gpu_ids == (1, 2)
    assert facts[0].explicit_cuda_gpu_ids == (1, 2)
