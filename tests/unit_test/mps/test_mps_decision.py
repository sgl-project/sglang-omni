# SPDX-License-Identifier: Apache-2.0
"""Tests for MPS fact extraction from resolved process specs."""

from __future__ import annotations

from dataclasses import dataclass, field

from sglang_omni.mps.decision import collect_mps_facts

_FACTORY = f"{__name__}.unused_factory"


@dataclass
class ResolvedStageLaunch:
    """Minimal resolved launch record consumed by MPS planning."""

    stage_name: str
    gpu_id: int | None
    tp_size: int = 1
    placement_gpu_id: int | None = None
    factory: str = _FACTORY
    factory_kwargs: dict = field(default_factory=dict)
    typed_kwargs: dict = field(default_factory=dict)
    factory_arg_defaults: dict = field(default_factory=dict)


@dataclass
class ResolvedProcessSpec:
    process_name: str
    stage_specs: list[ResolvedStageLaunch] = field(default_factory=list)


def proc(name, gpu_id, tp_size=1):
    return ResolvedProcessSpec(
        process_name=name,
        stage_specs=[
            ResolvedStageLaunch(
                stage_name=name,
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
