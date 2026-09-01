# SPDX-License-Identifier: Apache-2.0
"""MPS-specific facts extracted from resolved pipeline process specs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

MPS_MODES = ("off", "on", "auto")
_CUDA_DEVICE = re.compile(r"cuda:(\d+)", re.IGNORECASE)


class MpsDecisionError(ValueError):
    pass


@dataclass(frozen=True)
class MpsProcessFact:
    process_name: str
    placement_gpu_ids: tuple[int, ...]
    explicit_cuda_gpu_ids: tuple[int, ...]
    contains_tp: bool


def _process_gpu_ids(process_spec) -> set[int]:
    return {
        int(gpu_id)
        for stage_spec in process_spec.stage_specs
        if (
            gpu_id := (
                stage_spec.placement_gpu_id
                if stage_spec.placement_gpu_id is not None
                else stage_spec.gpu_id
            )
        )
        is not None
    }


def _explicit_cuda_gpu_ids(value) -> set[int]:
    """Find every explicit local CUDA ordinal in resolved launch values."""

    if isinstance(value, str):
        match = _CUDA_DEVICE.fullmatch(value.strip())
        return {int(match.group(1))} if match is not None else set()
    if isinstance(value, Mapping):
        values = value.values()
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = value
    else:
        return set()
    return {gpu_id for item in values for gpu_id in _explicit_cuda_gpu_ids(item)}


def _process_explicit_cuda_gpu_ids(process_spec) -> set[int]:
    return {
        gpu_id
        for stage_spec in process_spec.stage_specs
        for values in (
            stage_spec.factory_kwargs,
            stage_spec.typed_kwargs,
            stage_spec.factory_arg_defaults,
        )
        for gpu_id in _explicit_cuda_gpu_ids(values)
    }


def collect_mps_facts(process_specs) -> tuple[MpsProcessFact, ...]:
    """Extract facts needed for physical MPS planning from resolved specs.

    This deliberately does not inspect pipeline edges or decide workload
    topology. The caller resolves the collected logical ordinals against the
    parent CUDA visibility before making MPS decisions.
    """

    process_specs = list(process_specs)
    process_order: list[str] = []
    placement_by_process: dict[str, set[int]] = {}
    explicit_by_process: dict[str, set[int]] = {}
    contains_tp: dict[str, bool] = {}

    for process_spec in process_specs:
        name = process_spec.process_name
        if name not in placement_by_process:
            process_order.append(name)
            placement_by_process[name] = set()
            explicit_by_process[name] = set()
            contains_tp[name] = False
        placement_by_process[name].update(_process_gpu_ids(process_spec))
        contains_tp[name] = contains_tp[name] or any(
            stage_spec.tp_size > 1 for stage_spec in process_spec.stage_specs
        )

    for process_spec in process_specs:
        name = process_spec.process_name
        if placement_by_process[name] and not contains_tp[name]:
            explicit_by_process[name].update(
                _process_explicit_cuda_gpu_ids(process_spec)
            )

    return tuple(
        MpsProcessFact(
            process_name=name,
            placement_gpu_ids=tuple(sorted(placement_by_process[name])),
            explicit_cuda_gpu_ids=tuple(sorted(explicit_by_process[name])),
            contains_tp=contains_tp[name],
        )
        for name in process_order
    )
