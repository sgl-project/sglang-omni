# SPDX-License-Identifier: Apache-2.0
"""Compile logical Prefill/Decode stages into physical scheduler stages."""

from __future__ import annotations

from dataclasses import dataclass

from sglang_omni.config.schema import EngineArgs, PDExecution, StageConfig, _pd_gpu_set


@dataclass(frozen=True)
class PDExpansion:
    stages: list[StageConfig]
    entry_stage: str
    routing_map: dict[str, str]
    output_map: dict[str, str]


def expand_pd_stages(stages: list[StageConfig], *, entry_stage: str) -> PDExpansion:
    pd_names = {stage.name for stage in stages if stage.pd_disaggregation is not None}
    if not pd_names:
        routing_map, output_map = _compiled_pd_maps(stages)
        return PDExpansion(
            list(stages),
            routing_map.get(entry_stage, entry_stage),
            routing_map,
            output_map,
        )

    inbound = {name: f"{name}_prefill" for name in pd_names}
    output = {name: f"{name}_decode" for name in pd_names}
    physical: list[StageConfig] = []
    for stage in stages:
        if stage.pd_disaggregation is None:
            physical.append(_rewrite_refs(stage, inbound, output))
            continue
        physical.extend(_split(stage, inbound, output))
    return PDExpansion(
        physical,
        inbound.get(entry_stage, entry_stage),
        inbound,
        output,
    )


def _compiled_pd_maps(
    stages: list[StageConfig],
) -> tuple[dict[str, str], dict[str, str]]:
    """Recover compiler maps when an already-expanded graph is compiled again."""

    routing: dict[str, str] = {}
    output: dict[str, str] = {}
    for stage in stages:
        execution = stage.pd_execution
        suffix = f"_{execution.role}" if execution is not None else ""
        if execution is None or not stage.name.endswith(suffix):
            continue
        logical_name = stage.name[: -len(suffix)]
        (routing if execution.role == "prefill" else output)[logical_name] = stage.name
    return routing, output


def _rename(value: str | list[str] | None, names: dict[str, str]):
    if value is None:
        return None
    if isinstance(value, str):
        return names.get(value, value)
    return [names.get(item, item) for item in value]


def _rewrite_refs(
    stage: StageConfig,
    inbound: dict[str, str],
    output: dict[str, str],
) -> StageConfig:
    return stage.model_copy(
        deep=True,
        update={
            "next": _rename(stage.next, inbound),
            "stream_to": [inbound.get(name, name) for name in stage.stream_to],
            "wait_for": (
                [output.get(name, name) for name in stage.wait_for]
                if stage.wait_for
                else stage.wait_for
            ),
            "project_payload": {
                inbound.get(name, name): path
                for name, path in stage.project_payload.items()
            },
        },
    )


def _half_memory_fraction(stage: StageConfig, placement) -> float | None:
    """This half's share of its card.

    `gpu_memory_fraction` on the logical stage describes one occupant. Copying
    it to both halves would have each size itself against the whole card, and
    `_validate_memory_budgets` sums declared fractions per GPU, so two halves
    on one card would read as double. An explicit per-half share replaces it;
    otherwise the logical value stands, which is right when the halves are on
    separate cards.
    """
    if placement.memory_fraction is not None:
        return placement.memory_fraction
    return stage.gpu_memory_fraction


def _half_engine(stage: StageConfig, placement) -> EngineArgs | None:
    """The logical stage's engine arguments with this half's overrides on top.

    The two halves want different settings: Prefill sizes batches for one
    forward, Decode for many steps. Without an override they share whatever the
    logical stage declared.
    """
    if not placement.engine:
        return stage.engine
    base = stage.engine if stage.engine is not None else EngineArgs()
    return base.model_copy(update=dict(placement.engine))


def _publishing_half_is_prefill(pd) -> bool:
    """Whether the Prefill half is the one that keeps the copy it loaded.

    The publisher holds the weights on top of its own KV; the adopter releases
    what it loaded and needs only KV. Deciding this from the declared shares
    rather than from whichever half wins `gpu_startup_lock` is what makes a
    placement start the same way twice: at prefill 0.30 / decode 0.62 the
    smaller half cannot hold a 56.94 GiB copy, so a run where it published
    failed and a run where it adopted came up.

    Equal shares are a tie and Prefill takes it, which only fixes an order
    that was otherwise a race. When either share is absent there is nothing to
    compare, and the halves are on separate GPUs anyway -- `_validate_pd`
    requires both shares to share a card -- so sharing will decline itself.
    """
    prefill_share = pd.prefill.memory_fraction
    decode_share = pd.decode.memory_fraction
    if prefill_share is None or decode_share is None:
        return True
    return prefill_share >= decode_share


def _split(
    stage: StageConfig,
    inbound: dict[str, str],
    output: dict[str, str],
) -> tuple[StageConfig, StageConfig]:
    pd = stage.pd_disaggregation
    assert pd is not None
    prefill_name = f"{stage.name}_prefill"
    decode_name = f"{stage.name}_decode"
    shares_a_gpu = bool(_pd_gpu_set(pd.prefill.gpu) & _pd_gpu_set(pd.decode.gpu))
    share_weights = pd.share_weights and shares_a_gpu
    prefill_publishes = _publishing_half_is_prefill(pd)
    prefill = stage.model_copy(
        deep=True,
        update={
            "name": prefill_name,
            "gpu": pd.prefill.gpu,
            "process": pd.prefill.process or prefill_name,
            "gpu_memory_fraction": _half_memory_fraction(stage, pd.prefill),
            "engine": _half_engine(stage, pd.prefill),
            "next": None,
            "terminal": False,
            "route_fn": None,
            "stream_to": [],
            "stream_done_to_fn": None,
            "project_payload": {},
            "wait_for": (
                [output.get(name, name) for name in stage.wait_for]
                if stage.wait_for
                else stage.wait_for
            ),
            "pd_disaggregation": None,
            "pd_execution": PDExecution(
                role="prefill",
                partner=decode_name,
                decode_targets=(decode_name,),
                share_weights=share_weights,
                publishes_weights=share_weights and prefill_publishes,
            ),
        },
    )
    decode = stage.model_copy(
        deep=True,
        update={
            "name": decode_name,
            "gpu": pd.decode.gpu,
            "process": pd.decode.process or decode_name,
            "gpu_memory_fraction": _half_memory_fraction(stage, pd.decode),
            "engine": _half_engine(stage, pd.decode),
            "next": _rename(stage.next, inbound),
            "stream_to": [inbound.get(name, name) for name in stage.stream_to],
            "project_payload": {
                inbound.get(name, name): path
                for name, path in stage.project_payload.items()
            },
            "wait_for": None,
            "wait_for_fn": None,
            "merge_fn": None,
            "pd_disaggregation": None,
            "pd_execution": PDExecution(
                role="decode",
                partner=prefill_name,
                share_weights=share_weights,
                publishes_weights=share_weights and not prefill_publishes,
            ),
        },
    )
    return prefill, decode
