# SPDX-License-Identifier: Apache-2.0
"""Generic PD (prefill/decode) disaggregation graph rewrite.

Expands each logical stage marked ``pd_disaggregation`` into two physical
stages, ``<stage>_prefill`` and ``<stage>_decode``. Inbound edges terminate at
the prefill half; downstream edges, terminal/``stream_to``/``route_fn`` logic,
and completion accounting move to the decode half. The prefill -> decode
handoff is recorded in typed ``pd_execution`` (not an ordinary ``next`` edge),
otherwise the coordinator would admit decode as a fresh downstream request and
double-transport payload alongside the KV plane.
"""

from __future__ import annotations

from dataclasses import dataclass

from sglang_omni.config.schema import PDExecution, StageConfig

PREFILL_SUFFIX = "_prefill"
DECODE_SUFFIX = "_decode"


@dataclass(frozen=True)
class PDExpansion:
    """Result of the PD graph rewrite.

    ``routing_map`` sends logical stage names to the prefill half; ``output_map``
    resolves logical output sources and terminal completion to the decode half.
    """

    stages: list[StageConfig]
    entry_stage: str
    routing_map: dict[str, str]
    output_map: dict[str, str]


def expand_pd_stages(
    stages: list[StageConfig],
    *,
    entry_stage: str,
) -> PDExpansion:
    """Expand PD-disaggregation stages into prefill/decode physical stages.

    Stages without ``pd_disaggregation`` are preserved in order, with inbound
    references to any PD stage rewritten to that stage's prefill half. Calling
    this again on an already-expanded stage list is a safe no-op because the
    generated halves clear ``pd_disaggregation`` (idempotent).
    """
    pd_names = {s.name for s in stages if s.pd_disaggregation is not None}
    if not pd_names:
        return PDExpansion(
            stages=list(stages),
            entry_stage=entry_stage,
            routing_map={},
            output_map={},
        )

    # Note (Yue Yin): Split routing from completion because only Decode can
    # report the terminal result after a handoff.
    inbound_rename = {name: f"{name}{PREFILL_SUFFIX}" for name in pd_names}
    output_rename = {name: f"{name}{DECODE_SUFFIX}" for name in pd_names}

    out: list[StageConfig] = []
    for s in stages:
        if s.pd_disaggregation is None:
            out.append(_rewrite_refs(s, inbound_rename, output_rename))
            continue
        prefill, decode = _split_pd_stage(s, inbound_rename, output_rename)
        out.append(prefill)
        out.append(decode)

    new_entry = inbound_rename.get(entry_stage, entry_stage)
    return PDExpansion(
        stages=out,
        entry_stage=new_entry,
        routing_map=dict(inbound_rename),
        output_map=dict(output_rename),
    )


def _rename_targets(
    targets: str | list[str] | None,
    rename: dict[str, str],
) -> str | list[str] | None:
    if targets is None:
        return None
    if isinstance(targets, str):
        return rename.get(targets, targets)
    return [rename.get(t, t) for t in targets]


def _rewrite_refs(
    s: StageConfig,
    destination_rename: dict[str, str],
    output_rename: dict[str, str],
) -> StageConfig:
    next_ = _rename_targets(s.next, destination_rename)
    stream_to = [destination_rename.get(t, t) for t in s.stream_to]
    wait_for = (
        [output_rename.get(t, t) for t in s.wait_for] if s.wait_for else s.wait_for
    )
    project_payload = {
        destination_rename.get(k, k): v for k, v in s.project_payload.items()
    }

    if (
        next_ == s.next
        and stream_to == list(s.stream_to)
        and wait_for == s.wait_for
        and project_payload == s.project_payload
    ):
        return s

    return s.model_copy(
        deep=True,
        update={
            "next": next_,
            "stream_to": stream_to,
            "wait_for": wait_for,
            "project_payload": project_payload,
        },
    )


def _split_pd_stage(
    s: StageConfig,
    inbound_rename: dict[str, str],
    output_rename: dict[str, str],
) -> tuple[StageConfig, StageConfig]:
    pd = s.pd_disaggregation
    assert pd is not None
    prefill_name = f"{s.name}{PREFILL_SUFFIX}"
    decode_name = f"{s.name}{DECODE_SUFFIX}"

    prefill = s.model_copy(
        deep=True,
        update={
            "name": prefill_name,
            "gpu": pd.prefill.gpu if pd.prefill.gpu is not None else s.gpu,
            "process": pd.prefill.process or prefill_name,
            # Note (Yue Yin): Preserve the result path when the first sampled
            # token finishes the request before a KV handoff is needed.
            "next": _rename_targets(s.next, inbound_rename),
            "project_payload": {
                inbound_rename.get(k, k): v for k, v in s.project_payload.items()
            },
            "wait_for": (
                [output_rename.get(source, source) for source in s.wait_for]
                if s.wait_for
                else s.wait_for
            ),
            "terminal": False,
            "route_fn": None,
            # Note (Yue Yin): Prefill owns the first sampled token, and when it
            # also finishes the request it must close the downstream stream.
            "stream_to": [inbound_rename.get(t, t) for t in s.stream_to],
            "pd_disaggregation": None,
            # Note (Yue Yin): Keep compiler metadata out of user factory kwargs.
            "pd_execution": PDExecution(role="prefill", partner=decode_name),
        },
    )

    decode = s.model_copy(
        deep=True,
        update={
            "name": decode_name,
            "gpu": pd.decode.gpu if pd.decode.gpu is not None else s.gpu,
            "process": pd.decode.process or decode_name,
            "next": _rename_targets(s.next, inbound_rename),
            "stream_to": [inbound_rename.get(t, t) for t in s.stream_to],
            "project_payload": {
                inbound_rename.get(k, k): v for k, v in s.project_payload.items()
            },
            "wait_for": None,
            "wait_for_fn": None,
            "merge_fn": None,
            "pd_disaggregation": None,
            "pd_execution": PDExecution(role="decode", partner=prefill_name),
        },
    )

    return prefill, decode
