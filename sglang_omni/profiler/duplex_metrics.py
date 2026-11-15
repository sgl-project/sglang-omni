# SPDX-License-Identifier: Apache-2.0
"""Reconstruct Full Duplex Unit and stage timing from profiler events."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from sglang_omni.profiler.views import load_events
from sglang_omni.profiler.views import percentile as view_percentile
from sglang_omni.utils.json import JsonValue

ScalarValue = str | int | float | None
SessionIdentityKey = tuple[str, int, int]

SESSION_EVENT_NAMES = frozenset(
    {
        "session_unit_admitted",
        "session_unit_ready",
        "session_stage_started",
        "session_stage_bypassed",
        "session_output_emitted",
        "session_unit_finished",
    }
)
GENERIC_STAGE_EVENT_NAMES = frozenset(
    {
        "stage_dispatch",
        "scheduler_queue_enter",
        "scheduler_prefill_start",
        "stage_complete",
    }
)
STAGE_EVENT_NAMES = frozenset({"session_stage_started", "session_stage_bypassed"}) | (
    GENERIC_STAGE_EVENT_NAMES
)


@dataclass
class StageTiming:
    """Observed timestamps and derived values for one stage."""

    stage_name: str
    dispatch_timestamp_ns: int | None = None
    started_timestamp_ns: int | None = None
    scheduler_queue_enter_timestamp_ns: int | None = None
    scheduler_prefill_start_timestamp_ns: int | None = None
    finished_timestamp_ns: int | None = None
    bypassed_timestamp_ns: int | None = None
    bypass_reason: str | None = None

    def resolved_boundaries(
        self,
    ) -> tuple[int | None, int | None, int | None]:
        """Select AR scheduler boundaries when either AR boundary exists."""
        if (
            self.scheduler_queue_enter_timestamp_ns is not None
            or self.scheduler_prefill_start_timestamp_ns is not None
        ):
            return (
                self.scheduler_queue_enter_timestamp_ns,
                self.scheduler_prefill_start_timestamp_ns,
                self.finished_timestamp_ns,
            )
        else:
            return (
                self.dispatch_timestamp_ns,
                self.started_timestamp_ns,
                self.finished_timestamp_ns,
            )

    def timing_status(self) -> str:
        """Classify a stage as observed, bypassed, or incomplete."""
        if self.bypassed_timestamp_ns is not None:
            return "bypassed"
        else:
            queued_timestamp_ns, started_timestamp_ns, finished_timestamp_ns = (
                self.resolved_boundaries()
            )
            if (
                queued_timestamp_ns is not None
                and started_timestamp_ns is not None
                and finished_timestamp_ns is not None
            ):
                return "observed"
            else:
                return "incomplete"

    def record_bypass(self, timestamp_ns: int | None, reason: str | None) -> None:
        """Keep the earliest real bypass observation."""
        if timestamp_ns is not None and (
            self.bypassed_timestamp_ns is None
            or timestamp_ns < self.bypassed_timestamp_ns
        ):
            self.bypassed_timestamp_ns = timestamp_ns
            self.bypass_reason = reason
        else:
            pass

    def to_report(
        self, previous_finished_timestamp_ns: int | None
    ) -> dict[str, ScalarValue]:
        queued_timestamp_ns, started_timestamp_ns, finished_timestamp_ns = (
            self.resolved_boundaries()
        )
        timing_status = self.timing_status()
        if timing_status == "bypassed":
            queue_ms = None
            service_ms = None
        else:
            queue_ms = elapsed_ms(queued_timestamp_ns, started_timestamp_ns)
            service_ms = elapsed_ms(started_timestamp_ns, finished_timestamp_ns)
        return {
            "stage": self.stage_name,
            "dispatch_timestamp_ns": self.dispatch_timestamp_ns,
            "queued_timestamp_ns": queued_timestamp_ns,
            "started_timestamp_ns": started_timestamp_ns,
            "finished_timestamp_ns": finished_timestamp_ns,
            "bypassed_timestamp_ns": self.bypassed_timestamp_ns,
            "bypass_reason": self.bypass_reason,
            "timing_status": timing_status,
            "queue_ms": queue_ms,
            "service_ms": service_ms,
            "handoff_from_previous_ms": elapsed_ms(
                previous_finished_timestamp_ns, self.dispatch_timestamp_ns
            ),
        }


@dataclass
class UnitTiming:
    """Raw events collected for one request-backed session Unit."""

    request_id: str
    session_id: str
    session_open_index: int
    input_seq: int
    media_start_ms: float | None = None
    media_duration_ms: float | None = None
    input_modality: str | None = None
    ready_timestamp_ns: int | None = None
    admitted_timestamp_ns: int | None = None
    finished_timestamp_ns: int | None = None
    stages: dict[str, StageTiming] = field(default_factory=dict)
    outputs: list[dict[str, ScalarValue]] = field(default_factory=list)


def elapsed_ms(
    start_timestamp_ns: int | None, end_timestamp_ns: int | None
) -> float | None:
    """Return a timestamp difference in milliseconds when both ends exist."""
    if start_timestamp_ns is None or end_timestamp_ns is None:
        return None
    else:
        return (end_timestamp_ns - start_timestamp_ns) / 1_000_000.0


def reduce_timestamp_ns(
    existing_timestamp_ns: int | None,
    observed_timestamp_ns: int | None,
    *,
    latest: bool,
) -> int | None:
    """Reduce duplicate TP observations to the appropriate time envelope."""
    if observed_timestamp_ns is None:
        return existing_timestamp_ns
    elif existing_timestamp_ns is None:
        return observed_timestamp_ns
    elif latest:
        return max(existing_timestamp_ns, observed_timestamp_ns)
    else:
        return min(existing_timestamp_ns, observed_timestamp_ns)


def fill_input_metadata(
    unit_timing: UnitTiming,
    metadata: Mapping[str, JsonValue],
) -> None:
    input_modality = metadata.get("input_modality")
    if unit_timing.input_modality is None and isinstance(input_modality, str):
        unit_timing.input_modality = input_modality
    else:
        pass
    media_start_ms = metadata.get("input_t_start_ms")
    if unit_timing.media_start_ms is None and isinstance(media_start_ms, (int, float)):
        unit_timing.media_start_ms = float(media_start_ms)
    else:
        pass
    media_duration_ms = metadata.get("input_duration_ms")
    if unit_timing.media_duration_ms is None and isinstance(
        media_duration_ms, (int, float)
    ):
        unit_timing.media_duration_ms = float(media_duration_ms)
    else:
        pass


def distribution(values: list[float]) -> dict[str, ScalarValue]:
    """Summarize observed values without turning missing values into zero."""
    if not values:
        return {"count": 0, "p50": None, "p95": None, "max": None}
    else:
        pass
    ordered_values = sorted(values)
    return {
        "count": len(ordered_values),
        "p50": view_percentile(ordered_values, 0.50),
        "p95": view_percentile(ordered_values, 0.95),
        "max": ordered_values[-1],
    }


def load_duplex_events(
    events: Iterable[Mapping[str, JsonValue]] | str | Path,
) -> list[Mapping[str, JsonValue]]:
    """Load events and apply stable timestamp ordering."""
    if isinstance(events, (str, Path)):
        loaded_events = list(load_events(events))
    else:
        loaded_events = list(events)
        loaded_events.sort(
            key=lambda event: (
                event["timestamp_ns"]
                if isinstance(event.get("timestamp_ns"), int)
                else 0
            )
        )
    return loaded_events


def session_identity_from_event(
    event: Mapping[str, JsonValue],
) -> SessionIdentityKey | None:
    metadata = event.get("metadata")
    if not isinstance(metadata, dict):
        return None
    else:
        pass
    session_id = metadata.get("session_id")
    session_open_index = metadata.get("session_open_index")
    input_seq = metadata.get("input_seq")
    if (
        isinstance(session_id, str)
        and isinstance(session_open_index, int)
        and isinstance(input_seq, int)
    ):
        return session_id, session_open_index, input_seq
    else:
        return None


def stage_for_unit(unit_timing: UnitTiming, stage_name: str) -> StageTiming:
    stage_timing = unit_timing.stages.get(stage_name)
    if stage_timing is None:
        stage_timing = StageTiming(stage_name=stage_name)
        unit_timing.stages[stage_name] = stage_timing
    else:
        pass
    return stage_timing


def stage_order_key(stage_timing: StageTiming) -> tuple[int, str]:
    queued_timestamp_ns, started_timestamp_ns, finished_timestamp_ns = (
        stage_timing.resolved_boundaries()
    )
    timestamps = (
        stage_timing.dispatch_timestamp_ns,
        queued_timestamp_ns,
        started_timestamp_ns,
        finished_timestamp_ns,
        stage_timing.bypassed_timestamp_ns,
    )
    first_timestamp_ns = min(
        (timestamp_ns for timestamp_ns in timestamps if timestamp_ns is not None),
        default=0,
    )
    return first_timestamp_ns, stage_timing.stage_name


def unit_report(unit_timing: UnitTiming) -> dict[str, JsonValue]:
    """Build one request-backed Unit report."""
    ordered_stages = sorted(unit_timing.stages.values(), key=stage_order_key)
    stage_reports: list[dict[str, ScalarValue]] = []
    previous_finished_timestamp_ns: int | None = None
    for stage_timing in ordered_stages:
        stage_reports.append(stage_timing.to_report(previous_finished_timestamp_ns))
        if stage_timing.finished_timestamp_ns is not None:
            previous_finished_timestamp_ns = stage_timing.finished_timestamp_ns
        else:
            pass

    unit_timing.outputs.sort(
        key=lambda output: (
            output["emitted_timestamp_ns"]
            if isinstance(output["emitted_timestamp_ns"], int)
            else 0
        )
    )
    first_output_timestamp_ns = next(
        (
            output["emitted_timestamp_ns"]
            for output in unit_timing.outputs
            if isinstance(output["emitted_timestamp_ns"], int)
        ),
        None,
    )
    return {
        "request_id": unit_timing.request_id,
        "session_id": unit_timing.session_id,
        "session_open_index": unit_timing.session_open_index,
        "input_seq": unit_timing.input_seq,
        "input_modality": unit_timing.input_modality,
        "media_start_ms": unit_timing.media_start_ms,
        "media_duration_ms": unit_timing.media_duration_ms,
        "ready_timestamp_ns": unit_timing.ready_timestamp_ns,
        "admitted_timestamp_ns": unit_timing.admitted_timestamp_ns,
        "finished_timestamp_ns": unit_timing.finished_timestamp_ns,
        "ready_to_admitted_ms": elapsed_ms(
            unit_timing.ready_timestamp_ns, unit_timing.admitted_timestamp_ns
        ),
        "ready_to_first_output_ms": elapsed_ms(
            unit_timing.ready_timestamp_ns, first_output_timestamp_ns
        ),
        "ready_to_finished_ms": elapsed_ms(
            unit_timing.ready_timestamp_ns, unit_timing.finished_timestamp_ns
        ),
        "admitted_to_first_output_ms": elapsed_ms(
            unit_timing.admitted_timestamp_ns, first_output_timestamp_ns
        ),
        "admitted_to_finished_ms": elapsed_ms(
            unit_timing.admitted_timestamp_ns, unit_timing.finished_timestamp_ns
        ),
        "stages": stage_reports,
        "outputs": [dict(output) for output in unit_timing.outputs],
    }


def build_duplex_session_metrics(
    events: Iterable[Mapping[str, JsonValue]] | str | Path,
) -> dict[str, JsonValue]:
    """Build auditable Full Duplex Unit and stage timing metrics."""
    events_by_request_id: dict[str, list[Mapping[str, JsonValue]]] = defaultdict(list)
    for event in load_duplex_events(events):
        event_name = event.get("event_name")
        request_id = event.get("request_id")
        if (
            isinstance(event_name, str)
            and (
                event_name in SESSION_EVENT_NAMES
                or event_name in GENERIC_STAGE_EVENT_NAMES
            )
            and isinstance(request_id, str)
        ):
            events_by_request_id[request_id].append(event)
        else:
            pass

    units: list[UnitTiming] = []
    for request_id, request_events in events_by_request_id.items():
        identity: SessionIdentityKey | None = next(
            (
                candidate
                for event in request_events
                if event.get("event_name") in SESSION_EVENT_NAMES
                if (candidate := session_identity_from_event(event)) is not None
            ),
            None,
        )
        if identity is None:
            continue
        else:
            pass

        unit_timing = UnitTiming(
            request_id=request_id,
            session_id=identity[0],
            session_open_index=identity[1],
            input_seq=identity[2],
        )

        for event in request_events:
            event_name = event.get("event_name")
            timestamp_value = event.get("timestamp_ns")
            timestamp_ns = timestamp_value if isinstance(timestamp_value, int) else None
            metadata = event.get("metadata")
            stage_name = event.get("stage")
            stage_timing = (
                stage_for_unit(unit_timing, stage_name)
                if event_name in STAGE_EVENT_NAMES and isinstance(stage_name, str)
                else None
            )
            if event_name in SESSION_EVENT_NAMES and isinstance(metadata, Mapping):
                fill_input_metadata(unit_timing, metadata)
            else:
                pass
            if event_name in STAGE_EVENT_NAMES and stage_timing is None:
                continue
            else:
                pass
            match event_name:
                case "session_unit_ready":
                    unit_timing.ready_timestamp_ns = reduce_timestamp_ns(
                        unit_timing.ready_timestamp_ns, timestamp_ns, latest=False
                    )
                case "session_unit_admitted":
                    unit_timing.admitted_timestamp_ns = reduce_timestamp_ns(
                        unit_timing.admitted_timestamp_ns, timestamp_ns, latest=False
                    )
                case "session_unit_finished":
                    unit_timing.finished_timestamp_ns = reduce_timestamp_ns(
                        unit_timing.finished_timestamp_ns, timestamp_ns, latest=True
                    )
                case "session_stage_started":
                    stage_timing.started_timestamp_ns = reduce_timestamp_ns(
                        stage_timing.started_timestamp_ns,
                        timestamp_ns,
                        latest=False,
                    )
                case "session_stage_bypassed":
                    reason = (
                        metadata.get("reason")
                        if isinstance(metadata, Mapping)
                        else None
                    )
                    stage_timing.record_bypass(
                        timestamp_ns, reason if isinstance(reason, str) else None
                    )
                case "session_output_emitted":
                    if (
                        not isinstance(metadata, Mapping)
                        or metadata.get("kind") == "input_done"
                    ):
                        pass
                    else:
                        unit_timing.outputs.append(
                            {
                                "emitted_timestamp_ns": timestamp_ns,
                                "output_seq": metadata.get("output_seq"),
                                "output_modality": metadata.get("output_modality"),
                                "output_t_start_ms": metadata.get("output_t_start_ms"),
                                "output_duration_ms": metadata.get(
                                    "output_duration_ms"
                                ),
                                "kind": metadata.get("kind"),
                            }
                        )
                case "stage_dispatch":
                    stage_timing.dispatch_timestamp_ns = reduce_timestamp_ns(
                        stage_timing.dispatch_timestamp_ns,
                        timestamp_ns,
                        latest=False,
                    )
                case "scheduler_queue_enter":
                    stage_timing.scheduler_queue_enter_timestamp_ns = (
                        reduce_timestamp_ns(
                            stage_timing.scheduler_queue_enter_timestamp_ns,
                            timestamp_ns,
                            latest=False,
                        )
                    )
                case "scheduler_prefill_start":
                    stage_timing.scheduler_prefill_start_timestamp_ns = (
                        reduce_timestamp_ns(
                            stage_timing.scheduler_prefill_start_timestamp_ns,
                            timestamp_ns,
                            latest=False,
                        )
                    )
                case "stage_complete":
                    stage_timing.finished_timestamp_ns = reduce_timestamp_ns(
                        stage_timing.finished_timestamp_ns,
                        timestamp_ns,
                        latest=True,
                    )
                case _:
                    pass
        units.append(unit_timing)

    ordered_units = sorted(
        units,
        key=lambda unit: (
            unit.session_id,
            unit.session_open_index,
            unit.input_seq,
            unit.request_id,
        ),
    )
    unit_reports = [unit_report(unit) for unit in ordered_units]
    session_groups: dict[tuple[str, int], list[UnitTiming]] = defaultdict(list)
    for unit in ordered_units:
        session_groups[(unit.session_id, unit.session_open_index)].append(unit)
    sessions = [
        {
            "session_id": session_id,
            "session_open_index": session_open_index,
            "unit_count": len(group),
        }
        for (session_id, session_open_index), group in sorted(session_groups.items())
    ]

    def metric_values(metric_name: str) -> list[float]:
        return [
            float(report[metric_name])
            for report in unit_reports
            if isinstance(report[metric_name], (int, float))
        ]

    stage_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"queue_ms": [], "service_ms": [], "handoff_ms": []}
    )
    for report in unit_reports:
        for stage in report["stages"]:
            stage_name = stage["stage"]
            if isinstance(stage_name, str):
                for metric_name, report_name in (
                    ("queue_ms", "queue_ms"),
                    ("service_ms", "service_ms"),
                    ("handoff_ms", "handoff_from_previous_ms"),
                ):
                    metric_value = stage[report_name]
                    if isinstance(metric_value, (int, float)):
                        stage_values[stage_name][metric_name].append(
                            float(metric_value)
                        )
                    else:
                        pass
            else:
                pass
    stage_summary = {
        stage_name: {
            metric_name: distribution(values)
            for metric_name, values in metric_values.items()
        }
        for stage_name, metric_values in sorted(stage_values.items())
    }
    return {
        "sessions": sessions,
        "units": unit_reports,
        "summary": {
            "ready_to_admitted_ms": distribution(metric_values("ready_to_admitted_ms")),
            "ready_to_first_output_ms": distribution(
                metric_values("ready_to_first_output_ms")
            ),
            "ready_to_finished_ms": distribution(metric_values("ready_to_finished_ms")),
            "admitted_to_finished_ms": distribution(
                metric_values("admitted_to_finished_ms")
            ),
            "admitted_to_first_output_ms": distribution(
                metric_values("admitted_to_first_output_ms")
            ),
            "stages": stage_summary,
        },
    }
