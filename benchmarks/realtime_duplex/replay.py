# SPDX-License-Identifier: Apache-2.0
"""Offline replay: deterministic verdicts over a recorded session log.

Replay reads the manifest and entries produced by session_log and nothing
else — no server, no audio file, no wall clock. The same log always
yields the same report. Verdicts answer protocol and accounting
questions; timing_profile carries the recorded-latency numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchmarks.realtime_duplex import schema
from benchmarks.realtime_duplex.session_log import (
    CLIENT_TO_SERVER,
    SERVER_TO_CLIENT,
    LogEntry,
    SessionManifest,
)

UNDERRUN_GAP_FACTOR = 3.0
"""A response-audio gap longer than frame budget times this factor is an
underrun, not jitter (single-session, single-stream assumption)."""

NEGATIVE_SCENARIO_PREFIX = "negative"


@dataclass(slots=True)
class Verdict:
    name: str
    passed: bool
    detail: str


@dataclass(slots=True)
class ResponseWindow:
    """One response: created at window[0], closed by response.done or None."""

    created: LogEntry
    deltas: list[LogEntry] = field(default_factory=list)
    audio_done: LogEntry | None = None
    done: LogEntry | None = None
    cancelled_by_client: bool = False


@dataclass(slots=True)
class ReplayReport:
    verdicts: list[Verdict]
    timing_profile: dict[str, Any]
    counts: dict[str, int]
    errors: list[dict[str, Any]]

    @property
    def passed(self) -> bool:
        return all(verdict.passed for verdict in self.verdicts)

    @property
    def summary(self) -> str:
        failed = [verdict.name for verdict in self.verdicts if not verdict.passed]
        if not failed:
            return f"all {len(self.verdicts)} verdicts passed"
        return "failed: " + ", ".join(failed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "summary": self.summary,
            "verdicts": [
                {
                    "name": verdict.name,
                    "passed": verdict.passed,
                    "detail": verdict.detail,
                }
                for verdict in self.verdicts
            ],
            "timing_profile": self.timing_profile,
            "counts": self.counts,
            "errors": self.errors,
        }


def group_responses(entries: list[LogEntry]) -> list[ResponseWindow]:
    """Pair each response.created with everything up to its response.done."""
    windows: list[ResponseWindow] = []
    for entry in entries:
        if entry.direction != SERVER_TO_CLIENT:
            if entry.type == "response.cancel" and windows and windows[-1].done is None:
                windows[-1].cancelled_by_client = True
            continue
        if entry.type == "response.created":
            windows.append(ResponseWindow(created=entry))
        elif not windows:
            continue
        elif entry.type in schema.RESPONSE_DELTA_EVENTS:
            windows[-1].deltas.append(entry)
        elif entry.type == "response.audio.done":
            windows[-1].audio_done = entry
        elif entry.type == "response.done":
            windows[-1].done = entry
    return windows


def session_ordering(entries: list[LogEntry]) -> Verdict:
    s2c = [entry for entry in entries if entry.direction == SERVER_TO_CLIENT]
    created = [entry for entry in s2c if entry.type == "session.created"]
    problems: list[str] = []
    if not s2c:
        return Verdict("session_ordering", False, "no server events recorded")
    if not created:
        problems.append("no session.created")
    elif created[0] is not s2c[0]:
        problems.append("first server event is not session.created")
    if len(created) > 1:
        problems.append(f"{len(created)} session.created events")
    for entry in s2c:
        if entry.type != "session.updated":
            continue
        earlier_updates = [
            sent
            for sent in entries[: entries.index(entry)]
            if sent.direction == CLIENT_TO_SERVER and sent.type == "session.update"
        ]
        if not earlier_updates:
            problems.append("session.updated without a preceding session.update")
            break
    return Verdict(
        "session_ordering",
        not problems,
        "; ".join(problems) if problems else "session lifecycle ordering holds",
    )


def response_lifecycle(
    windows: list[ResponseWindow], *, expect_responses: bool = True
) -> Verdict:
    problems: list[str] = []
    if not windows:
        if expect_responses:
            problems.append("no response windows recorded")
        return Verdict(
            "response_lifecycle",
            not problems,
            (
                "; ".join(problems)
                if problems
                else "no responses expected by this scenario"
            ),
        )
    for index, window in enumerate(windows):
        label = f"response[{index}]"
        if window.done is None:
            problems.append(f"{label} never closed with response.done")
            continue
        if window.done.t_s < window.created.t_s:
            problems.append(f"{label} closed before it was created")
        done_status = str(window.done.event.get("response", {}).get("status", ""))
        if window.cancelled_by_client and done_status != "cancelled":
            problems.append(
                f"{label} cancelled by client but done.status={done_status!r}"
            )
        if not window.cancelled_by_client and done_status == "cancelled":
            problems.append(f"{label} reported cancelled without a client cancel")
        for delta in window.deltas:
            if delta.t_s < window.created.t_s:
                problems.append(f"{label} delta before response.created")
            elif delta.t_s > window.done.t_s:
                problems.append(f"{label} delta after response.done")
            elif window.audio_done is not None and delta.t_s > window.audio_done.t_s:
                problems.append(f"{label} delta after response.audio.done")
    if not windows:
        problems.append("no response windows recorded")
    return Verdict(
        "response_lifecycle",
        not problems,
        (
            "; ".join(problems)
            if problems
            else f"{len(windows)} responses opened and closed correctly"
        ),
    )


def ack_closure(entries: list[LogEntry]) -> Verdict:
    problems: list[str] = []
    pairs = (
        ("session.update", "session.updated"),
        ("conversation.item.truncate", "conversation.item.truncated"),
        ("response.cancel", "response.done"),
    )
    for sent_type, ack_type in pairs:
        sent_at = [
            entry.t_s
            for entry in entries
            if entry.direction == CLIENT_TO_SERVER and entry.type == sent_type
        ]
        ack_at = [
            entry.t_s
            for entry in entries
            if entry.direction == SERVER_TO_CLIENT and entry.type == ack_type
        ]
        unmatched = [t for t in sent_at if not any(t < ack for ack in ack_at)]
        if unmatched:
            problems.append(f"{len(unmatched)} {sent_type} without {ack_type}")
    return Verdict(
        "ack_closure",
        not problems,
        (
            "; ".join(problems)
            if problems
            else "every acknowledged client event was answered"
        ),
    )


def byte_accounting(windows: list[ResponseWindow]) -> Verdict:
    problems: list[str] = []
    for index, window in enumerate(windows):
        label = f"response[{index}]"
        delta_bytes = sum(
            entry.audio_bytes
            for entry in window.deltas
            if entry.type == "response.audio.delta"
        )
        if window.audio_done is not None:
            declared = window.audio_done.event.get("output_audio_bytes")
            if isinstance(declared, int) and declared != delta_bytes:
                problems.append(
                    f"{label} audio bytes: {delta_bytes} summed vs {declared} declared"
                )
        elif window.done is not None and delta_bytes:
            done_status = str(window.done.event.get("response", {}).get("status", ""))
            if done_status != "cancelled":
                problems.append(
                    f"{label} streamed {delta_bytes} audio bytes without response.audio.done"
                )
    return Verdict(
        "byte_accounting",
        not problems,
        "; ".join(problems) if problems else "audio byte totals are consistent",
    )


def error_accounting(manifest: SessionManifest, entries: list[LogEntry]) -> Verdict:
    errors = [
        entry.event
        for entry in entries
        if entry.direction == SERVER_TO_CLIENT and entry.type == "error"
    ]
    expected = manifest.scenario.startswith(NEGATIVE_SCENARIO_PREFIX)
    if expected:
        passed = bool(errors)
        detail = f"{len(errors)} error event(s) recorded as expected by scenario"
    else:
        passed = not errors
        detail = (
            "no server error events"
            if passed
            else f"{len(errors)} unexpected server error event(s)"
        )
    return Verdict("error_accounting", passed, detail)


def clean_shutdown(entries: list[LogEntry]) -> Verdict:
    problems: list[str] = []
    close = [
        entry
        for entry in entries
        if entry.direction == CLIENT_TO_SERVER and entry.type == "client.close"
    ]
    if not close:
        return Verdict("clean_shutdown", False, "session never logged client.close")
    close_t = close[-1].t_s
    for entry in entries:
        if entry.direction == SERVER_TO_CLIENT and entry.t_s > close_t:
            problems.append(f"server event {entry.type} after client.close")
    for window in group_responses(entries):
        if window.done is not None and window.done.t_s > close_t:
            problems.append("response.done after client.close")
    return Verdict(
        "clean_shutdown",
        not problems,
        "; ".join(problems) if problems else "session closed with nothing in flight",
    )


def timing_profile(
    manifest: SessionManifest, entries: list[LogEntry], windows: list[ResponseWindow]
) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "frame_ms": manifest.frame_ms,
        "responses": [],
    }
    for index, window in enumerate(windows):
        first_delta_t = window.deltas[0].t_s if window.deltas else None
        gaps = [
            later.t_s - earlier.t_s
            for earlier, later in zip(window.deltas, window.deltas[1:])
        ]
        budget_s = manifest.frame_ms / 1000.0
        underruns = sum(1 for gap in gaps if gap > budget_s * UNDERRUN_GAP_FACTOR)
        profile["responses"].append(
            {
                "index": index,
                "first_delta_after_created_s": (
                    first_delta_t - window.created.t_s
                    if first_delta_t is not None
                    else None
                ),
                "delta_count": len(window.deltas),
                "inter_delta_gap_max_s": max(gaps) if gaps else None,
                "inter_delta_gap_mean_s": (sum(gaps) / len(gaps) if gaps else None),
                "underrun_gaps": underruns,
                "cancelled": window.cancelled_by_client,
            }
        )
    return profile


def count_events(entries: list[LogEntry]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        key = f"{entry.direction}.{entry.type or 'unknown'}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def replay(manifest: SessionManifest, entries: list[LogEntry]) -> ReplayReport:
    windows = group_responses(entries)
    expect_responses = not manifest.scenario.startswith(NEGATIVE_SCENARIO_PREFIX)
    verdicts = [
        session_ordering(entries),
        response_lifecycle(windows, expect_responses=expect_responses),
        ack_closure(entries),
        byte_accounting(windows),
        error_accounting(manifest, entries),
        clean_shutdown(entries),
    ]
    return ReplayReport(
        verdicts=verdicts,
        timing_profile=timing_profile(manifest, entries, windows),
        counts=count_events(entries),
        errors=[
            entry.event
            for entry in entries
            if entry.direction == SERVER_TO_CLIENT and entry.type == "error"
        ],
    )
