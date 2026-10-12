# SPDX-License-Identifier: Apache-2.0
"""Format saved evaluation evidence without executing capture or scoring."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from benchmarks.duplex.reference_core import (
    C_LABELS,
    VARIANTS,
    canonical_hash,
    read_json,
)
from benchmarks.duplex.report_models import (
    ManifestReceipt,
    ReferenceSummary,
    ReplayReceipt,
    ReportManifest,
    ReportVariant,
    SemanticSummary,
    VariantName,
)


def format_rows(rows: list[tuple[str, str]]) -> str:
    width = max(len(label) for label, _ in rows)
    return "\n".join(f"{label:<{width}}  {value}" for label, value in rows)


def format_statuses(counts: dict[str, int]) -> str:
    return (
        " + ".join(
            f"{count} {status}" for status, count in sorted(counts.items()) if count
        )
        or "none recorded"
    )


def format_number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def format_percent(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.1f}%" if denominator else "n/a"


def replay_rows(
    path: Path | None, variants: dict[tuple[str, VariantName], ReportVariant]
) -> list[tuple[str, str]]:
    if path is None:
        return [("Offline replay agreement", "not supplied")]
    else:
        receipt = ReplayReceipt.model_validate_json(path.read_bytes(), strict=True)
    sessions = {
        (session.sample, session.variant): session for session in receipt.samples
    }
    if (
        len(sessions) != len(receipt.samples)
        or set(sessions) - set(variants)
        or receipt.selected != len(variants)
        or receipt.replayed != len(sessions)
    ):
        raise ValueError("Replay receipt population does not match the report")
    else:
        pass
    for session_key, session in sessions.items():
        diagnostics = variants[session_key].protocol_diagnostics
        if diagnostics is None or session.recorded_status != diagnostics.status:
            raise ValueError(f"Replay recorded status differs for {session_key}")
        else:
            pass
    counts = Counter(session.status for session in sessions.values())
    return [
        ("Offline replay agreement", f"{counts['match']} / {len(variants)} matched"),
        ("Replay receipt statuses", format_statuses(dict(counts))),
        ("Replay receipt missing sessions", str(len(variants) - len(sessions))),
        ("Replay evidence", "saved receipt; no trace hash binding or new replay"),
    ]


def semantic_report(path: Path) -> str:
    summary = SemanticSummary.model_validate_json(path.read_bytes(), strict=True)
    rows = []
    for axis, result in {
        **summary.overall_axes,
        "joint": summary.overall_joint,
    }.items():
        if result.accepted + result.failed + result.unresolved != result.selected:
            raise ValueError(f"Semantic denominator disagrees for {axis}")
        else:
            pass
        rows.append(
            (
                axis.replace("_", " "),
                f"quality {format_percent(result.accepted, result.accepted + result.failed)}; "
                f"coverage {format_percent(result.accepted + result.failed, result.selected)}; "
                f"A/F/U {result.accepted}/{result.failed}/{result.unresolved}",
            )
        )
    return "\n".join(
        [
            "Custom semantic quality (not official FDB behavior)",
            f"Source: {path}",
            f"Judge input SHA-256: {summary.inputs_sha256}",
            "Separate supplied cohort; not joined to the reference population.",
            summary.scope,
            format_rows(rows),
            "Quality = accepted / (accepted + failed); coverage = resolved / selected.",
            summary.uncertainty_note,
        ]
    )


def render_report(
    scores: Path,
    engine: str,
    replay: Path | None = None,
    semantic_summary: Path | None = None,
) -> str:
    summary = ReferenceSummary.model_validate_json(
        (scores / "summary.json").read_bytes(), strict=True
    )
    if engine not in summary.engines:
        raise ValueError(f"No saved summary for engine {engine!r}")
    else:
        groups = summary.engines[engine]
    if "all" not in groups:
        raise ValueError("Summary has no overall population ('all')")
    else:
        pass
    engine_directory = scores / "engines" / engine
    manifest_path = engine_directory / "projected-manifest.json"
    manifest = ReportManifest.model_validate_json(
        manifest_path.read_bytes(), strict=True
    )
    receipt = ManifestReceipt.model_validate_json(
        (engine_directory / "manifest-receipt.json").read_bytes(), strict=True
    )
    manifest_hash = canonical_hash(read_json(manifest_path))
    if receipt.engine != engine or manifest_hash != receipt.projected_manifest_sha256:
        raise ValueError("Scoring manifest differs from its frozen receipt")
    else:
        pass
    sample_ids = [sample.sample_id for sample in manifest.samples]
    if len(set(sample_ids)) != len(sample_ids) or any(
        set(sample.variants) != set(VARIANTS) for sample in manifest.samples
    ):
        raise ValueError(
            "Manifest requires unique samples with overlap and clean variants"
        )
    else:
        pass
    population = summary.populations.get(engine)
    if population is None:
        binding = "legacy summary; manifest binding not recorded"
    elif population.manifest_sha256 != manifest_hash or sorted(
        population.sample_ids
    ) != sorted(sample_ids):
        raise ValueError("Summary population differs from the scoring manifest")
    else:
        binding = "manifest and selected sample IDs match"
    variants = {
        (sample.sample_id, name): variant
        for sample in manifest.samples
        for name, variant in sample.variants.items()
    }
    all_scores = groups["all"]
    selected_pairs = len(manifest.samples)
    selected_sessions = len(variants)
    eligible_sessions = sum(variant.eligible for variant in variants.values())
    eligible_pairs = sum(
        all(variant.eligible for variant in sample.variants.values())
        for sample in manifest.samples
    )
    timings = (
        all_scores.timing_official_overlap,
        all_scores.timing_supplementary_clean,
    )
    if any(timing.population != selected_pairs for timing in timings) or (
        all_scores.behavior.population != selected_pairs
    ):
        raise ValueError("Summary must cover the full scoring manifest, not a subset")
    else:
        pass
    asr_successes = 0
    for (variant_name, filenames), timing in zip(VARIANTS.items(), timings):
        eligible = sum(
            sample.variants[variant_name].eligible for sample in manifest.samples
        )
        successes = [
            all_scores.asr_files.get(f"{filename}:ok", 0) for filename in filenames
        ]
        if max(*successes, timing.status.get("ok", 0)) > eligible:
            raise ValueError(
                f"Successful scoring exceeds eligible {variant_name} sessions"
            )
        else:
            asr_successes += sum(successes)
    behavior = all_scores.behavior
    if (
        set(behavior.valid_label_proportions) != set(C_LABELS)
        or sum(label.count for label in behavior.valid_label_proportions.values())
        != behavior.valid_n
        or behavior.valid_n != behavior.status.get("valid", 0)
        or behavior.valid_n > eligible_pairs
    ):
        raise ValueError(
            "Official behavior label counts disagree with eligible pairs or status"
        )
    else:
        pass
    timing_successes = sum(timing.status.get("ok", 0) for timing in timings)
    protocols = Counter(
        (
            (variant.protocol_diagnostics.protocol_verdict or "not_recorded")
            if variant.protocol_diagnostics is not None
            else "not_recorded"
        )
        for variant in variants.values()
    )
    recorded_traces = sum(
        variant.source is not None and variant.source.trace_sha256 is not None
        for variant in variants.values()
    )
    rows = [
        ("Selected pairs", str(selected_pairs)),
        (
            "Selected / recorded session traces",
            f"{selected_sessions} / {recorded_traces}",
        ),
        ("Native protocol checks", f"{protocols['pass']} / {selected_sessions} passed"),
        ("Native protocol statuses", format_statuses(dict(protocols))),
        *replay_rows(replay, variants),
        ("Reference-eligible sessions", f"{eligible_sessions} / {selected_sessions}"),
        (
            "Complete eligible overlap-clean pairs",
            f"{eligible_pairs} / {selected_pairs}",
        ),
        ("Technical exclusions", str(selected_sessions - eligible_sessions)),
        (
            "Successful ASR file roles",
            f"{asr_successes} / {2 * eligible_sessions} eligible / {2 * selected_sessions} selected",
        ),
        (
            "Successful timing results",
            f"{timing_successes} / {eligible_sessions} eligible / {selected_sessions} selected",
        ),
        ("Valid official behavior labels", str(all_scores.behavior.valid_n)),
        ("Behavior pair statuses", format_statuses(all_scores.behavior.status)),
        ("Reply WER", "not available in supplied artifacts"),
        ("Concurrent-session capacity", "not available in supplied artifacts"),
        ("Summary identity", binding),
    ]
    lines = [
        f"{engine} FDB v1.5 reference result",
        f"Saved scores: {scores}",
        f"Reference revision: {summary.reference_revision}",
        format_rows(rows),
        "\nRecorded traces may include partial sessions; protocol passes are not quality scores.",
        "This command formats saved evidence; it does not revalidate audio or rerun scoring.",
        "\nScoring receipt statuses",
        format_rows(
            [
                ("ASR roles", format_statuses(all_scores.asr_files)),
                ("Timing overlap", format_statuses(timings[0].status)),
                ("Timing clean", format_statuses(timings[1].status)),
            ]
        ),
        "Empty-transcript and timing label counts are diagnostics within the status counts.",
        "\nReference whole-file intervals (seconds; not serving or event-specific latency)",
        "group / variant / interval: mean, median, mean 95% CI; intervals, samples, empty samples",
    ]
    for group_name, group in sorted(groups.items()):
        for variant_name, timing in (
            ("overlap", group.timing_official_overlap),
            ("clean (supplementary)", group.timing_supplementary_clean),
        ):
            for interval_name, distribution in (
                ("stop", timing.official_all_intervals.stop),
                ("response", timing.official_all_intervals.response),
            ):
                confidence = distribution.bootstrap_pooled_mean.ci95
                interval = (
                    "n/a"
                    if confidence is None
                    else f"[{format_number(confidence[0])}, {format_number(confidence[1])}]"
                )
                lines.append(
                    f"{group_name} / {variant_name} / {interval_name}: "
                    f"{format_number(distribution.mean_s)}, {format_number(distribution.median_s)}, "
                    f"{interval}; {distribution.interval_n}, {distribution.samples}, "
                    f"{distribution.zero_interval_samples}"
                )
    lines.extend(
        [
            "Empty interval sets remain n/a; observation-window endings can be censored.",
            "CIs describe sample uncertainty for one generation, not generation or judge variance.",
            "\nOfficial behavior label distribution (not accuracy)",
            format_rows(
                [
                    (
                        label,
                        f"{result.count} / {all_scores.behavior.valid_n}; "
                        + format_percent(result.count, all_scores.behavior.valid_n),
                    )
                    for label, result in sorted(
                        all_scores.behavior.valid_label_proportions.items()
                    )
                ]
            ),
        ]
    )
    if semantic_summary is not None:
        lines.extend(["", semantic_report(semantic_summary)])
    else:
        pass
    return "\n".join(lines)
