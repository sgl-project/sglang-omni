# SPDX-License-Identifier: Apache-2.0
"""Summarize recorded speculative decoding runs without changing goldens."""

import json
import statistics
from pathlib import Path

from check_golden import COMPARED_FIELDS, check, chip, edit_distance, quality, tokens


def write_report(
    output_directory: Path,
    golden_path: Path,
    corpus_path: Path,
    long_path: Path,
    smoke: bool,
) -> bool:
    golden = json.loads(golden_path.read_text())
    manifest = {
        clip["id"]: clip
        for clip in map(json.loads, corpus_path.read_text().splitlines())
    }
    long_clips = json.loads(long_path.read_text())
    provenance = json.loads((output_directory / "provenance.json").read_text())
    all_passes = []
    for index, mode in enumerate(provenance["pass_order"]):
        records = output_directory / f"{index + 1}-{mode}" / "requests.jsonl"
        all_passes.append(
            [
                row
                for row in map(json.loads, records.read_text().splitlines())
                if not row["warmup"]
            ]
        )
    failures = []
    summary = []
    for pass_index, rows in enumerate(all_passes):
        corpus_rows = {
            row["id"]: {field: row[field] for field in COMPARED_FIELDS}
            for row in rows
            if row["suite"] == "corpus"
        }
        metrics = quality(
            manifest, {clip_id: row["text"] for clip_id, row in corpus_rows.items()}
        )
        if not smoke:
            # note (ruiling): Speculative block arithmetic can differ at ties, so score it against the same published tolerances.
            comparison = golden
            if provenance["pass_order"][pass_index] == "draft":
                reference = golden["devices"].get(
                    chip(), next(iter(golden["devices"].values()))
                )
                comparison = {**golden, "devices": {"reference": reference}}
            else:
                pass
            lines, violations = check(comparison, chip(), corpus_rows, metrics)
            (output_directory / f"golden-{pass_index + 1}.txt").write_text(
                "\n".join(lines + violations) + "\n"
            )
            failures.extend(
                f"pass {pass_index + 1}: {violation}" for violation in violations
            )
        else:
            pass
        groups = {}
        for suite, language in sorted({(row["suite"], row["lang"]) for row in rows}):
            group = [
                row for row in rows if (row["suite"], row["lang"]) == (suite, language)
            ]
            seconds = sorted(row["seconds"] for row in group)
            rounds = sum(row["verify_round_count"] for row in group)
            accepted = sum(row["accepted_draft_token_count"] for row in group)
            proposed = sum(row["proposed_draft_token_count"] for row in group)
            groups[f"{suite}/{language}"] = dict(
                clips=len(group),
                seconds=sum(seconds),
                p50_seconds=statistics.median(seconds),
                p95_seconds=seconds[min(len(seconds) - 1, int(len(seconds) * 0.95))],
                generated_tokens=sum(row["generated_token_count"] for row in group),
                verify_rounds=rounds,
                accepted_draft_tokens=accepted,
                proposed_draft_tokens=proposed,
                accepted_tokens_per_round=accepted / rounds if rounds else None,
                acceptance_rate=accepted / proposed if proposed else None,
                length_limited=[
                    row["id"] for row in group if row["finish_reason"] == "length"
                ],
            )
        long_errors = long_words = 0
        for row in rows:
            if row["suite"] == "long":
                reference = tokens(
                    next(
                        clip["reference"]
                        for clip in long_clips
                        if clip["id"] == row["id"]
                    ),
                    "en",
                )
                long_errors += edit_distance(reference, tokens(row["text"], "en"))
                long_words += len(reference)
            else:
                pass
        summary.append(
            dict(
                mode=provenance["pass_order"][pass_index],
                metrics=metrics,
                groups=groups,
                long_wer=long_errors / long_words if long_words else None,
            )
        )
    parity = []
    for index in (1, 2, 3):
        differences = [
            baseline["id"]
            for baseline, actual in zip(all_passes[0], all_passes[index], strict=True)
            if any(baseline[field] != actual[field] for field in COMPARED_FIELDS)
        ]
        parity.append(dict(pass_number=index + 1, different_clips=differences))
    if not smoke:
        baseline_metrics = summary[0]["metrics"]
        for index in (1, 2):
            for name, value in summary[index]["metrics"].items():
                if (
                    value - baseline_metrics[name]
                    > golden["tolerance"]["metric_pp"] / 100
                ):
                    failures.append(
                        f"draft pass {index + 1}: {name} regressed against the paired plain run"
                    )
                else:
                    pass
            if (
                summary[index]["long_wer"] - summary[0]["long_wer"]
                > golden["tolerance"]["metric_pp"] / 100
            ):
                failures.append(
                    f"draft pass {index + 1}: long-audio WER regressed against plain"
                )
            else:
                pass
        for index, rows in enumerate(all_passes):
            for row in rows:
                if row["finish_reason"] == "length" and row["lang"] != "none":
                    failures.append(
                        f"pass {index + 1}: speech clip {row['id']} reached the output limit"
                    )
                else:
                    pass
    else:
        pass
    speed_ratios = {}
    for group in summary[0]["groups"]:
        ordinary = (
            summary[0]["groups"][group]["seconds"]
            + summary[3]["groups"][group]["seconds"]
        ) / 2
        accelerated = (
            summary[1]["groups"][group]["seconds"]
            + summary[2]["groups"][group]["seconds"]
        ) / 2
        speed_ratios[group] = ordinary / accelerated
    report = dict(
        status=(
            "smoke_only"
            if smoke
            else ("needs_review" if failures else "golden_checks_passed")
        ),
        failures=failures,
        plain_seconds_over_draft_seconds=speed_ratios,
        passes=summary,
        parity=parity,
        note="Review paired quality, length-limited clips, repeat variability and long-audio transcripts before reporting speedup. Golden files were not modified.",
    )
    (output_directory / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "outputs": str(output_directory),
                "failures": failures,
            },
            indent=2,
        )
    )
    return bool(failures)
