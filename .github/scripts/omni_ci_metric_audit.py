#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect Omni CI benchmark metrics and detect trend regressions."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import smtplib
import statistics
import subprocess
import sys
from email.message import EmailMessage
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_BASELINE_WINDOW = 5
HIGH_CARDINALITY_KEYS = {
    "per_request",
    "per_requests",
    "per_sample",
    "per_samples",
    "records",
    "samples",
}
SKIP_METRIC_KEYS = {
    "base_url",
    "lang",
    "model",
    "repo_id",
    "split",
}
SKIP_METRIC_PREFIXES = ("config.", "metadata.")
LOWER_BETTER_TOKENS = (
    "error",
    "failed",
    "failure",
    "latency",
    "loss",
    "rtf",
    "timeout",
    "ttfp",
    "ttft",
    "wer",
)
HIGHER_BETTER_TOKENS = (
    "accuracy",
    "completed_requests",
    "correct",
    "mos",
    "pass_rate",
    "similarity",
    "success",
    "throughput",
    "tok_per_req_s",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)


def read_matches(matches_file: str | None) -> list[Path]:
    if not matches_file:
        return []
    path = Path(matches_file)
    if not path.exists():
        return []
    matches = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if raw:
                matches.append(Path(raw))
    return sorted(dict.fromkeys(matches))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def metric_direction(json_path: str) -> str:
    leaf = json_path.rsplit(".", 1)[-1].lower()
    lowered = json_path.lower()
    if any(lowered.startswith(prefix) for prefix in SKIP_METRIC_PREFIXES):
        return "neutral"
    if leaf in SKIP_METRIC_KEYS or leaf.startswith("max_"):
        return "neutral"
    if leaf.startswith("total_") or leaf.endswith("_total"):
        return "neutral"
    if any(token in leaf for token in LOWER_BETTER_TOKENS):
        return "lower"
    if any(token in leaf for token in HIGHER_BETTER_TOKENS):
        return "higher"
    return "neutral"


def metric_unit(json_path: str) -> str | None:
    leaf = json_path.rsplit(".", 1)[-1].lower()
    if leaf.endswith("_s") or "_s_" in leaf:
        return "s"
    if "qps" in leaf:
        return "req/s"
    if "rtf" in leaf:
        return "rtf"
    if "wer" in leaf:
        return "wer"
    if "accuracy" in leaf:
        return "ratio"
    return None


def _flatten_metrics(
    value: Any,
    *,
    prefix: str,
    out: list[dict[str, Any]],
) -> None:
    if isinstance(value, dict):
        for key in sorted(value):
            if key in HIGH_CARDINALITY_KEYS:
                continue
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_metrics(value[key], prefix=next_prefix, out=out)
        return
    if isinstance(value, list):
        return
    if not _is_number(value):
        return
    if not math.isfinite(float(value)):
        return
    direction = metric_direction(prefix)
    if direction == "neutral":
        return
    out.append(
        {
            "json_path": prefix,
            "name": prefix.rsplit(".", 1)[-1],
            "value": float(value),
            "direction": direction,
            "unit": metric_unit(prefix),
        }
    )


def prune_payload(value: Any, *, depth: int = 0) -> Any:
    if depth > 12:
        return {"omitted": "max_depth"}
    if isinstance(value, dict):
        pruned: dict[str, Any] = {}
        for key, item in value.items():
            if key in HIGH_CARDINALITY_KEYS:
                count = len(item) if isinstance(item, list) else None
                pruned[key] = {"omitted": True, "count": count}
            else:
                pruned[key] = prune_payload(item, depth=depth + 1)
        return pruned
    if isinstance(value, list):
        if len(value) > 20:
            return {"omitted": True, "count": len(value)}
        return [prune_payload(item, depth=depth + 1) for item in value]
    return value


def artifact_id(path: Path) -> str:
    return path.name


def relpath_for_display(path: Path, roots: list[Path]) -> str:
    for root in roots:
        try:
            return str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            pass
    return str(path)


def load_pull_request_metadata() -> dict[str, Any] | None:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path or not Path(event_path).exists():
        return None
    try:
        event = load_json(Path(event_path))
    except Exception:
        return None
    pr = event.get("pull_request")
    if not isinstance(pr, dict):
        return None
    return {
        "number": pr.get("number"),
        "title": pr.get("title"),
        "url": pr.get("html_url"),
        "base_ref": pr.get("base", {}).get("ref"),
        "head_ref": pr.get("head", {}).get("ref"),
        "head_sha": pr.get("head", {}).get("sha"),
        "author": pr.get("user", {}).get("login"),
    }


def collect_audit(
    *,
    stage_label: str,
    matches_file: str | None,
    artifact_search_root: str | None,
    artifact_path_globs: str | None,
) -> dict[str, Any]:
    matches = read_matches(matches_file)
    workspace = Path(os.environ.get("GITHUB_WORKSPACE", os.getcwd()))
    roots = [workspace]
    if artifact_search_root:
        roots.insert(0, Path(artifact_search_root))

    event: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "repository": os.environ.get("GITHUB_REPOSITORY"),
        "workflow": {
            "name": os.environ.get("GITHUB_WORKFLOW"),
            "job": os.environ.get("GITHUB_JOB"),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "event_name": os.environ.get("GITHUB_EVENT_NAME"),
            "ref": os.environ.get("GITHUB_REF"),
            "sha": os.environ.get("GITHUB_SHA"),
            "server_url": os.environ.get("GITHUB_SERVER_URL"),
        },
        "pull_request": load_pull_request_metadata(),
        "stage": {
            "label": stage_label,
            "artifact_search_root": artifact_search_root,
            "artifact_path_globs": artifact_path_globs,
        },
        "artifact_files": [],
        "metrics": [],
    }

    for attempt_ordinal, path in enumerate(matches, start=1):
        file_entry: dict[str, Any] = {
            "path": str(path),
            "relative_path": relpath_for_display(path, roots),
            "artifact_id": artifact_id(path),
            "attempt_ordinal": attempt_ordinal,
        }
        try:
            data = load_json(path)
            file_entry["sha256"] = sha256_file(path)
            file_entry["summary"] = prune_payload(data)
            metrics: list[dict[str, Any]] = []
            _flatten_metrics(data, prefix="", out=metrics)
            for metric in metrics:
                metric.update(
                    {
                        "stage_label": stage_label,
                        "artifact_id": file_entry["artifact_id"],
                        "source_path": file_entry["relative_path"],
                        "attempt_ordinal": attempt_ordinal,
                    }
                )
                metric["metric_id"] = (
                    f"{stage_label}|{file_entry['artifact_id']}|"
                    f"{metric['json_path']}"
                )
            file_entry["metrics"] = metrics
            event["metrics"].extend(metrics)
        except Exception as exc:
            file_entry["error"] = str(exc)
        event["artifact_files"].append(file_entry)
    return event


def latest_metrics_by_id(event: dict[str, Any]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for metric in event.get("metrics", []):
        metric_id = metric.get("metric_id")
        if not metric_id:
            continue
        previous = latest.get(metric_id)
        if previous is None or int(metric.get("attempt_ordinal") or 0) >= int(
            previous.get("attempt_ordinal") or 0
        ):
            latest[metric_id] = metric
    return latest


def history_record(event: dict[str, Any]) -> dict[str, Any]:
    workflow = event.get("workflow") or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": event.get("created_at"),
        "repository": event.get("repository"),
        "workflow": {
            "name": workflow.get("name"),
            "job": workflow.get("job"),
            "run_id": workflow.get("run_id"),
            "run_attempt": workflow.get("run_attempt"),
            "sha": workflow.get("sha"),
        },
        "pull_request": event.get("pull_request"),
        "stage": event.get("stage"),
        "metrics": list(latest_metrics_by_id(event).values()),
    }


def load_history_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                print(
                    f"warning: skipping invalid history line {line_number}: {exc}",
                    file=sys.stderr,
                )
    return records


def _baseline(values: list[float], direction: str) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return None
    reverse = direction == "higher"
    best = sorted(clean, reverse=reverse)[:DEFAULT_BASELINE_WINDOW]
    return float(statistics.median(best))


def compare_event(
    event: dict[str, Any],
    history: list[dict[str, Any]],
    *,
    threshold: float,
    min_baseline_count: int,
) -> dict[str, Any]:
    prior: dict[str, list[float]] = {}
    for record in history:
        for metric in record.get("metrics", []):
            metric_id = metric.get("metric_id")
            value = metric.get("value")
            if metric_id and _is_number(value):
                prior.setdefault(metric_id, []).append(float(value))

    regressions = []
    for metric in latest_metrics_by_id(event).values():
        metric_id = metric["metric_id"]
        direction = metric.get("direction")
        if direction not in {"lower", "higher"}:
            continue
        values = prior.get(metric_id, [])
        if len(values) < min_baseline_count:
            continue
        baseline = _baseline(values, direction)
        current = float(metric["value"])
        if baseline is None or baseline == 0:
            continue
        if direction == "lower":
            ratio = (current - baseline) / abs(baseline)
        else:
            ratio = (baseline - current) / abs(baseline)
        if ratio >= threshold:
            regressions.append(
                {
                    "metric_id": metric_id,
                    "stage_label": metric.get("stage_label"),
                    "artifact_id": metric.get("artifact_id"),
                    "json_path": metric.get("json_path"),
                    "direction": direction,
                    "unit": metric.get("unit"),
                    "current": current,
                    "baseline": baseline,
                    "regression_ratio": ratio,
                    "baseline_sample_count": len(values),
                    "threshold": threshold,
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "repository": event.get("repository"),
        "workflow": event.get("workflow"),
        "pull_request": event.get("pull_request"),
        "stage": event.get("stage"),
        "history_records": len(history),
        "regressions": sorted(
            regressions,
            key=lambda item: item["regression_ratio"],
            reverse=True,
        ),
    }


def _split_recipients(raw: str) -> list[str]:
    return [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]


def build_email(alert: dict[str, Any], recipients: list[str]) -> EmailMessage:
    stage = (alert.get("stage") or {}).get("label") or "Omni CI"
    regressions = alert.get("regressions", [])
    pr = alert.get("pull_request") or {}
    workflow = alert.get("workflow") or {}
    run_id = workflow.get("run_id")
    subject = f"[sglang-omni] CI metric regression in {stage}"
    if pr.get("number"):
        subject += f" (PR #{pr['number']})"
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = (
        os.environ.get("OMNI_CI_AUDIT_SMTP_FROM") or "omni-ci-audit@github.local"
    )
    msg["To"] = ", ".join(recipients)
    lines = [
        f"Stage: {stage}",
        f"Repository: {alert.get('repository')}",
        f"Run ID: {run_id}",
    ]
    if pr.get("number"):
        lines.append(f"PR: #{pr['number']} {pr.get('title') or ''}".rstrip())
        if pr.get("url"):
            lines.append(f"PR URL: {pr['url']}")
    lines.append("")
    lines.append(f"Detected {len(regressions)} metric regression(s):")
    for item in regressions[:20]:
        unit = item.get("unit") or ""
        lines.append(
            "- {json_path}: current={current:g}{unit} baseline={baseline:g}{unit} "
            "regression={ratio:.1%} direction={direction}".format(
                json_path=item["json_path"],
                current=item["current"],
                unit=unit,
                baseline=item["baseline"],
                ratio=item["regression_ratio"],
                direction=item["direction"],
            )
        )
    msg.set_content("\n".join(lines) + "\n")
    return msg


def send_email(alert: dict[str, Any], recipients: list[str]) -> tuple[bool, str]:
    if not recipients:
        return False, "no recipients configured"
    msg = build_email(alert, recipients)
    smtp_host = os.environ.get("OMNI_CI_AUDIT_SMTP_HOST")
    if smtp_host:
        port = int(os.environ.get("OMNI_CI_AUDIT_SMTP_PORT") or 587)
        username = os.environ.get("OMNI_CI_AUDIT_SMTP_USERNAME")
        password = os.environ.get("OMNI_CI_AUDIT_SMTP_PASSWORD")
        use_ssl = (os.environ.get("OMNI_CI_AUDIT_SMTP_SSL") or "0") == "1"
        if use_ssl:
            server: smtplib.SMTP = smtplib.SMTP_SSL(smtp_host, port, timeout=30)
        else:
            server = smtplib.SMTP(smtp_host, port, timeout=30)
        with server:
            if not use_ssl and (os.environ.get("OMNI_CI_AUDIT_SMTP_TLS") or "1") == "1":
                server.starttls()
            if username:
                server.login(username, password or "")
            server.send_message(msg)
        return True, "sent via SMTP"

    sendmail = os.environ.get("OMNI_CI_AUDIT_SENDMAIL") or shutil.which("sendmail")
    if sendmail:
        proc = subprocess.run(
            [sendmail, "-t", "-oi"],
            input=msg.as_bytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode == 0:
            return True, "sent via sendmail"
        return False, proc.stderr.decode("utf-8", errors="replace").strip()

    return False, "no SMTP host or sendmail binary configured"


def slugify(value: str | None, default: str) -> str:
    raw = value or default
    out = []
    for char in raw.lower():
        if char.isalnum():
            out.append(char)
        elif char in {".", "-", "_"}:
            out.append(char)
        else:
            out.append("-")
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or default


def storage_path(event: dict[str, Any]) -> str:
    repository = slugify(str(event.get("repository") or "unknown"), "unknown")
    workflow = event.get("workflow") or {}
    stage = event.get("stage") or {}
    pr = event.get("pull_request") or {}
    pr_or_ref = (
        f"pr-{pr['number']}"
        if pr.get("number") is not None
        else slugify(str(workflow.get("ref") or "no-pr"), "no-pr")
    )
    run_id = slugify(str(workflow.get("run_id") or "no-run"), "no-run")
    attempt = slugify(str(workflow.get("run_attempt") or "1"), "1")
    job = slugify(str(workflow.get("job") or "unknown-job"), "unknown-job")
    stage_slug = slugify(str(stage.get("label") or "unknown-stage"), "unknown-stage")
    return (
        f"events/{repository}/{pr_or_ref}/run-{run_id}/"
        f"attempt-{attempt}/{job}/{stage_slug}.json"
    )


def cmd_collect(args: argparse.Namespace) -> int:
    event = collect_audit(
        stage_label=args.stage_label,
        matches_file=args.matches_file,
        artifact_search_root=args.artifact_search_root,
        artifact_path_globs=args.artifact_path_globs,
    )
    write_json(Path(args.output), event)
    print(
        f"wrote audit record with {len(event['artifact_files'])} file(s) and "
        f"{len(event['metrics'])} metric(s): {args.output}"
    )
    return 0


def cmd_history_record(args: argparse.Namespace) -> int:
    event = load_json(Path(args.current))
    record = history_record(event)
    line = compact_json(record) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(line, encoding="utf-8")
    else:
        print(line, end="")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    event = load_json(Path(args.current))
    history = load_history_jsonl(Path(args.history_jsonl))
    alert = compare_event(
        event,
        history,
        threshold=args.threshold,
        min_baseline_count=args.min_baseline_count,
    )
    write_json(Path(args.alert_output), alert)
    regressions = alert["regressions"]
    if not regressions:
        print(
            "no metric regression detected "
            f"(history_records={alert['history_records']})"
        )
        return 0

    print(f"detected {len(regressions)} metric regression(s)")
    for item in regressions[:20]:
        print(
            "  {metric_id}: current={current:g} baseline={baseline:g} "
            "regression={regression_ratio:.1%}".format(**item)
        )
    if args.send_email:
        ok, detail = send_email(alert, _split_recipients(args.email_to))
        if ok:
            print(f"email alert sent: {detail}")
        else:
            print(f"warning: email alert not sent: {detail}", file=sys.stderr)
    return 0


def cmd_storage_path(args: argparse.Namespace) -> int:
    event = load_json(Path(args.current))
    print(storage_path(event))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--stage-label", required=True)
    collect.add_argument("--matches-file")
    collect.add_argument("--artifact-search-root")
    collect.add_argument("--artifact-path-globs")
    collect.add_argument("--output", required=True)
    collect.set_defaults(func=cmd_collect)

    check = subparsers.add_parser("check")
    check.add_argument("--current", required=True)
    check.add_argument("--history-jsonl", required=True)
    check.add_argument("--alert-output", required=True)
    check.add_argument("--threshold", type=float, default=0.10)
    check.add_argument("--min-baseline-count", type=int, default=1)
    check.add_argument("--email-to", default="")
    check.add_argument("--send-email", action="store_true")
    check.set_defaults(func=cmd_check)

    record = subparsers.add_parser("history-record")
    record.add_argument("--current", required=True)
    record.add_argument("--output")
    record.set_defaults(func=cmd_history_record)

    path = subparsers.add_parser("storage-path")
    path.add_argument("--current", required=True)
    path.set_defaults(func=cmd_storage_path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
