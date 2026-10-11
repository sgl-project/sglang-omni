# SPDX-License-Identifier: Apache-2.0
"""Export and score Full-Duplex-Bench v1.5 captures with pinned reference code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.duplex.reference_asr import run_asr
from benchmarks.duplex.reference_behavior import run_judge, run_prepare_judge
from benchmarks.duplex.reference_capture import TraceFormat
from benchmarks.duplex.reference_core import Engine, HashCache
from benchmarks.duplex.reference_custom_judge import run_custom
from benchmarks.duplex.reference_export import export_runs
from benchmarks.duplex.reference_report import render_report
from benchmarks.duplex.reference_source import verify_reference
from benchmarks.duplex.reference_summary import run_summarize
from benchmarks.duplex.reference_timing import run_timing


def parse_pairs(values: list[str], flag: str) -> dict[str, Path]:
    pairs = {}
    for value in values or []:
        name, separator, path = value.partition("=")
        if not separator or name in pairs:
            raise SystemExit(f"{flag} expects unique NAME=PATH, got {value!r}")
        else:
            pass
        pairs[name] = Path(path)
    return pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="phase", required=True)
    report = commands.add_parser("report", help="Print saved results without rescoring")
    report.add_argument("--scores", type=Path, required=True)
    report.add_argument("--engine", required=True, help="Engine label in summary.json")
    report.add_argument("--replay", type=Path, help="Saved offline replay receipt")
    report.add_argument(
        "--semantic-summary", type=Path, help="Separate custom A/F/U quality summary"
    )
    export = commands.add_parser(
        "export", help="Build fixed-window audio from saved runs"
    )
    export.add_argument(
        "--engine", required=True, help="Engine or cohort label for reports"
    )
    export.add_argument(
        "--trace-format",
        choices=[trace_format.value for trace_format in TraceFormat],
        help="Capture encoding (default: realtime-pcm16-v1)",
    )
    export.add_argument("--run", type=Path, action="append", required=True)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--dataset-root", type=Path)
    export.add_argument("--only", action="append")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--reference-source", type=Path, required=True)
    common.add_argument("--out", type=Path, required=True)
    common.add_argument(
        "--tree", action="append", required=True, help="ENGINE=reference-audio tree"
    )
    common.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="ENGINE=manifest (default tree/reference-manifest.json)",
    )
    common.add_argument(
        "--manifest-projection",
        type=Path,
        help="Python file defining project(doc) -> canonical manifest",
    )
    common.add_argument(
        "--only", action="append", default=[], help="category/id subset (validation)"
    )

    asr = commands.add_parser("asr", parents=[common])
    asr.add_argument("--nemo", required=True, help="local parakeet-tdt-0.6b-v2.nemo")
    asr.add_argument("--nemo-sha256")
    asr.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    timing = commands.add_parser("timing", parents=[common])
    timing.add_argument(
        "--audio-loader", choices=("auto", "official", "soundfile"), default="auto"
    )
    commands.add_parser("prepare-judge", parents=[common])
    judge = commands.add_parser("judge", parents=[common])
    judge.add_argument("--judge", required=True)
    judge.add_argument("--api-key-env", default="OPENAI_API_KEY")
    judge.add_argument("--base-url", help="Configured OpenAI-compatible endpoint")
    judge.add_argument("--timeout-s", type=float, default=120.0)
    judge.add_argument("--retry-sleep-s", type=float, default=5.0)
    summarize = commands.add_parser("summarize", parents=[common])
    summarize.add_argument("--bootstrap", type=int, default=2000)
    summarize.add_argument("--seed", type=int, default=20260925)
    work_commands = [asr, timing, judge]
    for phase in ("custom-judge", "custom-summarize"):
        custom = commands.add_parser(phase, parents=[common])
        custom.add_argument("--source-scores", type=Path, required=True)
        custom.add_argument("--judge-config", type=Path, required=True)
        if phase == "custom-judge":
            work_commands.append(custom)
            custom.add_argument("--base-url", required=True)
            custom.add_argument("--api-key-env", default="CUSTOM_JUDGE_API_KEY")
            custom.add_argument("--timeout-s", type=float, default=120.0)
            custom.add_argument("--retry-sleep-s", type=float, default=5.0)
        else:
            pass
    for command in work_commands:
        limit = command.add_mutually_exclusive_group()
        limit.add_argument("--limit", type=int, help="max work units this invocation")
        if command is judge:
            limit.add_argument(
                "--max-requests", dest="limit", type=int, help="Alias for --limit"
            )
        else:
            pass
        command.add_argument("--retry-failed", action="store_true")
    return parser


def open_engines(args: argparse.Namespace) -> list[Engine]:
    trees = parse_pairs(args.tree, "--tree")
    manifests = parse_pairs(args.manifest, "--manifest")
    if set(manifests) - set(trees):
        raise SystemExit("--manifest names an unknown engine")
    else:
        pass
    if args.phase in ("custom-judge", "custom-summarize"):
        scores = args.source_scores
        for name in trees:
            for filename in (
                "manifest-receipt.json",
                "source-manifest.json",
                "projected-manifest.json",
            ):
                if not (scores / "engines" / name / filename).is_file():
                    raise SystemExit(
                        f"--source-scores lacks existing {name}/{filename}"
                    )
                else:
                    pass
    else:
        scores = args.out
    return [
        Engine(
            scores,
            name,
            tree,
            manifests.get(name, tree / "reference-manifest.json"),
            args.manifest_projection,
        )
        for name, tree in sorted(trees.items())
    ]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.phase == "report":
        try:
            report = render_report(
                args.scores, args.engine, args.replay, args.semantic_summary
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(report)
        return 0
    else:
        pass
    if args.phase == "export":
        try:
            result = export_runs(
                args.run,
                args.out,
                args.engine,
                args.only,
                args.dataset_root,
                trace_format=args.trace_format,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps(result["counts"], sort_keys=True))
        return 0
    else:
        pass
    for name in ("limit", "bootstrap"):
        value = vars(args).get(name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
        else:
            pass
    for tree in parse_pairs(args.tree, "--tree").values():
        if args.out.resolve().is_relative_to(tree.resolve()):
            parser.error("--out must be outside every source audio tree")
        else:
            pass
    if args.phase in ("custom-judge", "custom-summarize"):
        sources = [args.source_scores, *parse_pairs(args.tree, "--tree").values()]
        for source in sources:
            if args.out.resolve().is_relative_to(
                source.resolve()
            ) or source.resolve().is_relative_to(args.out.resolve()):
                parser.error(
                    "custom --out must be independent of --source-scores and audio trees"
                )
            else:
                pass
        if args.phase == "custom-judge" and (
            not args.base_url.startswith(("http://", "https://"))
            or args.timeout_s <= 0
            or args.retry_sleep_s < 0
        ):
            parser.error("custom judge requires an HTTP(S) endpoint and valid timeouts")
        else:
            pass
    else:
        pass
    paths = verify_reference(args.reference_source)
    args.out.mkdir(parents=True, exist_ok=True)
    engines = open_engines(args)
    hashes = HashCache(args.out / "hash-cache.json")
    if args.phase == "asr":
        result = run_asr(args, engines, paths, hashes)
    elif args.phase == "timing":
        result = run_timing(args, engines, paths, hashes)
    elif args.phase == "prepare-judge":
        result = run_prepare_judge(args, engines, paths)
    elif args.phase == "judge":
        result = run_judge(args, engines, paths)
    elif args.phase in ("custom-judge", "custom-summarize"):
        result = run_custom(args, engines, paths)
    else:
        run_summarize(args, engines, paths)
        result = {"summary": str(args.out / "summary.json")}
    print(json.dumps({"phase": args.phase, "result": dict(result)}, sort_keys=True))
    failures = (
        "failed",
        "missing_audio",
        "invalid_transcript",
        "invalid_intervals",
        "model_mismatch",
        "invalid_label",
        "stale_request",
        "asr_audio_changed",
        "invalid_finish",
        "result_request_mismatch",
    )
    return int(
        any(
            count and (key == status or key.endswith("_" + status))
            for key, count in result.items()
            for status in failures
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
else:
    pass
