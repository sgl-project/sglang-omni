#!/usr/bin/env python3
"""Summarize three TestMeetingOverlap output directories without running models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.profile_loop import summarize_reports, write_json

METHODS = ("one_shot", "serial", "overlap")


def summarize_run_directories(
    run_directories: list[Path], output_dir: Path
) -> dict[str, object]:
    if len(run_directories) != 3:
        raise ValueError("exactly three overlap run directories are required")
    methods = {}
    for method in METHODS:
        paths = [directory / method / "result.json" for directory in run_directories]
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        summary = summarize_reports(reports)
        summary["raw_results"] = [str(path) for path in paths]
        write_json(output_dir / method / "summary.json", summary)
        methods[method] = summary
    return methods


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs=3, type=Path)
    parser.add_argument("output_dir", type=Path)
    arguments = parser.parse_args()
    try:
        methods = summarize_run_directories(arguments.runs, arguments.output_dir)
    except (
        OSError,
        StopIteration,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as error:
        parser.error(str(error))
    print(json.dumps(methods, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
