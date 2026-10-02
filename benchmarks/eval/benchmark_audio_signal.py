# SPDX-License-Identifier: Apache-2.0
"""Inspect existing WAV files on CPU and emit report-only signal diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import soundfile

from benchmarks.metrics.audio_signal import (
    DEFAULT_SILENCE_THRESHOLD_DBFS,
    DEFAULT_SILENCE_WINDOW_MS,
    analyze_audio_file,
    validate_silence_settings,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="WAV files or directories")
    parser.add_argument(
        "--output", type=Path, help="JSON destination; defaults to stdout"
    )
    parser.add_argument(
        "--silence-window-ms", type=float, default=DEFAULT_SILENCE_WINDOW_MS
    )
    parser.add_argument(
        "--silence-threshold-dbfs", type=float, default=DEFAULT_SILENCE_THRESHOLD_DBFS
    )
    arguments = parser.parse_args(argv)
    try:
        validate_silence_settings(
            arguments.silence_window_ms, arguments.silence_threshold_dbfs
        )
    except ValueError as error:
        parser.error(str(error))

    paths: set[Path] = set()
    for path in arguments.paths:
        if path.is_dir():
            paths.update(
                child.resolve()
                for child in path.rglob("*")
                if child.is_file() and child.suffix.lower() == ".wav"
            )
        else:
            paths.add(path.resolve())
    if not paths:
        parser.error("no WAV files found")
    elif arguments.output is not None and (
        arguments.output.resolve() in paths
        or (
            arguments.output.exists()
            and any(path.exists() and arguments.output.samefile(path) for path in paths)
        )
    ):
        parser.error("output must not overwrite an input file")
    else:
        pass

    reports = []
    errors: list[dict[str, str]] = []
    for path in sorted(paths):
        try:
            report = analyze_audio_file(
                path,
                silence_window_ms=arguments.silence_window_ms,
                silence_threshold_dbfs=arguments.silence_threshold_dbfs,
            )
            reports.append(asdict(report))
        except (OSError, ValueError, soundfile.LibsndfileError) as error:
            errors.append({"path": str(path), "error": str(error)})

    output = json.dumps(
        {
            "config": {
                "silence_window_ms": arguments.silence_window_ms,
                "silence_threshold_dbfs": arguments.silence_threshold_dbfs,
            },
            "per_file": reports,
            "errors": errors,
        },
        indent=2,
        allow_nan=False,
    )
    if arguments.output is None:
        print(output)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(output + "\n", encoding="utf-8")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
