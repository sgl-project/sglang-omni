# SPDX-License-Identifier: Apache-2.0
"""Checks a native model's parity tool against its golden outputs on the frozen corpus.

    python check_model_golden.py --runtime-bin DIR --data-root DIR --golden FILE
        [--write] [--outputs DIR]
    python check_model_golden.py --golden FILE --import OUTPUTS

Like check_golden.py, with the same per-chip golden outputs and tolerance, for a
model served by its own binary. The golden file also names the parity tool
(whisper_transcribe, ...), its request flags (a true one passed bare, a false
one left out), the language each clip language is sent with (a user with that
main language), and optionally the flags Voxt adds for clips past some
duration. A flag given as {"model": repo} is that provisioned model's
directory. Every corpus clip (or, with clips_over_seconds, every clip longer
than that) is transcribed, one tool run per language sent and length.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TypedDict

from check_golden import COMPARED_FIELDS, model_directory, run


class ModelReference(TypedDict):
    model: str


RequestValue = bool | str | int | float | ModelReference


def request_flags(data_root: Path, request: dict[str, RequestValue]) -> list[str]:
    flags = []
    for field, value in request.items():
        option = f"--{field.replace('_', '-')}"
        if value is True:
            flags.append(option)
        elif value is False:
            pass
        elif isinstance(value, dict):
            flags += [option, str(model_directory(data_root, value["model"]))]
        else:
            flags += [option, str(value)]
    return flags


def transcribe(
    command: list[str], clips: list[Path], language: str | None
) -> dict[str, dict]:
    language_flags = ["--language", language] if language is not None else []
    completed = subprocess.run(
        command + language_flags + [str(clip) for clip in clips],
        check=True,
        capture_output=True,
        text=True,
    )
    results = {}
    for line in completed.stdout.splitlines():
        row = json.loads(line)
        results[Path(row["file"]).stem] = {
            field: row[field] for field in COMPARED_FIELDS
        }
    return results


def transcribe_with_tool(
    runtime_bin: Path, data_root: Path, golden: dict, manifest: dict[str, dict]
) -> dict[str, dict]:
    command = [
        str(runtime_bin / golden["tool"]),
        "--model-path",
        str(model_directory(data_root, golden["model"])),
    ] + request_flags(data_root, golden["request"])
    long_audio = golden.get("long_audio")
    clip_groups: dict[tuple[str | None, bool], list[Path]] = {}
    for clip_id, clip in manifest.items():
        is_long = (
            long_audio is not None and clip["duration"] > long_audio["over_seconds"]
        )
        clip_groups.setdefault(
            (golden["language_by_clip_language"][clip["lang"]], is_long), []
        ).append(data_root / "corpus" / "v1" / "clips" / f"{clip_id}.wav")
    results: dict[str, dict] = {}
    for (language, is_long), clips in clip_groups.items():
        long_flags = request_flags(data_root, long_audio["request"]) if is_long else []
        results.update(transcribe(command + long_flags, clips, language))
    return {clip_id: results[clip_id] for clip_id in manifest}


if __name__ == "__main__":
    run(transcribe_with_tool)
