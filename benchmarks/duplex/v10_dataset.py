# SPDX-License-Identifier: Apache-2.0
"""Discover and validate Full-Duplex-Bench v1.0 subsets, grouped by task."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import soundfile
from pydantic import JsonValue

from benchmarks.duplex.v15_dataset import list_sample_dirs

Task = Literal["pause_handling", "turn_taking", "user_interruption", "backchannel"]

SUBSET_TASKS: dict[str, Task] = {
    "synthetic_pause_handling": "pause_handling",
    "candor_pause_handling": "pause_handling",
    "candor_turn_taking": "turn_taking",
    "synthetic_user_interruption": "user_interruption",
    "icc_backchannel": "backchannel",
}
SUBSETS = tuple(SUBSET_TASKS)
DECLARED_COUNTS = {
    "synthetic_pause_handling": 137,
    "candor_pause_handling": 216,
    "candor_turn_taking": 119,
    "synthetic_user_interruption": 200,
    "icc_backchannel": 55,
}
ANNOTATION_FILES = {
    "pause_handling": "pause.json",
    "turn_taking": "turn_taking.json",
    "user_interruption": "interrupt.json",
}


@dataclass(kw_only=True)
class Sample:
    id: str
    subset: str
    task: Task
    directory: str
    paths: dict[str, str] = field(default_factory=dict)
    sha256: dict[str, str] = field(default_factory=dict)
    audio: dict[str, int | float | str] = field(default_factory=dict)
    # Note (Jeffro): Pause samples may list several pauses; the other tasks have one event.
    # A turn-taking event may be zero-length; its start is the user turn end.
    events: list[list[float]] = field(default_factory=list)
    texts: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def inventory(root: Path) -> dict[str, dict[str, int] | list[str]]:
    names, ignored = list_sample_dirs(root.resolve(), SUBSETS)
    return {
        "declared": dict(DECLARED_COUNTS),
        "observed": {subset: len(names[subset]) for subset in SUBSETS},
        "ignored": ignored,
    }


def finite_span(value: JsonValue) -> list[float] | None:
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(
            isinstance(bound, (int, float)) and math.isfinite(bound) for bound in value
        )
        and 0 <= value[0] <= value[1]
    ):
        return [float(value[0]), float(value[1])]
    return None


def validate_sample(root: Path, subset: str, name: str) -> Sample:
    directory = f"{subset}/{name}"
    task = SUBSET_TASKS[subset]
    sample = Sample(id=directory, subset=subset, task=task, directory=directory)
    sample_dir = root / directory
    files = {"input": "input.wav"}
    if task in ANNOTATION_FILES:
        files["annotation"] = ANNOTATION_FILES[task]
    for key, filename in files.items():
        path = sample_dir / filename
        if path.is_symlink() or not path.is_file():
            sample.errors.append(f"missing {filename}")
            continue
        sample.paths[key] = f"{directory}/{filename}"
        sample.sha256[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    if "input" in sample.paths:
        try:
            info = soundfile.info(str(sample_dir / "input.wav"))
        except (OSError, RuntimeError) as exc:
            sample.errors.append(f"input.wav unreadable: {exc}")
        else:
            if info.frames <= 0:
                sample.errors.append("input.wav has no audio frames")
            sample.audio = {
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "frames": info.frames,
                "duration_s": info.frames / info.samplerate,
                "subtype": info.subtype,
            }
    if "annotation" not in sample.paths:
        return sample
    annotation_name = files["annotation"]
    try:
        entries = json.loads((root / sample.paths["annotation"]).read_text("utf-8"))
    except (UnicodeError, ValueError) as exc:
        sample.errors.append(f"{annotation_name} is not valid JSON: {exc}")
        return sample
    if not isinstance(entries, list) or not entries:
        sample.errors.append(f"{annotation_name} must be a nonempty list")
        return sample
    duration_s = sample.audio.get("duration_s")
    for index, entry in enumerate(entries):
        span = finite_span(entry.get("timestamp")) if isinstance(entry, dict) else None
        if span is None:
            sample.errors.append(f"{annotation_name}[{index}] lacks a finite timestamp")
        elif duration_s is not None and span[1] > duration_s:
            sample.errors.append(
                f"{annotation_name}[{index}] ends at {span[1]} past input.wav {duration_s}"
            )
        else:
            sample.events.append(span)
    if task != "pause_handling" and len(entries) != 1:
        sample.errors.append(f"{annotation_name} must hold exactly one event")
    if task == "user_interruption":
        entry = entries[0]
        for key in ("context", "interrupt"):
            text = entry.get(key) if isinstance(entry, dict) else None
            if not isinstance(text, str) or not text.strip():
                sample.errors.append(f"{annotation_name} lacks {key} text")
            else:
                sample.texts[key] = text
    return sample


def discover_samples(
    root: Path,
    sample_ids: list[str] | None = None,
    max_per_subset: int | None = None,
) -> list[Sample]:
    """Select samples deterministically; malformed selections carry their errors."""
    if sample_ids is not None and max_per_subset is not None:
        raise ValueError("sample_ids and max_per_subset are mutually exclusive")
    if max_per_subset is not None and max_per_subset <= 0:
        raise ValueError("max_per_subset must be positive")
    root = root.resolve()
    names, _ = list_sample_dirs(root, SUBSETS)
    available = [f"{subset}/{name}" for subset in SUBSETS for name in names[subset]]
    if sample_ids is None:
        selected = [
            f"{subset}/{name}"
            for subset in SUBSETS
            for name in names[subset][:max_per_subset]
        ]
    else:
        duplicates = sorted(i for i, n in Counter(sample_ids).items() if n > 1)
        unknown = sorted(set(sample_ids) - set(available))
        if duplicates:
            raise ValueError(f"duplicate requested sample IDs: {duplicates}")
        if unknown:
            raise ValueError(f"unknown requested sample IDs: {unknown}")
        requested = set(sample_ids)
        selected = [i for i in available if i in requested]
    return [validate_sample(root, *sample_id.split("/", 1)) for sample_id in selected]
