# SPDX-License-Identifier: Apache-2.0
"""Discover and validate paired Full-Duplex-Bench v1.5 samples."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import soundfile
from pydantic import JsonValue

SUBSETS = (
    "user_interruption",
    "user_backchannel",
    "talking_to_other",
    "background_speech",
)
# Note (wenyao): README counts; the released backchannel archive holds 98.
DECLARED_COUNTS = {
    "user_interruption": 200,
    "user_backchannel": 99,
    "talking_to_other": 100,
    "background_speech": 100,
}
REQUIRED_FILES = {
    "input": "input.wav",
    "clean_input": "clean_input.wav",
    "metadata": "metadata.json",
}
OPTIONAL_FILES = {
    "input_transcript": "input.json",
    "clean_input_transcript": "clean_input.json",
    "context": "context.wav",
    "interrupt": "interrupt.wav",
    "backchannel": "backchannel.wav",
    "current_turn": "current_turn.wav",
    "background": "background.wav",
}
TRANSCRIPT_KEYS = ("input_transcript", "clean_input_transcript")
SAMPLE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(kw_only=True)
class Sample:
    id: str
    subset: str
    directory: str
    metadata: dict[str, JsonValue] | None = None
    paths: dict[str, str] = field(default_factory=dict)
    sha256: dict[str, str] = field(default_factory=dict)
    audio: dict[str, dict[str, int | float | str]] = field(default_factory=dict)
    transcripts: dict[str, JsonValue] = field(default_factory=dict)
    event_span_s: list[float] | None = None
    errors: list[str] = field(default_factory=list)


def natural_key(name: str) -> tuple[int, int, str]:
    return (0, int(name), name) if name.isdigit() else (1, 0, name)


def list_sample_dirs(
    root: Path, subsets: tuple[str, ...] = SUBSETS
) -> tuple[dict[str, list[str]], list[str]]:
    """Return sample directory names per subset and every ignored entry."""
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root is not a directory: {root}")
    names: dict[str, list[str]] = {}
    ignored = sorted(
        entry.name for entry in root.iterdir() if entry.name not in subsets
    )
    for subset in subsets:
        names[subset] = []
        subset_dir = root / subset
        if subset_dir.is_dir() and not subset_dir.is_symlink():
            for entry in subset_dir.iterdir():
                if (
                    SAMPLE_NAME.fullmatch(entry.name)
                    and entry.is_dir()
                    and not entry.is_symlink()
                ):
                    names[subset].append(entry.name)
                else:
                    ignored.append(f"{subset}/{entry.name}")
        names[subset].sort(key=natural_key)
    return names, sorted(ignored)


def inventory(root: Path) -> dict[str, dict[str, int] | list[str]]:
    """Declared versus observed subset sizes; shortfalls stay visible."""
    names, ignored = list_sample_dirs(root.resolve())
    return {
        "declared": dict(DECLARED_COUNTS),
        "observed": {subset: len(names[subset]) for subset in SUBSETS},
        "ignored": ignored,
    }


def validate_sample(root: Path, subset: str, name: str) -> Sample:
    directory = f"{subset}/{name}"
    sample = Sample(id=directory, subset=subset, directory=directory)
    sample_dir = root / directory
    for key, filename in {**REQUIRED_FILES, **OPTIONAL_FILES}.items():
        path = sample_dir / filename
        if not path.exists() and not path.is_symlink():
            if key in REQUIRED_FILES:
                sample.errors.append(f"missing {filename}")
        elif path.is_symlink() or not path.is_file():
            sample.errors.append(f"{filename} is not a regular file")
        else:
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                sample.errors.append(f"{filename} unreadable: {exc}")
                continue
            sample.paths[key] = f"{directory}/{filename}"
            sample.sha256[key] = digest

    for key, relative in sample.paths.items():
        if relative.endswith(".wav"):
            try:
                info = soundfile.info(str(root / relative))
            except (OSError, RuntimeError) as exc:
                sample.errors.append(f"{Path(relative).name} unreadable: {exc}")
                continue
            if info.frames <= 0:
                sample.errors.append(f"{Path(relative).name} has no audio frames")
            sample.audio[key] = {
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "frames": info.frames,
                "duration_s": info.frames / info.samplerate,
                "subtype": info.subtype,
            }
        elif relative.endswith(".json"):
            try:
                value = json.loads((root / relative).read_text(encoding="utf-8"))
            except OSError as exc:
                sample.errors.append(f"{Path(relative).name} unreadable: {exc}")
                continue
            except (UnicodeError, ValueError) as exc:
                sample.errors.append(f"{Path(relative).name} is not valid JSON: {exc}")
                continue
            if key in TRANSCRIPT_KEYS:
                sample.transcripts[key] = value
            elif isinstance(value, dict):
                sample.metadata = value
            else:
                sample.errors.append("metadata.json must hold a JSON object")

    if sample.metadata is not None:
        span = sample.metadata.get("timestamps")
        duration_s = sample.audio.get("input", {}).get("duration_s")
        if not (
            isinstance(span, list)
            and len(span) == 2
            and all(
                type(bound) in (int, float) and math.isfinite(bound) for bound in span
            )
        ):
            sample.errors.append("metadata timestamps must be finite [start, end]")
        elif not 0 <= span[0] < span[1]:
            sample.errors.append(f"metadata timestamps {span} are not ordered")
        elif duration_s is not None and span[1] > duration_s:
            sample.errors.append(
                f"metadata timestamps end {span[1]} exceeds input.wav duration "
                f"{duration_s}"
            )
        else:
            sample.event_span_s = [float(span[0]), float(span[1])]
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
    names, _ = list_sample_dirs(root)
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
        elif unknown:
            raise ValueError(f"unknown requested sample IDs: {unknown}")
        else:
            requested = set(sample_ids)
            selected = [i for i in available if i in requested]
    return [validate_sample(root, *sample_id.split("/", 1)) for sample_id in selected]
