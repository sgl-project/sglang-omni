# SPDX-License-Identifier: Apache-2.0
"""Which dataset pairs one run evaluates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from benchmarks.duplex.v15_dataset import SUBSETS, select_sample_ids

ALL_SAMPLES = "all"


@dataclass(frozen=True)
class SampleSelection:
    """per_subset applies to every category; subset_counts overrides single
    categories (0 skips one). None means every pair. sample_ids, when set,
    replaces both."""

    per_subset: int | None
    subset_counts: dict[str, int | None] = field(default_factory=dict)
    sample_ids: list[str] | None = None


def parse_count(text: str) -> int | None:
    """A non-negative pair count, or "all"."""
    if text == ALL_SAMPLES:
        return None
    elif text.isdigit():
        return int(text)
    else:
        raise argparse.ArgumentTypeError(
            f"expected a count or '{ALL_SAMPLES}', got '{text}'"
        )


def parse_subset_count(text: str) -> tuple[str, int | None]:
    """CATEGORY=COUNT, for example user_interruption=20 or talking_to_other=all."""
    subset, separator, count = text.partition("=")
    if not separator or subset not in SUBSETS:
        raise argparse.ArgumentTypeError(
            f"expected CATEGORY=COUNT with CATEGORY in {', '.join(SUBSETS)}, got '{text}'"
        )
    else:
        return subset, parse_count(count)


def select_samples(dataset: Path, selection: SampleSelection) -> list[str]:
    """Sample IDs in dataset order; each category takes its first N samples."""
    if selection.sample_ids is not None:
        return select_sample_ids(dataset, SUBSETS, selection.sample_ids, None)
    else:
        pass
    sample_ids = []
    for subset in SUBSETS:
        count = selection.subset_counts.get(subset, selection.per_subset)
        if count == 0:
            continue
        else:
            sample_ids += select_sample_ids(dataset, (subset,), None, count)
    if not sample_ids:
        raise SystemExit("ERROR: the selection is empty.")
    else:
        return sample_ids


def describe(sample_ids: list[str]) -> str:
    counts = {subset: 0 for subset in SUBSETS}
    for sample_id in sample_ids:
        counts[sample_id.split("/", 1)[0]] += 1
    return ", ".join(f"{subset} {count}" for subset, count in counts.items() if count)


def check_matches_other_repeats(repeat_dir: Path, sample_ids: list[str]) -> None:
    """Repeats of one run must evaluate the same pairs, or their mean is meaningless."""
    expected = "\n".join(sample_ids) + "\n"
    for ids_file in sorted(repeat_dir.parent.glob("repeat-*/sample-ids.txt")):
        if ids_file.parent != repeat_dir and ids_file.read_text() != expected:
            raise SystemExit(
                f"ERROR: {ids_file} selects different pairs. Use the same selection "
                "for every repeat of a run, or a new --run-name."
            )
        else:
            pass
