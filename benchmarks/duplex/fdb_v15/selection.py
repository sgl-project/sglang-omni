# SPDX-License-Identifier: Apache-2.0
"""Which dataset pairs (v1.5) and samples (v1.0) one run evaluates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from benchmarks.duplex.fdb_v15.common import V10_DIR
from benchmarks.duplex.v10_dataset import SUBSETS as V10_SUBSETS
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


def parse_subset_count(
    text: str, subsets: tuple[str, ...] = SUBSETS
) -> tuple[str, int | None]:
    """CATEGORY=COUNT, for example user_interruption=20 or talking_to_other=all."""
    subset, separator, count = text.partition("=")
    if not separator or subset not in subsets:
        raise argparse.ArgumentTypeError(
            f"expected CATEGORY=COUNT with CATEGORY in {', '.join(subsets)}, got '{text}'"
        )
    else:
        return subset, parse_count(count)


def first_per_subset(
    dataset: Path, selection: SampleSelection, subsets: tuple[str, ...]
) -> list[str]:
    sample_ids = []
    for subset in subsets:
        count = selection.subset_counts.get(subset, selection.per_subset)
        if count == 0:
            continue
        else:
            sample_ids += select_sample_ids(dataset, (subset,), None, count)
    return sample_ids


def select_samples(dataset: Path, selection: SampleSelection) -> list[str]:
    """Sample IDs in dataset order; each category takes its first N samples."""
    if selection.sample_ids is not None:
        return select_sample_ids(dataset, SUBSETS, selection.sample_ids, None)
    else:
        pass
    sample_ids = first_per_subset(dataset, selection, SUBSETS)
    if not sample_ids:
        raise SystemExit("ERROR: the selection is empty.")
    else:
        return sample_ids


def require_v10_subsets(dataset: Path, subsets: tuple[str, ...]) -> None:
    """Every v1.0 subset a selection reads must be staged; a partial setup
    would otherwise narrow the run without a word."""
    missing = [subset for subset in subsets if not (dataset / subset).is_dir()]
    if missing:
        raise SystemExit(
            f"ERROR: {dataset} lacks the v1.0 subset(s) {', '.join(missing)}. "
            "Run `python -m benchmarks.duplex.fdb_v15 setup` (it adds the v1.0 "
            "dataset to an existing workspace), or pass --v10-per-subset 0."
        )
    else:
        pass


def select_v10_samples(dataset: Path, selection: SampleSelection) -> list[str]:
    """v1.0 sample IDs: explicit IDs, or the first N per subset. Empty when every
    count is 0, which leaves v1.0 out of the run without touching its dataset."""
    if selection.sample_ids is not None:
        named = tuple(
            subset
            for subset in V10_SUBSETS
            if any(
                sample_id.startswith(f"{subset}/") for sample_id in selection.sample_ids
            )
        )
        require_v10_subsets(dataset, named)
        return select_sample_ids(dataset, V10_SUBSETS, selection.sample_ids, None)
    else:
        pass
    wanted = tuple(
        subset
        for subset in V10_SUBSETS
        if selection.subset_counts.get(subset, selection.per_subset) != 0
    )
    require_v10_subsets(dataset, wanted)
    return first_per_subset(dataset, selection, V10_SUBSETS)


def describe(sample_ids: list[str], subsets: tuple[str, ...] = SUBSETS) -> str:
    counts = {subset: 0 for subset in subsets}
    for sample_id in sample_ids:
        counts[sample_id.split("/", 1)[0]] += 1
    return ", ".join(f"{subset} {count}" for subset, count in counts.items() if count)


def ids_text(sample_ids: list[str]) -> str:
    return "".join(f"{sample_id}\n" for sample_id in sample_ids)


def check_matches_other_repeats(
    repeat_dir: Path, sample_ids: list[str], v10_sample_ids: list[str]
) -> None:
    """Repeats of one run must evaluate the same pairs, or their mean is meaningless."""
    expected = ids_text(sample_ids)
    v10_expected = ids_text(v10_sample_ids)
    for ids_file in sorted(repeat_dir.parent.glob("repeat-*/sample-ids.txt")):
        if ids_file.parent == repeat_dir:
            continue
        elif ids_file.read_text() != expected:
            raise SystemExit(
                f"ERROR: {ids_file} selects different pairs. Use the same selection "
                "for every repeat of a run, or a new --run-name."
            )
        else:
            pass
        v10_ids_file = ids_file.parent / V10_DIR / "sample-ids.txt"
        if not v10_ids_file.is_file() and v10_expected:
            raise SystemExit(
                f"ERROR: {ids_file.parent} is a v1.5-only repeat (no "
                f"{V10_DIR}/sample-ids.txt). Pass --v10-per-subset 0 to continue "
                "this run, or use a new --run-name."
            )
        elif v10_ids_file.is_file() and v10_ids_file.read_text() != v10_expected:
            raise SystemExit(
                f"ERROR: {v10_ids_file} selects different v1.0 samples. Use the "
                "same v1.0 selection for every repeat of a run, or a new --run-name."
            )
        else:
            pass
