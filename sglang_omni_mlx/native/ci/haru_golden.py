# SPDX-License-Identifier: Apache-2.0
"""Haru golden: a seeded full-duplex session's token stream through
haru_session_probe.

Haru is full-duplex, so its golden is not a corpus transcript or a text-to-speech
request: it is the token sequence the model emits for one seeded session (a
synthetic 16 kHz voice reference, a character card, speak-first, then three
silent user frames). Each chip that has golden outputs must give that sequence
exactly; another chip is gated on the same comparison against the first
reference chip.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TypedDict


class HaruOutputs(TypedDict):
    tokens: list[int]


def synthesize(
    runtime_bin: Path, model_directory: Path, case: dict
) -> HaruOutputs:
    """Run one seeded session and return its token stream."""
    seed = case.get("seed", 42)
    completed = subprocess.run(
        [
            str(runtime_bin / "haru_session_probe"),
            str(model_directory),
            str(seed),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        "tokens": [
            int(line.split("token ")[1].split()[0])
            for line in completed.stdout.splitlines()
            if "token " in line
        ]
    }


def check(
    golden: dict, device: str, results: dict[str, HaruOutputs]
) -> tuple[list[str], list[str]]:
    """Returns (report lines, failures)."""
    devices = golden["devices"]
    if not devices:
        return [f"### {golden['model']} on {device}"], [
            "the golden file has no outputs"
        ]
    reference_device = device if device in devices else next(iter(devices))
    reference = devices[reference_device]["clips"]
    differences = {
        case_id: outputs["tokens"] != results[case_id]["tokens"]
        for case_id, outputs in reference.items()
        if case_id in results
    }
    missing_results = [
        f"`{case_id}` was not run" for case_id in reference if case_id not in results
    ]
    missing_golden = [
        f"`{case_id}` has no golden output"
        for case_id in golden["cases"]
        if case_id not in reference
    ]
    failures = missing_results + missing_golden + [
        f"`{case_id}` token stream differs" for case_id, differs in differences.items() if differs
    ]
    identical = len(reference) - len(failures)
    mode = (
        "exact"
        if device in devices
        else f"no golden outputs for this chip; compared against {reference_device}"
    )
    lines = [
        f"### {golden['model']} on {device}",
        "",
        f"Golden cases identical to {reference_device}: "
        f"{identical}/{len(reference)} ({mode})",
    ]
    return lines, failures
