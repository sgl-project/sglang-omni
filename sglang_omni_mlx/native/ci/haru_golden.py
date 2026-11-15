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
    audio_sha256: str


def synthesize(
    runtime_bin: Path, model_directory: Path, case: dict
) -> HaruOutputs:
    """Run one seeded session and return its token stream and decoded audio hash."""
    seed = case.get("seed", 42)
    command = [
        str(runtime_bin / "haru_session_probe"),
        str(model_directory),
        str(seed),
    ]
    if case.get("quantize"):
        command.append(str(case["quantize"]))
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = completed.stdout.splitlines()
    return {
        "tokens": [
            int(line.split("token ")[1].split()[0])
            for line in lines
            if "token " in line
        ],
        "audio_sha256": next(
            line.split("audio_sha256 ")[1]
            for line in lines
            if line.startswith("audio_sha256 ")
        ),
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
    failures = []
    for case_id, want in reference.items():
        actual = results.get(case_id)
        if actual is None:
            failures.append(f"`{case_id}` was not run")
        elif actual.get("tokens") != want.get("tokens"):
            failures.append(f"`{case_id}` token stream differs")
        elif actual.get("audio_sha256") != want.get("audio_sha256"):
            failures.append(f"`{case_id}` decoded audio differs")
    failures += [
        f"`{case_id}` has no golden output"
        for case_id in golden["cases"]
        if case_id not in reference
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
