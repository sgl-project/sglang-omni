# SPDX-License-Identifier: Apache-2.0
"""Haru golden: a seeded greedy full-duplex session through haru_session_probe. A chip with golden outputs must give every case's tokens and, when the golden holds one, its audio hash exactly; another chip, whose GPU arithmetic may sample other tokens, only as many frames."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import NotRequired, TypedDict


class HaruCase(TypedDict):
    seed: int


class HaruOutputs(TypedDict):
    tokens: list[int]
    audio_sha256: NotRequired[str]


class HaruDevice(TypedDict):
    metrics: dict[str, float]
    clips: dict[str, HaruOutputs]


class HaruGolden(TypedDict):
    model: str
    kind: str
    tool: str
    cases: dict[str, HaruCase]
    devices: dict[str, HaruDevice]


def synthesize(runtime_bin: Path, model_directory: Path, case: HaruCase) -> HaruOutputs:
    """Run one seeded session and return its tokens and audio hash."""
    completed = subprocess.run(
        [
            str(runtime_bin / "haru_session_probe"),
            str(model_directory),
            str(case["seed"]),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = completed.stdout.splitlines()
    return {
        "tokens": [int(line.split()[3]) for line in lines if line.startswith("frame ")],
        "audio_sha256": next(
            line.split()[1] for line in lines if line.startswith("audio_sha256 ")
        ),
    }


def difference(
    expected: HaruOutputs, actual: HaruOutputs | None, is_exact: bool
) -> str | None:
    """How actual departs from expected, or None when it does not."""
    if actual is None:
        return "was not run"
    elif len(actual["tokens"]) != len(expected["tokens"]):
        return f"has {len(actual['tokens'])} frames, not {len(expected['tokens'])}"
    elif not is_exact:
        return None
    elif actual["tokens"] != expected["tokens"]:
        return "has other tokens"
    elif "audio_sha256" in expected and (
        actual.get("audio_sha256") != expected["audio_sha256"]
    ):
        return "decodes to other audio"
    else:
        return None


def check(
    golden: HaruGolden, device: str, results: dict[str, HaruOutputs]
) -> tuple[list[str], list[str]]:
    """Returns (report lines, failures)."""
    devices = golden["devices"]
    if not devices:
        return [f"### {golden['model']} on {device}"], [
            "the golden file has no outputs"
        ]
    else:
        reference_device = device if device in devices else next(iter(devices))
    reference = devices[reference_device]["clips"]
    is_exact = device in devices
    differences = {
        case_id: difference(expected, results.get(case_id), is_exact)
        for case_id, expected in reference.items()
    }
    failures = [
        f"`{case_id}` has no golden output"
        for case_id in golden["cases"]
        if case_id not in reference
    ] + [
        f"`{case_id}` {found}"
        for case_id, found in differences.items()
        if found is not None
    ]
    mode = "exact" if is_exact else "no golden outputs for this chip; frame counts only"
    lines = [
        f"### {golden['model']} on {device}",
        "",
        f"Golden cases matching {reference_device}: "
        f"{sum(found is None for found in differences.values())}/{len(reference)} ({mode})",
    ]
    return lines, failures
