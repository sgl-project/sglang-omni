# SPDX-License-Identifier: Apache-2.0
"""Voxtral TTS golden check for check_golden.py: seeded speech requests through
voxtral_tts_synthesize, whose prompt ids and audio codes a chip with golden outputs
must give exactly. Another chip's GPU arithmetic may sample other codes, so there
only the prompt ids, which the tokenizer alone sets, must match.

A golden file of kind voxtral_tts names its cases (text, voice, seed and
max_new_tokens); each chip's outputs keep one string of codes per audio frame, so a
divergence reads as the frame it starts at.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


def joined(values: list[int]) -> str:
    return " ".join(str(value) for value in values)


def synthesize(runtime_bin: Path, model_directory: Path, case: dict) -> dict:
    with tempfile.TemporaryDirectory(prefix="voxtral-golden-") as directory:
        codes_path = Path(directory) / "codes.json"
        completed = subprocess.run(
            [
                str(runtime_bin / "voxtral_tts_synthesize"),
                "--model-path",
                str(model_directory),
                "--text",
                case["text"],
                "--voice",
                case["voice"],
                "--seed",
                str(case["seed"]),
                "--max-new-tokens",
                str(case["max_new_tokens"]),
                "--codes-output",
                str(codes_path),
            ],  # fmt: skip
            check=True,
            capture_output=True,
            text=True,
        )
        codes = json.loads(codes_path.read_text())
    return {
        "prompt_ids": joined(codes["prompt_ids"]),
        "finish_reason": json.loads(completed.stdout.splitlines()[-1])["finish_reason"],
        "codes": [joined(frame) for frame in codes["codes"]],
    }


def synthesize_cases(
    runtime_bin: Path, model_directory: Path, cases: dict[str, dict]
) -> dict[str, dict]:
    return {
        case_id: synthesize(runtime_bin, model_directory, case)
        for case_id, case in cases.items()
    }


def difference(expected: dict, actual: dict | None) -> str | None:
    """Where actual first departs from expected, or None when they match."""
    if actual is None:
        return "was not synthesized"
    elif actual["prompt_ids"] != expected["prompt_ids"]:
        return "has other prompt ids"
    else:
        pass
    for frame, (want, got) in enumerate(zip(expected["codes"], actual["codes"])):
        if want != got:
            return f"differs from frame {frame}"
        else:
            pass
    if len(actual["codes"]) != len(expected["codes"]):
        return f"has frame count {len(actual['codes'])}, not {len(expected['codes'])}"
    elif actual["finish_reason"] != expected["finish_reason"]:
        return (
            f"finished with {actual['finish_reason']}, not {expected['finish_reason']}"
        )
    else:
        return None


def check(
    golden: dict, device: str, results: dict[str, dict]
) -> tuple[list[str], list[str]]:
    """Returns (report lines, failures)."""
    devices = golden["devices"]
    if not devices:
        return [f"### {golden['model']} on {device}"], [
            "the golden file has no outputs"
        ]
    elif device in devices:
        reference_device = device
    else:
        reference_device = next(iter(devices))
    reference = devices[reference_device]["clips"]
    differences = {
        case_id: difference(outputs, results.get(case_id))
        for case_id, outputs in reference.items()
    }
    identical = sum(found is None for found in differences.values())
    # Every chip checks only cases with a reference, so a case without one fails.
    missing = [
        f"`{case_id}` has no golden output"
        for case_id in golden["cases"]
        if case_id not in reference
    ]
    if not reference:
        failures = ["the golden file has no cases"]
    elif device in devices:
        failures = missing + [
            f"`{case_id}` {found}"
            for case_id, found in differences.items()
            if found is not None
        ]
    else:
        failures = missing + [
            f"`{case_id}` has other prompt ids than {reference_device}'s"
            for case_id, outputs in reference.items()
            if results.get(case_id, {}).get("prompt_ids") != outputs["prompt_ids"]
        ]
    mode = (
        "exact"
        if device in devices
        else "no golden outputs for this chip; prompt ids exact"
    )
    lines = [
        f"### {golden['model']} on {device}",
        "",
        f"Golden cases identical to {reference_device}: "
        f"{identical}/{len(reference)} ({mode})",
    ]
    return lines, failures
