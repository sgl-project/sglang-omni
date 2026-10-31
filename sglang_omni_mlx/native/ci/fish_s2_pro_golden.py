# SPDX-License-Identifier: Apache-2.0
"""Fish S2-Pro golden check for check_golden.py: seeded speech requests through
fish_s2_pro_synthesize, checked as voxtral_golden checks Voxtral's.

A case names its text, seed and max_new_tokens, and optionally a corpus clip to
clone (ref_clip, with ref_text). A chip with golden outputs must also give the
reference's codes and a hash of the decoded 16-bit PCM exactly. The recorded
prompt ids are the text positions alone: the reference's semantic codes come
from the codec, whose arithmetic, like the sampled codes, may differ on another
chip, so another chip is held to the text prompt ids only.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import voxtral_golden
from voxtral_golden import joined

EXACT_FIELDS = ("reference_codes", "audio_sha256")


def synthesize(
    runtime_bin: Path, data_root: Path, model_directory: Path, case: dict
) -> dict:
    with tempfile.TemporaryDirectory(prefix="fish-golden-") as directory:
        codes_path = Path(directory) / "codes.json"
        audio_path = Path(directory) / "speech.wav"
        reference = (
            [
                "--ref-audio",
                str(data_root / "corpus" / "v1" / "clips" / f"{case['ref_clip']}.wav"),
                "--ref-text",
                case["ref_text"],
            ]
            if "ref_clip" in case
            else []
        )
        completed = subprocess.run(
            [
                str(runtime_bin / "fish_s2_pro_synthesize"),
                "--model-path",
                str(model_directory),
                "--text",
                case["text"],
                "--seed",
                str(case["seed"]),
                "--max-new-tokens",
                str(case["max_new_tokens"]),
                "--codes-output",
                str(codes_path),
                "--output",
                str(audio_path),
                *reference,
            ],  # fmt: skip
            check=True,
            capture_output=True,
            text=True,
        )
        codes = json.loads(codes_path.read_text())
        audio_sha256 = hashlib.sha256(audio_path.read_bytes()).hexdigest()
    return {
        "prompt_ids": joined(codes["prompt_text_ids"]),
        "reference_codes": [
            [joined(row) for row in reference] for reference in codes["reference_codes"]
        ],
        "audio_sha256": audio_sha256,
        "finish_reason": json.loads(completed.stdout.splitlines()[-1])["finish_reason"],
        "codes": [joined(frame) for frame in codes["codes"]],
    }


def synthesize_cases(
    runtime_bin: Path, data_root: Path, model_directory: Path, cases: dict[str, dict]
) -> dict[str, dict]:
    return {
        case_id: synthesize(runtime_bin, data_root, model_directory, case)
        for case_id, case in cases.items()
    }


def check(
    golden: dict, device: str, results: dict[str, dict]
) -> tuple[list[str], list[str]]:
    """voxtral_golden.check, plus the reference codes and PCM on a recorded chip."""
    lines, failures = voxtral_golden.check(golden, device, results)
    recorded = golden["devices"].get(device, {}).get("clips", {})
    for case_id, outputs in recorded.items():
        actual = results.get(case_id)
        for field in EXACT_FIELDS:
            if actual is not None and actual.get(field) != outputs.get(field):
                failures.append(f"`{case_id}` has other {field}")
            else:
                pass
    return lines, failures
