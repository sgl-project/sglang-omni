# SPDX-License-Identifier: Apache-2.0
"""Checks fixed-seed Voxtral TTS acoustic token frames against a named reference."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TypedDict


class TTSCase(TypedDict):
    text: str
    voice: str
    seed: int
    max_new_tokens: int


class TTSOutput(TypedDict):
    frames: list[list[int]]
    finish_reason: str


class VoxtralGolden(TypedDict):
    model: str
    cases: dict[str, TTSCase]
    reference_outputs: dict[str, TTSOutput]
    reference: str


def generate(
    golden: VoxtralGolden, runtime_bin: Path, model_directory: Path
) -> dict[str, TTSOutput]:
    subprocess.run(
        [str(runtime_bin / "voxtral_tts_convert_voices"), str(model_directory)],
        check=True,
        capture_output=True,
        text=True,
    )
    generated_token_outputs = {}
    for case_name, tts_case in golden["cases"].items():
        subprocess_result = subprocess.run(
            [
                str(runtime_bin / "voxtral_tts_generate"),
                "--model-path",
                str(model_directory),
                "--text",
                tts_case["text"],
                "--voice",
                tts_case["voice"],
                "--seed",
                str(tts_case["seed"]),
                "--max-new-tokens",
                str(tts_case["max_new_tokens"]),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        generated_token_outputs[case_name] = json.loads(subprocess_result.stdout)
    return generated_token_outputs


def check(
    golden: VoxtralGolden, generated_token_outputs: dict[str, TTSOutput]
) -> tuple[list[str], list[str]]:
    failures = []
    reference_token_outputs = golden["reference_outputs"]
    if not reference_token_outputs:
        failures.append("the golden file has no reference token outputs")
    else:
        pass
    if set(generated_token_outputs) != set(reference_token_outputs):
        failures.append(
            f"generated cases {sorted(generated_token_outputs)} do not match "
            f"reference cases {sorted(reference_token_outputs)}"
        )
    else:
        pass
    identical = 0
    for case_name, reference_token_output in reference_token_outputs.items():
        if generated_token_outputs.get(case_name) == reference_token_output:
            identical += 1
        else:
            failures.append(f"`{case_name}` token frames or finish reason differ")
    lines = [
        f"### {golden['model']} Voxtral TTS",
        "",
        f"Reference: {golden['reference']}",
        "",
        f"- Acoustic token outputs identical: "
        f"{identical}/{len(reference_token_outputs)}",
    ]
    return lines, failures
