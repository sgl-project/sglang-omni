# SPDX-License-Identifier: Apache-2.0
"""Seeded Qwen3-TTS codes and fixed-code waveform regression checks."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import NotRequired, TypedDict


class SpeechCase(TypedDict):
    input: str
    voice: str
    seed: int
    max_new_tokens: int
    language: NotRequired[str]
    temperature: NotRequired[float]


class SpeechOutput(TypedDict):
    prompt_ids: list[int]
    control_ids: list[int]
    instruction_ids: list[int]
    reference_text_ids: list[int]
    codes: list[list[int]]
    finish_reason: str
    sample_count: int
    sample_rate: int
    waveform_sha256: str


class ChipOutputs(TypedDict):
    metrics: dict[str, float]
    clips: dict[str, SpeechOutput]


class Golden(TypedDict):
    model: str
    kind: str
    cases: dict[str, SpeechCase]
    devices: dict[str, ChipOutputs]


def synthesize_cases(
    runtime_bin: Path, model_directory: Path, cases: dict[str, SpeechCase]
) -> dict[str, SpeechOutput]:
    results: dict[str, SpeechOutput] = {}
    with tempfile.TemporaryDirectory(prefix="qwen3-tts-golden-") as directory:
        root = Path(directory)
        request_path = root / "request.json"
        waveform_path = root / "speech.wav"
        codes_path = root / "codes.json"
        for identifier, request in cases.items():
            request_path.write_text(json.dumps(request, ensure_ascii=False))
            subprocess.run(
                [
                    str(runtime_bin / "qwen3_tts_synthesize"),
                    "--model-path",
                    str(model_directory),
                    "--request",
                    str(request_path),
                    "--output",
                    str(waveform_path),
                    "--codes-output",
                    str(codes_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            output: SpeechOutput = json.loads(codes_path.read_text())
            output["waveform_sha256"] = hashlib.sha256(
                waveform_path.read_bytes()
            ).hexdigest()
            results[identifier] = output
    return results


def validate_outputs(golden: Golden, results: dict[str, SpeechOutput]) -> list[str]:
    failures = []
    if set(results) != set(golden["cases"]):
        failures.append("Qwen3-TTS outputs must cover every named case exactly")
    else:
        pass
    for identifier, output in results.items():
        codes = output.get("codes", [])
        if (
            not codes
            or any(len(frame) != 16 for frame in codes)
            or any(
                type(code) is not int or not 0 <= code < 2048
                for frame in codes
                for code in frame
            )
            or output.get("sample_rate") != 24000
            or output.get("sample_count") != len(codes) * 1920
            or output.get("finish_reason") not in {"stop", "length"}
            or not output.get("prompt_ids")
            or not output.get("control_ids")
            or len(output.get("waveform_sha256", "")) != 64
        ):
            failures.append(f"{identifier}: incomplete or invalid Qwen3-TTS output")
        else:
            pass
    return failures


def check(
    golden: Golden, device: str, results: dict[str, SpeechOutput]
) -> tuple[list[str], list[str]]:
    failures = validate_outputs(golden, results)
    devices = golden["devices"]
    if not devices:
        return [], failures + ["Qwen3-TTS has no recorded reference outputs"]
    elif device not in devices:
        return [], failures + [f"Qwen3-TTS has no recorded outputs for {device}"]
    else:
        pass
    expected = devices[device]["clips"]
    failures.extend(validate_outputs(golden, expected))
    for identifier, output in results.items():
        if identifier not in expected or output != expected[identifier]:
            failures.append(
                f"{identifier}: Qwen3-TTS codes, prompt or waveform changed"
            )
        else:
            pass
    return [
        f"### {golden['model']} on {device}",
        f"Checked {len(results)} cases",
    ], failures
