# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS-Local golden check for check_golden.py: seeded speech requests through
moss_tts_local_synthesize, held to the Voxtral TTS golden's rule and one more. A chip
with golden outputs must give every prompt id and audio code exactly, and the same
16-bit PCM (by SHA-256), which gates the codec decoder; another chip, only the
prompt ids.

A golden file of kind moss_tts_local names the codec checkpoint and the load-time
quantization, and its cases (text, seed, max_new_tokens, and optionally language,
instructions and token_count).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import wave
from pathlib import Path

import voxtral_golden
from voxtral_golden import joined

OPTIONAL_FIELDS = ("language", "instructions", "token_count")


def synthesize(
    runtime_bin: Path,
    model_directory: Path,
    codec_directory: Path,
    golden: dict,
    case: dict,
) -> dict:
    optional_flags = []
    for field in OPTIONAL_FIELDS:
        if field in case:
            optional_flags += [f"--{field.replace('_', '-')}", str(case[field])]
        else:
            pass
    with tempfile.TemporaryDirectory(prefix="moss-tts-local-golden-") as directory:
        codes_path = Path(directory) / "codes.json"
        wav_path = Path(directory) / "speech.wav"
        completed = subprocess.run(
            [
                str(runtime_bin / "moss_tts_local_synthesize"),
                "--model-path",
                str(model_directory),
                "--codec-path",
                str(codec_directory),
                "--quantization-bits",
                str(golden["quantization_bits"]),
                "--text",
                case["text"],
                "--seed",
                str(case["seed"]),
                "--max-new-tokens",
                str(case["max_new_tokens"]),
                *optional_flags,
                "--codes-output",
                str(codes_path),
                "--output",
                str(wav_path),
            ],  # fmt: skip
            check=True,
            capture_output=True,
            text=True,
        )
        codes = json.loads(codes_path.read_text())
        with wave.open(str(wav_path)) as reader:
            pcm = reader.readframes(reader.getnframes())
    return {
        "prompt_ids": joined(codes["prompt_ids"]),
        "finish_reason": json.loads(completed.stdout.splitlines()[-1])["finish_reason"],
        "codes": [joined(frame) for frame in codes["codes"]],
        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
    }


def synthesize_cases(
    runtime_bin: Path, model_directory: Path, codec_directory: Path, golden: dict
) -> dict[str, dict]:
    return {
        case_id: synthesize(runtime_bin, model_directory, codec_directory, golden, case)
        for case_id, case in golden["cases"].items()
    }


def check(
    golden: dict, device: str, results: dict[str, dict]
) -> tuple[list[str], list[str]]:
    """Returns (report lines, failures)."""
    lines, failures = voxtral_golden.check(golden, device, results)
    if device in golden["devices"]:
        for case_id, outputs in golden["devices"][device]["clips"].items():
            actual = results.get(case_id)
            if actual is not None and actual["pcm_sha256"] != outputs["pcm_sha256"]:
                failures.append(f"`{case_id}` has other PCM")
            else:
                pass
    else:
        pass
    return lines, failures
