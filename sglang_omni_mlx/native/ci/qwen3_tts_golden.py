# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS golden check for check_golden.py: seeded speech requests through
qwen3_tts_synthesize, held to the rule of Voxtral TTS (voxtral_golden.py): a chip with
golden outputs must give every prompt id, audio code and 16-bit PCM sample exactly,
another chip only the prompt ids. The prompt ids are every id the prompt embeds: the
text's tokens and the codec's thinking, language and speaker ids.

A golden file of kind qwen3_tts names its cases (text, speaker, language, seed and
max_new_tokens), synthesized with the CUDA server's prompt and sampling.
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


def synthesize(runtime_bin: Path, model_directory: Path, case: dict) -> dict:
    with tempfile.TemporaryDirectory(prefix="qwen3-tts-golden-") as directory:
        codes_path = Path(directory) / "codes.json"
        wav_path = Path(directory) / "speech.wav"
        completed = subprocess.run(
            [
                str(runtime_bin / "qwen3_tts_synthesize"),
                "--model-path",
                str(model_directory),
                "--text",
                case["text"],
                "--speaker",
                case["speaker"],
                "--language",
                case["language"],
                "--seed",
                str(case["seed"]),
                "--max-new-tokens",
                str(case["max_new_tokens"]),
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
        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
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


def check(
    golden: dict, device: str, results: dict[str, dict]
) -> tuple[list[str], list[str]]:
    """Returns (report lines, failures): Voxtral's check, plus PCM on a recorded chip."""
    lines, failures = voxtral_golden.check(golden, device, results)
    recorded = golden["devices"].get(device, {}).get("clips", {})
    failures += [
        f"`{case_id}` has other PCM"
        for case_id, outputs in recorded.items()
        if case_id in results
        and results[case_id].get("pcm_sha256") != outputs.get("pcm_sha256")
    ]
    return lines, failures
