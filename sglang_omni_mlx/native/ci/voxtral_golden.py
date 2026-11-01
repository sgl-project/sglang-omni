# SPDX-License-Identifier: Apache-2.0
"""Voxtral TTS golden check for check_golden.py: seeded speech requests through
voxtral_tts_synthesize, whose prompt ids and audio codes a chip with golden outputs
must give exactly. Another chip's GPU arithmetic may sample other codes, so there
only the prompt ids, which the tokenizer alone sets, must match.

A golden file of kind voxtral_tts names its cases (text, voice, seed and
max_new_tokens); speech_golden.py compares the outputs.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from speech_golden import SpeechCase, SpeechOutputs, joined


def synthesize(
    runtime_bin: Path, model_directory: Path, case: SpeechCase
) -> SpeechOutputs:
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
