# SPDX-License-Identifier: Apache-2.0
"""Higgs Audio v3 TTS golden check for check_golden.py: seeded speech requests
through higgs_tts_synthesize, whose prompt ids and delayed rows a chip with golden
outputs must give exactly. Another chip's GPU arithmetic may sample other codes, so
there only the prompt ids, which the tokenizer alone sets, must match.

A golden file of kind higgs_tts names its cases (text, seed and max_new_tokens,
sampled with the server's defaults); each chip's outputs keep one string of codes
per delayed row, and speech_golden.py compares them.
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
    with tempfile.TemporaryDirectory(prefix="higgs-golden-") as directory:
        codes_path = Path(directory) / "codes.json"
        completed = subprocess.run(
            [
                str(runtime_bin / "higgs_tts_synthesize"),
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
            ],  # fmt: skip
            check=True,
            capture_output=True,
            text=True,
        )
        codes = json.loads(codes_path.read_text())
    return {
        "prompt_ids": joined(codes["prompt_ids"]),
        "finish_reason": json.loads(completed.stdout.splitlines()[-1])["finish_reason"],
        "codes": [joined(row) for row in codes["rows"]],
    }
