# SPDX-License-Identifier: Apache-2.0
"""Drama 3 golden check for check_golden.py: seeded speech requests through
fishaudio_drama3_synthesize, whose first-chunk prompt ids, audio codes and decoded
PCM a chip with golden outputs must give exactly. Another chip's GPU arithmetic may
sample other codes, so there only the prompt ids must match: the tokenizer and the
reference's length set them, since the tool writes a reference's audio rows as -1.

A golden file of kind fishaudio_drama3 names its cases (text, seed, max_new_tokens,
optional sampling fields, and an optional reference: its transcript with either a
WAV in this directory or one string of codes per codebook). Outputs are compared as
speech_golden.py compares them, the PCM's SHA-256 included.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import wave
from pathlib import Path

from speech_golden import SpeechCase, SpeechOutputs, joined

CI_DIRECTORY = Path(__file__).resolve().parent
SAMPLING_FIELDS = ("temperature", "top_p", "top_k")


def synthesize(
    runtime_bin: Path, model_directory: Path, case: SpeechCase
) -> SpeechOutputs:
    with tempfile.TemporaryDirectory(prefix="drama3-golden-") as directory:
        codes_path = Path(directory) / "codes.json"
        audio_path = Path(directory) / "speech.wav"
        command = [
            str(runtime_bin / "fishaudio_drama3_synthesize"),
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
        ]  # fmt: skip
        for field in SAMPLING_FIELDS:
            if field in case:
                command += [f"--{field.replace('_', '-')}", str(case[field])]
            else:
                pass
        reference = case.get("reference")
        if reference is None:
            pass
        elif "audio" in reference:
            command += [
                "--reference-audio",
                str(CI_DIRECTORY / reference["audio"]),
                "--reference-text",
                reference["text"],
            ]
        else:
            reference_path = Path(directory) / "reference.json"
            reference_path.write_text(
                json.dumps(
                    [
                        [int(code) for code in codebook.split()]
                        for codebook in reference["codes"]
                    ]
                )
            )
            command += [
                "--reference-codes",
                str(reference_path),
                "--reference-text",
                reference["text"],
            ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        codes = json.loads(codes_path.read_text())
        with wave.open(str(audio_path)) as reader:
            pcm = reader.readframes(reader.getnframes())
    return {
        "prompt_ids": joined(codes["prompt_ids"]),
        "finish_reason": json.loads(completed.stdout.splitlines()[-1])["finish_reason"],
        "codes": [joined(frame) for frame in codes["codes"]],
        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
    }
