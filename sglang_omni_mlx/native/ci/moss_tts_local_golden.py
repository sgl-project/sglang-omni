# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS-Local golden check for check_golden.py: seeded speech requests through
moss_tts_local_synthesize, on the 8-bit load-time quantization the server defaults to
and the pinned MOSS-Audio-Tokenizer-v2 codec. A chip with golden outputs must give
every prompt id, audio code and 16-bit PCM sample exactly, which gates the codec
decoder too; another chip only the prompt ids.

A golden file of kind moss_tts_local names its cases (text, seed, max_new_tokens, and
optionally language, instructions and token_count); speech_golden.py compares the
outputs.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import wave
from pathlib import Path

from speech_golden import SpeechCase, SpeechOutputs, joined

CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
QUANTIZATION_BITS = 8
OPTIONAL_FIELDS = ("language", "instructions", "token_count")


def synthesize(
    runtime_bin: Path, model_directory: Path, case: SpeechCase
) -> SpeechOutputs:
    # The codec is provisioned beside the model, under the same data root.
    codec_directory = model_directory.parent / CODEC_REPO.replace("/", "_")
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
                str(QUANTIZATION_BITS),
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
        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
        "prompt_ids": joined(codes["prompt_ids"]),
        "finish_reason": json.loads(completed.stdout.splitlines()[-1])["finish_reason"],
        "codes": [joined(frame) for frame in codes["codes"]],
    }
