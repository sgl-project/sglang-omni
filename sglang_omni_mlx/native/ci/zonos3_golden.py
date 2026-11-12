# SPDX-License-Identifier: Apache-2.0
"""ZONOS3-Mini golden check for check_golden.py: seeded speech requests through
zonos3_synthesize, whose prompt text ids, aligned audio codes and decoded PCM a chip
with golden outputs must give exactly. Another chip's GPU arithmetic may sample other
codes, so there only the prompt ids must match: the tokenizer and the text normalizer
set them.

A golden file of kind zonos3 names its cases (text, seed, max_new_tokens, and optional
voice, sampling fields, speed, stream, and a reference: a corpus clip to embed as the
speaker). Outputs are compared as speech_golden.py compares them, the PCM's SHA-256
included.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import wave
from pathlib import Path

from speech_golden import SpeechCase, SpeechOutputs, joined

SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "speed")


def synthesize(
    runtime_bin: Path, model_directory: Path, case: SpeechCase
) -> SpeechOutputs:
    with tempfile.TemporaryDirectory(prefix="zonos3-golden-") as directory:
        codes_path = Path(directory) / "codes.json"
        audio_path = Path(directory) / "speech.wav"
        command = [
            str(runtime_bin / "zonos3_synthesize"),
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
        if "voice" in case:
            command += ["--voice", case["voice"]]
        else:
            pass
        for field in SAMPLING_FIELDS:
            if field in case:
                command += [f"--{field.replace('_', '-')}", str(case[field])]
            else:
                pass
        if case.get("stream", False):
            command.append("--stream")
        else:
            pass
        reference = case.get("reference")
        if reference is not None:
            # model_directory is <data root>/models/<repo>, beside the corpus.
            clips = model_directory.parents[1] / "corpus" / "v1" / "clips"
            command += ["--reference-audio", str(clips / f"{reference['clip']}.wav")]
        else:
            pass
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        codes = json.loads(codes_path.read_text())
        with wave.open(str(audio_path)) as reader:
            pcm = reader.readframes(reader.getnframes())
    return {
        "prompt_ids": joined(codes["prompt_ids"]),
        "finish_reason": json.loads(completed.stdout.splitlines()[-1])["finish_reason"],
        "codes": [joined(frame) for frame in codes["frames"]],
        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
    }
