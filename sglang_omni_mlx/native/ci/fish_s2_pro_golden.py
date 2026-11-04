# SPDX-License-Identifier: Apache-2.0
"""Fish S2-Pro golden check for check_golden.py: seeded speech requests through
fish_s2_pro_synthesize, whose audio codes and decoded PCM a chip with golden outputs
must give exactly. Another chip's GPU arithmetic may sample other codes, so there only
the prompt ids must match, and those are the text positions alone: the reference's
semantic codes come from the codec.

A golden file of kind fish_s2_pro names its cases (text, seed, max_new_tokens, and an
optional reference: its transcript with a corpus clip to clone). Outputs are compared
as speech_golden.py compares them, the PCM's SHA-256 included.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import wave
from pathlib import Path

from speech_golden import SpeechCase, SpeechOutputs, joined


def synthesize(
    runtime_bin: Path, model_directory: Path, case: SpeechCase
) -> SpeechOutputs:
    with tempfile.TemporaryDirectory(prefix="fish-golden-") as directory:
        codes_path = Path(directory) / "codes.json"
        audio_path = Path(directory) / "speech.wav"
        command = [
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
        ]  # fmt: skip
        reference = case.get("reference")
        if reference is not None:
            # model_directory is <data root>/models/<repo>, beside the corpus.
            clips = model_directory.parents[1] / "corpus" / "v1" / "clips"
            command += [
                "--ref-audio",
                str(clips / f"{reference['clip']}.wav"),
                "--ref-text",
                reference["text"],
            ]
        else:
            pass
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
