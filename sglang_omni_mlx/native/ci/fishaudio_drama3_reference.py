# SPDX-License-Identifier: Apache-2.0
"""Reference parity for Drama 3: every case of the golden file through mlx-audio's
fish_drama3 and through fishaudio_drama3_synthesize, with the same checkpoint and
seed; their frames' codes and their decoded PCM16 must be identical.

Runs in its own Python environment with the reference, on the MLX build the native
runtime uses:

    pip install mlx==0.32.3 mlx-metal==0.32.3 soundfile
    pip install --no-deps mlx-audio==0.6.2
    python fishaudio_drama3_reference.py --runtime-bin DIR --model-path DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from importlib.metadata import version
from pathlib import Path
from typing import TypedDict

import mlx.core as mx
import numpy as np
import soundfile
from fishaudio_drama3_golden import CI_DIRECTORY, SAMPLING_FIELDS, synthesize
from mlx_audio.tts.models.fish_drama3.drama3 import Model, _Voice
from mlx_audio.tts.utils import load_model
from speech_golden import SpeechCase, SpeechReference, joined

GOLDEN_PATH = CI_DIRECTORY / "model_golden" / "fish-audio-drama-3-4bit.json"
PCM16_SCALE = 32767.0
# A WAV's PCM16 samples to floats, as the native WAV decoder reads them.
PCM16_FULL_SCALE = 32768.0
# The reference and the MLX build the native runtime pins; mlx-audio 0.6.2 itself
# asks for MLX 0.32.4, whose arithmetic samples other codes.
REQUIRED_VERSIONS = {"mlx": "0.32.3", "mlx-audio": "0.6.2"}


class ReferenceOutputs(TypedDict):
    codes: list[str]
    pcm_sha256: str


def reference_voice(model: Model, reference: SpeechReference | None) -> _Voice:
    """The reference's voice, encoded from its WAV or built from its codes."""
    if reference is None:
        return _Voice([])
    elif "audio" in reference:
        samples, _ = soundfile.read(CI_DIRECTORY / reference["audio"], dtype="int16")
        audio = samples.astype(np.float32) / PCM16_FULL_SCALE
        return _Voice([(reference["text"], model._encode_audio(audio))])
    else:
        codes = [[int(code) for code in row.split()] for row in reference["codes"]]
        return _Voice([(reference["text"], mx.array(codes, dtype=mx.int32))])


def reference_outputs(model: Model, case: SpeechCase) -> ReferenceOutputs:
    """The reference's codes per frame and the SHA-256 of its PCM16 audio."""
    voice = reference_voice(model, case.get("reference"))
    model._voice = lambda ref_audio, ref_text: voice
    codes: list[str] = []
    generate_frames = Model._frames.__get__(model)

    def recorded_frames(
        frames: mx.array,
        system_rows: int,
        voice: _Voice,
        max_frames: int,
        temperature: float,
        top_k: int | None,
        top_p: float,
    ) -> Iterator[mx.array]:
        for frame in generate_frames(
            frames, system_rows, voice, max_frames, temperature, top_k, top_p
        ):
            codes.append(joined(np.array(frame).tolist()))
            yield frame

    model._frames = recorded_frames
    mx.random.seed(case["seed"])
    pieces = [
        np.array(result.audio)
        for result in model.generate(
            text=case["text"],
            max_tokens=case["max_new_tokens"],
            **{field: case[field] for field in SAMPLING_FIELDS if field in case},
        )
    ]
    audio = np.concatenate(pieces) if pieces else np.zeros(0, np.float32)
    pcm = (np.nan_to_num(np.clip(audio, -1.0, 1.0)) * PCM16_SCALE).astype("<i2")
    return {"codes": codes, "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-bin", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    arguments = parser.parse_args()
    for package, required in REQUIRED_VERSIONS.items():
        if version(package) != required:
            raise SystemExit(
                f"{package} {version(package)} is installed; need {required}"
            )
        else:
            pass
    cases = json.loads(GOLDEN_PATH.read_text())["cases"]
    native = {
        case_id: synthesize(arguments.runtime_bin, arguments.model_path, case)
        for case_id, case in cases.items()
    }
    model = load_model(str(arguments.model_path))
    report = {}
    for case_id, case in cases.items():
        reference = reference_outputs(model, case)
        same_frames = next(
            (
                frame
                for frame, (expected, actual) in enumerate(
                    zip(reference["codes"], native[case_id]["codes"])
                )
                if expected != actual
            ),
            min(len(reference["codes"]), len(native[case_id]["codes"])),
        )
        report[case_id] = {
            "frames": len(reference["codes"]),
            "native_frames": len(native[case_id]["codes"]),
            "identical_frames": same_frames,
            "identical_pcm": reference["pcm_sha256"] == native[case_id]["pcm_sha256"],
        }
        print(case_id, json.dumps(report[case_id]), flush=True)
    identical = sum(
        row["identical_frames"] == row["frames"] == row["native_frames"]
        and row["identical_pcm"]
        for row in report.values()
    )
    print(f"{identical}/{len(report)} cases identical in codes and PCM16")
    if identical != len(report):
        raise SystemExit(1)
    else:
        pass


if __name__ == "__main__":
    main()
