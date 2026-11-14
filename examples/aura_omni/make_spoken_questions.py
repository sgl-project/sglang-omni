# SPDX-License-Identifier: Apache-2.0
"""Synthesize the spoken questions the AURA Omni tests ask, through a running Qwen3-TTS server.

    sgl-omni serve --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --port 8010
    python examples/aura_omni/make_spoken_questions.py --api-base http://127.0.0.1:8010 --out-dir tests/data/aura_omni

Each question is written as 16 kHz mono PCM16 WAV, the rate Qwen3-ASR reads, and
questions.json records the text each file speaks.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import librosa
import numpy as np
import requests
import soundfile

ASR_SAMPLE_RATE = 16000
SPEAKER = "ryan"
QUESTIONS = {
    "capital_of_france.wav": "What is the capital of France? Answer in one word.",
    "ball_color.wav": "What color is the ball in the video?",
    "secret_word_and_shape.wav": "The secret word is banana. What is the secret word, and what color is the shape in the video?",
    "first_color.wav": "Which color appeared first in the video, red or blue?",
    "red_ball_claim.wav": "I just put a red ball on the table. What color is the ball in the video?",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for file_name, text in QUESTIONS.items():
        response = requests.post(
            f"{args.api_base}/v1/audio/speech",
            json={
                "input": text,
                "voice": SPEAKER,
                "language": "English",
                "response_format": "wav",
                "temperature": 0.0,
            },
            timeout=300,
        )
        response.raise_for_status()
        waveform, sample_rate = soundfile.read(
            io.BytesIO(response.content), dtype="float32"
        )
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        else:
            pass
        waveform = librosa.resample(
            waveform, orig_sr=sample_rate, target_sr=ASR_SAMPLE_RATE
        )
        peak = float(np.max(np.abs(waveform)))
        soundfile.write(
            args.out_dir / file_name,
            waveform / peak * 0.9 if peak > 0 else waveform,
            ASR_SAMPLE_RATE,
            subtype="PCM_16",
        )
        print(f"{file_name}: {len(waveform) / ASR_SAMPLE_RATE:.2f} s  {text}")
    (args.out_dir / "questions.json").write_text(json.dumps(QUESTIONS, indent=2) + "\n")


if __name__ == "__main__":
    main()
