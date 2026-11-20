# SPDX-License-Identifier: Apache-2.0
"""Prepares the voice Drama 3's goldens and API tests clone: the Common Voice clip
(CC0) the CUDA cookbook clones, resampled to the Fish S1 DAC codec's 44.1 kHz.

    python fishaudio_drama3_voice.py <data_root>
"""

from __future__ import annotations

import argparse
from math import gcd
from pathlib import Path

import soundfile
from fishaudio_drama3_golden import VOICE_DIRECTORY, VOICE_NAME
from provision import download, sha256
from scipy.signal import resample_poly

SOURCE_URL = "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/0df0b57b45253d5a9fa0b0e98846642f235f828e/en/prompt-wavs/common_voice_en_10119832.wav"
SOURCE_SHA256 = "af3ac928e98cdb171c9abdc611813f2cc40c52b71167790b785628104f81f389"
VOICE_SHA256 = "df436ec78ec5b99cb16d4ca91057148bb490ac58715da6e49ce22ffa070f643b"
VOICE_SAMPLE_RATE = 44100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    arguments = parser.parse_args()
    voice = arguments.data_root / VOICE_DIRECTORY / VOICE_NAME
    if not voice.exists() or sha256(voice) != VOICE_SHA256:
        source = arguments.data_root / VOICE_DIRECTORY / "common_voice_en_10119832.wav"
        download(SOURCE_URL, source, SOURCE_SHA256)
        samples, source_rate = soundfile.read(source, dtype="float64")
        step = gcd(VOICE_SAMPLE_RATE, source_rate)
        soundfile.write(
            voice,
            resample_poly(samples, VOICE_SAMPLE_RATE // step, source_rate // step),
            VOICE_SAMPLE_RATE,
            subtype="PCM_16",
        )
        if sha256(voice) != VOICE_SHA256:
            raise SystemExit(f"{voice} does not match the pinned {VOICE_SHA256}")
        else:
            pass
    else:
        pass
    print(f"voice: {voice}", flush=True)


if __name__ == "__main__":
    main()
