# SPDX-License-Identifier: Apache-2.0
"""SeedTTS dataset loader for seed-tts-eval meta.lst files."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class SampleInput:
    sample_id: str
    ref_text: str
    ref_audio: str
    target_text: str


def load_seedtts_samples(
    path: str, max_samples: int | None = None
) -> list[SampleInput]:
    """Parse a seed-tts-eval meta.lst file.

    Format per line: ``id|ref_text|ref_audio_path|target_text``

    Note (chenyang): Detailed format description could be found at
    https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval
    """
    base_dir = os.path.dirname(path)
    samples: list[SampleInput] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            samples.append(
                SampleInput(
                    sample_id=parts[0],
                    ref_text=parts[1],
                    ref_audio=os.path.join(base_dir, parts[2]),
                    target_text=parts[3],
                )
            )
            if max_samples and len(samples) >= max_samples:
                break
    return samples
