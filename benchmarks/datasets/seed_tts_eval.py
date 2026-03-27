from __future__ import annotations

import os

from benchmarks.core.types import INPUT_PREVIEW_LENGTH, NormalizedSample

from .base import DatasetLoader

META_FIELD_COUNT = 4


class SeedTTSEvalLoader(DatasetLoader):
    name = "seed_tts_eval"

    def load(
        self,
        *,
        dataset_path: str | None,
        max_samples: int | None,
    ) -> list[NormalizedSample]:
        if dataset_path is None:
            raise ValueError("seed_tts_eval requires --dataset")
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        base_dir = os.path.dirname(dataset_path)
        samples: list[NormalizedSample] = []
        with open(dataset_path, encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                fields = line.split("|")
                if len(fields) < META_FIELD_COUNT:
                    continue
                ref_audio = os.path.join(base_dir, fields[2])
                text = fields[3]
                samples.append(
                    NormalizedSample(
                        sample_id=fields[0],
                        input_preview=text[:INPUT_PREVIEW_LENGTH],
                        payload={
                            "text": text,
                            "references": [
                                {
                                    "audio_path": ref_audio,
                                    "text": fields[1],
                                }
                            ],
                        },
                    )
                )
                if max_samples is not None and len(samples) >= max_samples:
                    break
        return samples
