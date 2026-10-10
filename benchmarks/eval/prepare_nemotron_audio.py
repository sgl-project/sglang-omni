"""Prepare shared PCM16 inputs for Nemotron service and reference evaluation."""

import argparse
import json
import math
from pathlib import Path

import soundfile
from scipy.signal import resample_poly

from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID, SEEDTTS_DATASET_REVISION
from benchmarks.dataset.seedtts import load_seedtts_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", default=SEEDTTS_DATASET_ID)
    parser.add_argument("--dataset-revision", default=SEEDTTS_DATASET_REVISION)
    parser.add_argument("--language", choices=["en", "zh"], default="en")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    samples = load_seedtts_samples(
        arguments.meta,
        max_samples=arguments.max_samples,
        split=arguments.language,
        revision=arguments.dataset_revision,
    )
    if not samples or len({sample.sample_id for sample in samples}) != len(samples):
        parser.error("Require nonempty metadata with unique sample identifiers")
    elif any("|" in sample.ref_text or "\n" in sample.ref_text for sample in samples):
        parser.error("Reference text cannot contain a pipe or newline in meta.lst")
    else:
        arguments.output.mkdir(parents=True, exist_ok=False)
    audio_paths: dict[str, str] = {}
    row_audio_seconds = 0.0
    with (arguments.output / "meta.lst").open("w", encoding="utf-8") as metadata:
        for sample in samples:
            if sample.ref_audio not in audio_paths:
                audio_channels, sample_rate = soundfile.read(
                    sample.ref_audio,
                    dtype="float32",
                    always_2d=True,
                )
                waveform = audio_channels.mean(axis=1)
                if sample_rate != 16000:
                    divisor = math.gcd(sample_rate, 16000)
                    waveform = resample_poly(
                        waveform, 16000 // divisor, sample_rate // divisor
                    )
                else:
                    pass
                filename = f"{len(audio_paths):05d}.wav"
                soundfile.write(
                    arguments.output / filename, waveform, 16000, subtype="PCM_16"
                )
                audio_paths[sample.ref_audio] = filename
            else:
                filename = audio_paths[sample.ref_audio]
            row_audio_seconds += soundfile.info(arguments.output / filename).duration
            metadata.write(
                f"{sample.sample_id}|{sample.ref_text}|{filename}|{sample.ref_text}\n"
            )
    manifest = {
        **vars(arguments),
        "output": str(arguments.output),
        "rows": len(samples),
        "unique_source_paths": len(audio_paths),
        "row_audio_seconds": row_audio_seconds,
        "sample_ids": [sample.sample_id for sample in samples],
    }
    (arguments.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
else:
    pass
