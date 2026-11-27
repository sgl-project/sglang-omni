# SPDX-License-Identifier: Apache-2.0
"""Export SeedTTS evaluation samples as a stress corpus: corpus.jsonl with each sample's reference wav beside it.

Run as python -m benchmarks.tts_serving.stream_stress.export_seedtts_corpus from the repository root.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import load_seedtts_samples


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmarks-tts-stream-stress-export-seedtts-corpus",
        description=(
            "Write the SeedTTS samples as corpus.jsonl for --text-jsonl, with each "
            "line's ref_audio and ref_text, and copy the reference wavs to refs/."
        ),
    )
    parser.add_argument(
        "--meta",
        default=SEEDTTS_DATASET_ID,
        help="SeedTTS dataset repo id, or a local meta.lst",
    )
    parser.add_argument("--split", choices=("en", "zh"), default="en")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", required=True, metavar="DIR")
    parser.add_argument(
        "--language",
        default=None,
        help=(
            "language written on every line and sent with its request; it must be a "
            "value the served model accepts, so it is omitted by default"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    options = build_arg_parser().parse_args(argv)
    samples = load_seedtts_samples(
        options.meta, options.max_samples, split=options.split
    )
    output_dir = Path(options.output_dir).expanduser().resolve()
    reference_dir = output_dir / "refs"
    reference_dir.mkdir(parents=True, exist_ok=True)
    # Note (Jiaxin Deng): many samples share a reference clip, and the loader's staged copies are deleted when this process exits.
    exported_name_by_source: dict[str, str] = {}
    lines: list[str] = []
    for sample in samples:
        if sample.ref_audio not in exported_name_by_source:
            exported_name = (
                f"{len(exported_name_by_source):05d}_{Path(sample.ref_audio).name}"
            )
            shutil.copyfile(sample.ref_audio, reference_dir / exported_name)
            exported_name_by_source[sample.ref_audio] = exported_name
        else:
            pass
        line = {
            "id": sample.sample_id,
            "text": sample.target_text,
            "ref_audio": f"refs/{exported_name_by_source[sample.ref_audio]}",
            "ref_text": sample.ref_text,
        }
        if options.language is not None:
            line["language"] = options.language
        else:
            pass
        lines.append(json.dumps(line, ensure_ascii=False))
    corpus_path = output_dir / "corpus.jsonl"
    corpus_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"{len(lines)} prompts and {len(exported_name_by_source)} reference clips "
        f"written to {corpus_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
