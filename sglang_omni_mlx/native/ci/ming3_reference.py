"""Evaluate the pinned Python MLX Audio reference on the native ASR corpus."""

from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import distribution, version
from pathlib import Path
from typing import Literal, TypedDict

import mlx.core as mx
from check_golden import CI_DIRECTORY, chip, quality
from mlx_audio.stt import load
from mlx_audio.stt.models.ming_v3_asr.ming_v3_asr import Model


class CorpusClip(TypedDict):
    id: str
    stratum: str
    lang: str
    source: str
    source_id: str
    duration: float
    reference: str


class ReferenceResult(TypedDict):
    text: str
    language: str | None
    generated_token_count: int
    finish_reason: Literal["stop", "length"]
    seconds: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--clip-id", action="append")
    parser.add_argument("--auto-language", action="store_true")
    parser.add_argument("--clips-over-seconds", type=float, default=0)
    arguments = parser.parse_args()
    rows: list[CorpusClip] = [
        json.loads(line)
        for line in (CI_DIRECTORY / "corpus" / "manifest.jsonl")
        .read_text()
        .splitlines()
    ]
    manifest = {
        row["id"]: row
        for row in rows
        if (not arguments.clip_id or row["id"] in arguments.clip_id)
        and row["duration"] > arguments.clips_over_seconds
    }
    started = time.perf_counter()
    model: Model = load(str(arguments.model_path), strict=True)
    load_seconds = time.perf_counter() - started
    outputs: dict[str, ReferenceResult] = {}
    labels = {"en": "en", "zh": "zh", "mixed": "zh", "none": None}
    codes = {
        "English": "en",
        "Chinese": "zh",
        "Canton": "yue",
        "Japanese": "ja",
        "Korean": "ko",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.with_suffix(".jsonl").open("w") as log:
        for clip_id, row in manifest.items():
            audio = arguments.data_root / "corpus" / "v1" / "clips" / f"{clip_id}.wav"
            started = time.perf_counter()
            result = model.generate(
                str(audio),
                language=None if arguments.auto_language else labels[row["lang"]],
                max_tokens=arguments.max_new_tokens,
            )
            elapsed = time.perf_counter() - started
            limit = min(
                arguments.max_new_tokens,
                model.config.text_config.max_position_embeddings - result.prompt_tokens,
            )
            outputs[clip_id] = {
                "text": result.text,
                "language": codes.get(result.language, result.language),
                "generated_token_count": result.generation_tokens,
                "finish_reason": (
                    "length" if result.generation_tokens == limit else "stop"
                ),
                "seconds": elapsed,
            }
            log.write(
                json.dumps({"id": clip_id, **outputs[clip_id]}, ensure_ascii=False)
                + "\n"
            )
            log.flush()
            print(
                f"{len(outputs)}/{len(manifest)} {clip_id}: {elapsed:.3f}s", flush=True
            )
    source = json.loads(distribution("mlx-audio").read_text("direct_url.json") or "{}")
    summary = {
        "device": chip(),
        "mlx_version": version("mlx"),
        "mlx_audio_version": version("mlx-audio"),
        "mlx_audio_revision": source.get("vcs_info", {}).get("commit_id"),
        "load_seconds": load_seconds,
        "total_seconds": sum(row["seconds"] for row in outputs.values()),
        "audio_seconds": sum(row["duration"] for row in manifest.values()),
        "peak_memory_bytes": mx.get_peak_memory(),
        "metrics": quality(
            manifest, {clip_id: row["text"] for clip_id, row in outputs.items()}
        ),
        "clips": outputs,
    }
    arguments.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "clips"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
