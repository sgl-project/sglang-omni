# SPDX-License-Identifier: Apache-2.0
"""Word-timestamped Whisper transcription of recorded v1.5 model output audio."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import math
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile
from pydantic import JsonValue
from scipy.signal import resample_poly

from benchmarks.duplex.artifacts import source_fingerprint
from benchmarks.duplex.profiles import PROFILES
from benchmarks.duplex.v15_audio import write_json
from benchmarks.duplex.v15_evaluation import (
    TIMELINES,
    Timeline,
    create_output,
    file_sha256,
    load_run,
)

logger = logging.getLogger(__name__)

TRANSCRIBE_FILES = (
    "benchmarks/duplex/v15_audio.py",
    "benchmarks/eval/benchmark_duplex_v15.py",
    "benchmarks/duplex/v15_evaluation.py",
    "benchmarks/duplex/v15_transcribe.py",
)
WHISPER_SAMPLE_RATE = 16000
WHISPER_OPTIONS = {"language": "en", "word_timestamps": True, "temperature": 0.0}
# Note (wenyao): Float rounding only; any larger overrun is an ASR error, never clipped.
WORD_END_TOLERANCE_S = 1e-3


class SpeechRecognizer(Protocol):
    def transcribe(
        self, audio: np.ndarray, **options: JsonValue
    ) -> dict[str, JsonValue]: ...


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), sample_rate


def normalize_words(
    raw: dict[str, JsonValue], duration_s: float
) -> list[dict[str, JsonValue]]:
    """Copy raw word times verbatim after checking them, or raise ValueError."""
    chunks = []
    previous_start = 0.0
    for segment in raw["segments"]:
        for word in segment["words"]:
            start_s, end_s = word["start"], word["end"]
            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in (start_s, end_s)
            ):
                raise ValueError(f"word {word['word']!r} has non-finite times")
            elif not previous_start <= start_s <= end_s:
                raise ValueError(
                    f"word {word['word']!r} [{start_s}, {end_s}] is negative, "
                    "reversed or out of order"
                )
            elif end_s > duration_s + WORD_END_TOLERANCE_S:
                raise ValueError(
                    f"word {word['word']!r} ends at {end_s}s past audio end {duration_s}s"
                )
            previous_start = start_s
            chunks.append({"text": word["word"].strip(), "timestamp": [start_s, end_s]})
    return chunks


def transcribe_run(
    run_dir: Path,
    output: Path,
    *,
    model: SpeechRecognizer,
    model_path: Path,
    device: str,
    timeline: Timeline,
) -> dict[str, JsonValue]:
    """Transcribe every qualified variant; per-variant failures are recorded, not fatal."""
    manifest, run, manifest_sha256 = load_run(run_dir)
    output_sample_rate = PROFILES[manifest["profile"]].output_sample_rate
    create_output(output, run_dir)
    options = {**WHISPER_OPTIONS, "fp16": device.startswith("cuda")}
    divisor = math.gcd(output_sample_rate, WHISPER_SAMPLE_RATE)
    source = source_fingerprint()
    repo = Path(__file__).resolve().parents[2]
    source["files_sha256"].update(
        {path: file_sha256(repo / path) for path in TRANSCRIBE_FILES}
    )
    result = {
        "schema_version": 1,
        "kind": "fdb-v15-output-asr",
        "source": source,
        "run": {"path": str(run_dir.resolve()), "manifest_sha256": manifest_sha256},
        "timeline": timeline,
        "asr": {
            "package": "openai-whisper",
            "version": importlib.metadata.version("openai-whisper"),
            "model_path": str(model_path),
            "model_sha256": file_sha256(model_path),
            "device": device,
            "options": options,
            "audio": f"mono float32 resampled to {WHISPER_SAMPLE_RATE} Hz with "
            "scipy.signal.resample_poly",
            "word_end_tolerance_s": WORD_END_TOLERANCE_S,
        },
        "variants": [
            {
                "sample_id": sample["id"],
                "variant": variant,
                "status": (
                    "pending"
                    if state["qualified"]
                    else f"not_qualified:{state['status']}"
                ),
                "audio": f"{state['directory']}/{TIMELINES[timeline]['audio']}",
            }
            for sample in run["samples"]
            for variant, state in sample["variants"].items()
        ],
    }
    write_json(output / "transcripts.json", result)
    for entry in result["variants"]:
        if entry["status"] != "pending":
            continue
        audio_path = Path(entry["audio"])
        raw_file = Path("raw") / f"{audio_path.parent}.json"
        try:
            audio, sample_rate = load_mono(run_dir / audio_path)
            if sample_rate != output_sample_rate:
                raise ValueError(f"{audio_path} is {sample_rate} Hz")
            duration_s = len(audio) / sample_rate
            resampled = resample_poly(
                audio,
                WHISPER_SAMPLE_RATE // divisor,
                output_sample_rate // divisor,
            ).astype(np.float32)
            raw = model.transcribe(resampled, **options)
            (output / raw_file).parent.mkdir(parents=True, exist_ok=True)
            # Note (wenyao): Verbatim, so a NaN the validator rejects is still auditable.
            (output / raw_file).write_text(json.dumps(raw, indent=2) + "\n")
            entry["raw_file"] = str(raw_file)
            entry.update(
                status="transcribed",
                audio_sha256=file_sha256(run_dir / audio_path),
                duration_s=duration_s,
                transcript={
                    "text": raw["text"].strip(),
                    "chunks": normalize_words(raw, duration_s),
                },
            )
        except Exception as exc:
            # Note (wenyao): One bad variant must not discard the rest of a long ASR pass.
            logger.exception(f"ASR failed for {entry['sample_id']}/{entry['variant']}")
            entry.update(status="error", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "transcripts.json", result)
    return result
