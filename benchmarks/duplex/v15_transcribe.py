# SPDX-License-Identifier: Apache-2.0
"""Word-timestamped transcription of recorded model output audio with Parakeet or Whisper."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import math
import tempfile
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import soundfile
from numpy.typing import NDArray
from pydantic import JsonValue
from scipy.signal import resample_poly

from benchmarks.duplex.artifacts import source_fingerprint
from benchmarks.duplex.profiles import PROFILES
from benchmarks.duplex.reference_core import ASR_END_TOLERANCE_S, ASR_MODEL_ID
from benchmarks.duplex.run_artifacts import (
    TIMELINES,
    Timeline,
    create_output,
    file_sha256,
    load_run,
)
from benchmarks.duplex.v15_audio import write_json

logger = logging.getLogger(__name__)

TRANSCRIBE_FILES = (
    "benchmarks/duplex/v15_audio.py",
    "benchmarks/eval/benchmark_duplex_v15.py",
    "benchmarks/duplex/run_artifacts.py",
    "benchmarks/duplex/v15_transcribe.py",
)
WHISPER_SAMPLE_RATE = 16000
WHISPER_OPTIONS = {"language": "en", "word_timestamps": True, "temperature": 0.0}
# note (wenyao): Float rounding only; any larger overrun is an ASR error, never clipped.
WORD_END_TOLERANCE_S = 1e-3
AsrName = Literal["parakeet", "whisper"]


class SpeechRecognizer(Protocol):
    def transcribe(
        self,
        audio: NDArray[np.float32],
        *,
        language: str,
        word_timestamps: bool,
        temperature: float,
        fp16: bool,
    ) -> dict[str, JsonValue]: ...


class ParakeetHypothesis(Protocol):
    timestamp: dict[str, list[dict[str, JsonValue]]]


class ParakeetRecognizer(Protocol):
    def transcribe(
        self, paths: list[str], timestamps: bool
    ) -> list[ParakeetHypothesis]: ...


def load_mono(path: Path) -> tuple[NDArray[np.float32], int]:
    audio, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), sample_rate


def normalize_words(
    words: list[dict[str, JsonValue]], duration_s: float, tolerance_s: float
) -> list[dict[str, JsonValue]]:
    """Copy raw word times verbatim after checking them, or raise ValueError."""
    word_chunks = []
    previous_start = 0.0
    for recognized_word in words:
        start_s, end_s = recognized_word["start"], recognized_word["end"]
        if not all(
            isinstance(time_value, (int, float))
            and not isinstance(time_value, bool)
            and math.isfinite(time_value)
            for time_value in (start_s, end_s)
        ):
            raise ValueError(f"word {recognized_word['word']!r} has non-finite times")
        elif not previous_start <= start_s <= end_s:
            raise ValueError(
                f"word {recognized_word['word']!r} [{start_s}, {end_s}] is negative, "
                "reversed or out of order"
            )
        elif end_s > duration_s + tolerance_s:
            raise ValueError(
                f"word {recognized_word['word']!r} ends at {end_s}s past audio end {duration_s}s"
            )
        else:
            pass
        previous_start = start_s
        word_chunks.append(
            {"text": recognized_word["word"].strip(), "timestamp": [start_s, end_s]}
        )
    return word_chunks


def parakeet_words(
    model: ParakeetRecognizer,
    audio: NDArray[np.float32],
    sample_rate: int,
    offset_s: float,
) -> dict[str, JsonValue]:
    """Transcribe as upstream asr.py does: from offset_s on, with times shifted back."""
    cropped = audio[int(offset_s * sample_rate) :]
    if not len(cropped):
        words = []
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            soundfile.write(audio_file.name, cropped, sample_rate)
            hypothesis = model.transcribe([audio_file.name], timestamps=True)[0]
        words = [
            {
                "word": recognized_word["word"],
                "start": recognized_word["start"] + offset_s,
                "end": recognized_word["end"] + offset_s,
            }
            for recognized_word in hypothesis.timestamp["word"]
        ]
    return {
        "text": " ".join(recognized_word["word"] for recognized_word in words),
        "offset_s": offset_s,
        "words": words,
    }


def transcribe_run(
    run_dir: Path,
    output: Path,
    *,
    model: SpeechRecognizer | ParakeetRecognizer,
    model_path: Path,
    device: str,
    timeline: Timeline,
    asr: AsrName = "whisper",
) -> dict[str, JsonValue]:
    """Transcribe every qualified variant; per-variant failures are recorded, not fatal."""
    manifest, run, manifest_sha256 = load_run(run_dir)
    output_sample_rate = PROFILES[manifest["profile"]].output_sample_rate
    create_output(output, run_dir)
    transcription_options = {**WHISPER_OPTIONS, "fp16": device.startswith("cuda")}
    divisor = math.gcd(output_sample_rate, WHISPER_SAMPLE_RATE)
    if asr == "parakeet":
        asr_record = {
            "package": "nemo_toolkit",
            "version": importlib.metadata.version("nemo_toolkit"),
            "model_id": ASR_MODEL_ID,
            "model_path": str(model_path),
            "model_sha256": file_sha256(model_path),
            "device": device,
            "options": {"timestamps": True},
            "audio": "mono output at its own rate; a user-interruption sample is "
            "transcribed from the interruption end, as upstream asr.py does",
            "word_end_tolerance_s": ASR_END_TOLERANCE_S,
        }
    else:
        asr_record = {
            "package": "openai-whisper",
            "version": importlib.metadata.version("openai-whisper"),
            "model_path": str(model_path),
            "model_sha256": file_sha256(model_path),
            "device": device,
            "options": transcription_options,
            "audio": f"mono float32 resampled to {WHISPER_SAMPLE_RATE} Hz with "
            "scipy.signal.resample_poly",
            "word_end_tolerance_s": WORD_END_TOLERANCE_S,
        }
    # note (Junnan Li): Upstream crops the interruption set at the interruption end before ASR.
    interruption_end_s = {
        sample["id"]: sample["events"][0][1]
        for sample in run["samples"]
        if sample.get("task") == "user_interruption"
    }
    source = source_fingerprint()
    repository_root = Path(__file__).resolve().parents[2]
    source["files_sha256"].update(
        {path: file_sha256(repository_root / path) for path in TRANSCRIBE_FILES}
    )
    transcription_result = {
        "schema_version": 1,
        "kind": "fdb-v15-output-asr",
        "source": source,
        "run": {"path": str(run_dir.resolve()), "manifest_sha256": manifest_sha256},
        "timeline": timeline,
        "asr": asr_record,
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
    write_json(output / "transcripts.json", transcription_result)
    for variant_record in transcription_result["variants"]:
        if variant_record["status"] != "pending":
            continue
        else:
            pass
        audio_path = Path(variant_record["audio"])
        raw_file = Path("raw") / f"{audio_path.parent}.json"
        try:
            audio, sample_rate = load_mono(run_dir / audio_path)
            if sample_rate != output_sample_rate:
                raise ValueError(f"{audio_path} is {sample_rate} Hz")
            else:
                pass
            duration_s = len(audio) / sample_rate
            if asr == "parakeet":
                raw = parakeet_words(
                    model,
                    audio,
                    sample_rate,
                    interruption_end_s.get(variant_record["sample_id"], 0.0),
                )
            else:
                resampled = resample_poly(
                    audio,
                    WHISPER_SAMPLE_RATE // divisor,
                    output_sample_rate // divisor,
                ).astype(np.float32)
                raw = model.transcribe(resampled, **transcription_options)
            (output / raw_file).parent.mkdir(parents=True, exist_ok=True)
            # note (wenyao): Verbatim, so a NaN the validator rejects is still auditable.
            (output / raw_file).write_text(json.dumps(raw, indent=2) + "\n")
            variant_record["raw_file"] = str(raw_file)
            if asr == "parakeet":
                words = raw["words"]
            else:
                words = [
                    recognized_word
                    for recognized_segment in raw["segments"]
                    for recognized_word in recognized_segment["words"]
                ]
            variant_record.update(
                status="transcribed",
                audio_sha256=file_sha256(run_dir / audio_path),
                duration_s=duration_s,
                transcript={
                    "text": raw["text"].strip(),
                    "chunks": normalize_words(
                        words, duration_s, asr_record["word_end_tolerance_s"]
                    ),
                },
            )
        except Exception as exc:
            # note (wenyao): One bad variant must not discard the rest of a long ASR pass.
            logger.exception(
                f"ASR failed for {variant_record['sample_id']}/{variant_record['variant']}"
            )
            variant_record.update(status="error", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "transcripts.json", transcription_result)
    return transcription_result


def transcribe_command(arguments: argparse.Namespace) -> int:
    if not arguments.model_path.is_file():
        raise FileNotFoundError(
            f"--model-path is not a local checkpoint: {arguments.model_path}"
        )
    else:
        if arguments.asr == "parakeet":
            from benchmarks.duplex.reference_asr import load_nemo_model

            model = load_nemo_model(arguments.model_path, arguments.device)
        else:
            import whisper

            model = whisper.load_model(
                str(arguments.model_path), device=arguments.device
            )
        transcription_result = transcribe_run(
            arguments.run,
            arguments.output,
            model=model,
            model_path=arguments.model_path,
            device=arguments.device,
            timeline=arguments.timeline,
            asr=arguments.asr,
        )
        statuses = [
            variant_record["status"]
            for variant_record in transcription_result["variants"]
        ]
        print(
            json.dumps(
                {status: statuses.count(status) for status in sorted(set(statuses))}
            )
        )
        return 1 if "error" in statuses else 0


def add_transcribe_arguments(
    parser: argparse.ArgumentParser, timeline_help: str, default_asr: AsrName
) -> None:
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--asr",
        choices=["parakeet", "whisper"],
        default=default_asr,
        help=f"parakeet is the upstream benchmark's ASR ({ASR_MODEL_ID}, needs "
        "nemo_toolkit); whisper invents words on silent output and is a diagnostic",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Local checkpoint: Parakeet .nemo or Whisper .pt",
    )
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--timeline", choices=TIMELINES, default="simulated_playout", help=timeline_help
    )
    parser.set_defaults(handler=transcribe_command)
