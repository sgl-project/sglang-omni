# SPDX-License-Identifier: Apache-2.0
"""Word-timestamped Whisper or Parakeet transcription of recorded duplex output audio."""

from __future__ import annotations

import argparse
import json
import logging
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import soundfile
from numpy.typing import NDArray
from pydantic import JsonValue
from scipy.signal import resample_poly

from benchmarks.duplex.artifacts import package_version, source_fingerprint
from benchmarks.duplex.profiles import PROFILES
from benchmarks.duplex.run_artifacts import (
    TIMELINES,
    Timeline,
    create_output,
    file_sha256,
    load_run,
)
from benchmarks.duplex.v15_audio import write_json

logger = logging.getLogger(__name__)

# note (wenyao): Both the v1.0 and v1.5 CLIs run this module; the run manifest says which.
TRANSCRIBE_FILES = (
    "benchmarks/duplex/v15_audio.py",
    "benchmarks/duplex/run_artifacts.py",
    "benchmarks/duplex/v15_transcribe.py",
)
ASR_SAMPLE_RATE = 16000
WHISPER_OPTIONS = {"language": "en", "word_timestamps": True, "temperature": 0.0}
PARAKEET_OPTIONS = {"timestamps": True}
ASR_PACKAGES = {"whisper": "openai-whisper", "parakeet": "nemo_toolkit"}
# note (wenyao): Float rounding only; any larger overrun is an ASR error, never clipped.
WORD_END_TOLERANCE_S = 1e-3


AsrBackend = Literal["whisper", "parakeet"]


class SpeechRecognizer(Protocol):
    def transcribe(
        self, audio: NDArray[np.float32], **options: bool | float | str
    ) -> dict[str, JsonValue]: ...


class WordHypothesis(Protocol):
    timestamp: dict[str, list[dict[str, JsonValue]]]


class NemoAsrModel(Protocol):
    def transcribe(
        self, audio: list[str], timestamps: bool
    ) -> list[WordHypothesis]: ...


@dataclass(frozen=True)
class ParakeetRecognizer:
    """Return Parakeet word timestamps in the Whisper result shape normalize_words reads.

    Audio goes through a temporary 16 kHz WAV, as the upstream Full-Duplex-Bench
    get_transcript/asr.py feeds Parakeet.
    """

    model: NemoAsrModel

    def transcribe(
        self, audio: NDArray[np.float32], *, timestamps: bool
    ) -> dict[str, JsonValue]:
        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "audio.wav"
            soundfile.write(str(wav_path), audio, ASR_SAMPLE_RATE)
            (hypothesis,) = self.model.transcribe(
                [str(wav_path)], timestamps=timestamps
            )
        words = [
            {"word": word["word"], "start": word["start"], "end": word["end"]}
            for word in hypothesis.timestamp["word"]
        ]
        return {
            "text": " ".join(word["word"] for word in words),
            "segments": [{"words": words}],
        }


def load_mono(path: Path) -> tuple[NDArray[np.float32], int]:
    audio, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), sample_rate


def normalize_words(
    raw: dict[str, JsonValue], duration_s: float
) -> list[dict[str, JsonValue]]:
    """Copy raw word times verbatim after checking them, or raise ValueError."""
    word_chunks = []
    previous_start = 0.0
    for recognized_segment in raw["segments"]:
        for recognized_word in recognized_segment["words"]:
            start_s, end_s = recognized_word["start"], recognized_word["end"]
            if not all(
                isinstance(time_value, (int, float))
                and not isinstance(time_value, bool)
                and math.isfinite(time_value)
                for time_value in (start_s, end_s)
            ):
                raise ValueError(
                    f"word {recognized_word['word']!r} has non-finite times"
                )
            elif not previous_start <= start_s <= end_s:
                raise ValueError(
                    f"word {recognized_word['word']!r} [{start_s}, {end_s}] is negative, "
                    "reversed or out of order"
                )
            elif end_s > duration_s + WORD_END_TOLERANCE_S:
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


def transcribe_run(
    run_dir: Path,
    output: Path,
    *,
    model: SpeechRecognizer,
    model_path: Path,
    device: str,
    timeline: Timeline,
    asr: AsrBackend = "whisper",
) -> dict[str, JsonValue]:
    """Transcribe every qualified variant; per-variant failures are recorded, not fatal."""
    manifest, run, manifest_sha256 = load_run(run_dir)
    output_sample_rate = PROFILES[manifest["profile"]].output_sample_rate
    create_output(output, run_dir)
    if asr == "parakeet":
        transcription_options = dict(PARAKEET_OPTIONS)
    else:
        transcription_options = {**WHISPER_OPTIONS, "fp16": device.startswith("cuda")}
    divisor = math.gcd(output_sample_rate, ASR_SAMPLE_RATE)
    source = source_fingerprint()
    repository_root = Path(__file__).resolve().parents[2]
    source["files_sha256"].update(
        {path: file_sha256(repository_root / path) for path in TRANSCRIBE_FILES}
    )
    transcription_result = {
        "schema_version": 1,
        "kind": "fdb-output-asr",
        "source": source,
        "run": {
            "path": str(run_dir.resolve()),
            "manifest_sha256": manifest_sha256,
            "kind": manifest["kind"],
        },
        "timeline": timeline,
        "asr": {
            "backend": asr,
            "package": ASR_PACKAGES[asr],
            "version": package_version(ASR_PACKAGES[asr]),
            "model_path": str(model_path),
            "model_sha256": file_sha256(model_path),
            "device": device,
            "options": transcription_options,
            "audio": f"mono float32 resampled to {ASR_SAMPLE_RATE} Hz with "
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
            resampled = resample_poly(
                audio,
                ASR_SAMPLE_RATE // divisor,
                output_sample_rate // divisor,
            ).astype(np.float32)
            raw = model.transcribe(resampled, **transcription_options)
            (output / raw_file).parent.mkdir(parents=True, exist_ok=True)
            # note (wenyao): Verbatim, so a NaN the validator rejects is still auditable.
            (output / raw_file).write_text(json.dumps(raw, indent=2) + "\n")
            variant_record["raw_file"] = str(raw_file)
            variant_record.update(
                status="transcribed",
                audio_sha256=file_sha256(run_dir / audio_path),
                duration_s=duration_s,
                transcript={
                    "text": raw["text"].strip(),
                    "chunks": normalize_words(raw, duration_s),
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
    elif arguments.asr == "parakeet":
        # note (Jeffro): NeMo lives in the separate reference scoring environment.
        import torch
        from nemo.collections.asr.models import ASRModel

        nemo_model = ASRModel.restore_from(
            restore_path=str(arguments.model_path),
            map_location=torch.device(arguments.device),
        )
        nemo_model.eval()
        model = ParakeetRecognizer(nemo_model)
    else:
        import whisper

        model = whisper.load_model(str(arguments.model_path), device=arguments.device)
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
        variant_record["status"] for variant_record in transcription_result["variants"]
    ]
    print(
        json.dumps({status: statuses.count(status) for status in sorted(set(statuses))})
    )
    return 1 if "error" in statuses else 0


def add_transcribe_arguments(
    parser: argparse.ArgumentParser,
    timeline_help: str,
    default_asr: AsrBackend = "whisper",
) -> None:
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--asr",
        choices=("whisper", "parakeet"),
        default=default_asr,
        help="parakeet needs NeMo and a local parakeet-tdt-0.6b-v2 .nemo checkpoint",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Local Whisper .pt or Parakeet .nemo checkpoint",
    )
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--timeline", choices=TIMELINES, default="simulated_playout", help=timeline_help
    )
    parser.set_defaults(handler=transcribe_command)
