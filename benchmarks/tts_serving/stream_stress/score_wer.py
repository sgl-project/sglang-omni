# SPDX-License-Identifier: Apache-2.0
"""Score the audio a stress run saved with --save-audio-dir: WER for English, CER for Chinese.

Run as python -m benchmarks.tts_serving.stream_stress.score_wer from the repository root, after the TTS server has stopped when both share a GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from benchmarks.benchmarker.utils import managed_omni_server
from benchmarks.tasks.asr import (
    DEFAULT_ASR_TRANSCRIBE_CONCURRENCY,
    QWEN3_ASR_MODEL_PATH,
)
from benchmarks.tasks.tts import run_seedtts_transcribe

SCORED_LANGUAGES = ("en", "zh")


@dataclass(kw_only=True)
class StressTranscribeConfig:
    """The fields the shared transcribe pipeline reads."""

    model: str
    output_dir: str
    lang: str
    device: str
    asr_model_path: str
    asr_concurrency: int


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmarks-tts-stream-stress-score-wer",
        description=(
            "Transcribe the audio saved by a stream stress run and score it against "
            "each request's text, with the pipeline the SeedTTS benchmark uses."
        ),
    )
    parser.add_argument(
        "--audio-dir",
        required=True,
        metavar="DIR",
        help="the stress run's --save-audio-dir, which holds generated.json",
    )
    parser.add_argument(
        "--lang",
        choices=SCORED_LANGUAGES,
        default="en",
        help="language of the requests whose corpus line and run set none",
    )
    parser.add_argument(
        "--model", default="tts", help="TTS model name recorded in the results"
    )
    parser.add_argument("--asr-model-path", default=QWEN3_ASR_MODEL_PATH)
    parser.add_argument(
        "--asr-concurrency", type=int, default=DEFAULT_ASR_TRANSCRIBE_CONCURRENCY
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--server-timeout",
        type=int,
        default=1200,
        help="seconds to wait for the ASR server to be ready",
    )
    parser.add_argument(
        "--use-existing-server",
        action="store_true",
        help="score with the ASR server already listening on --port",
    )
    return parser


def split_by_language(audio_dir: Path, default_language: str) -> dict[str, Path]:
    """Write one generated.json per language under the audio directory and return each language's directory."""
    generated: list[dict[str, str | float | bool | None]] = json.loads(
        (audio_dir / "generated.json").read_text(encoding="utf-8")
    )
    entries_by_language: dict[str, list[dict[str, str | float | bool | None]]] = {}
    for entry in generated:
        language = str(entry.get("language") or default_language).lower()
        if language not in SCORED_LANGUAGES:
            raise ValueError(
                f"request {entry['sample_id']} has language {language!r}; "
                f"only {', '.join(SCORED_LANGUAGES)} can be scored"
            )
        else:
            entries_by_language.setdefault(language, []).append(entry)
    output_dirs: dict[str, Path] = {}
    for language, entries in entries_by_language.items():
        output_dir = audio_dir / f"wer_{language}"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "generated.json").write_text(
            json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        output_dirs[language] = output_dir
    return output_dirs


def score(options: argparse.Namespace, output_dirs: dict[str, Path]) -> None:
    for language, output_dir in output_dirs.items():
        run_seedtts_transcribe(
            StressTranscribeConfig(
                model=options.model,
                output_dir=str(output_dir),
                lang=language,
                device=options.device,
                asr_model_path=options.asr_model_path,
                asr_concurrency=options.asr_concurrency,
            ),
            wer_config={
                "model": options.model,
                "asr_model": options.asr_model_path,
                "lang": language,
                "source": "tts_stream_stress",
            },
            generation_mode="streaming",
            asr_router_port=options.port,
        )


def main(argv: Sequence[str] | None = None) -> int:
    options = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    audio_dir = Path(options.audio_dir).expanduser().resolve()
    try:
        output_dirs = split_by_language(audio_dir, options.lang)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if options.use_existing_server:
        score(options, output_dirs)
    else:
        with managed_omni_server(
            model_path=options.asr_model_path,
            port=options.port,
            host=options.host,
            log_file=audio_dir / "asr_server.log",
            timeout=options.server_timeout,
        ):
            score(options, output_dirs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
