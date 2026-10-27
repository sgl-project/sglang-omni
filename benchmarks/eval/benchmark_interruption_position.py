# SPDX-License-Identifier: Apache-2.0
"""Record, transcribe and score the interruption-position trials against a turn-based session."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import re
from pathlib import Path

import numpy as np
import soundfile
import whisper
from pydantic import JsonValue
from scipy.signal import resample_poly

from benchmarks.duplex.artifacts import (
    add_server_identity_args,
    package_version,
    replay_run,
    server_identity,
    source_fingerprint,
    turn_detection_argument,
)
from benchmarks.duplex.client import (
    MAX_TIMEOUT_S,
    PACKET_MS,
    SAMPLE_RATE,
    TRANSPORT,
    TURN_BASED_TRANSPORT,
    TurnDetection,
    run_session,
)
from benchmarks.duplex.interruption_position import (
    CLIENT_MODES,
    OUTPUT_SAMPLE_RATE,
    PROFILE,
    START_BUFFER_MS,
    TRANSCRIBED_WORDS,
    TRIALS,
    ClientMode,
    PositionScore,
    TranscribedWord,
    counting_reply,
    position_tables,
    recorded_replies,
    score_run,
)
from benchmarks.duplex.minicpmo_scores import run_passed
from benchmarks.duplex.minicpmo_trials import (
    CLIPS_DIR,
    TRIALS_FILE,
    read_trial_inputs,
    write_inputs,
)
from benchmarks.duplex.v15_audio import TURN_BASED_BARGE_IN_CUT
from benchmarks.duplex.v15_transcribe import (
    ASR_SAMPLE_RATE,
    WHISPER_OPTIONS,
    normalize_words,
)
from benchmarks.eval.benchmark_minicpmo_baselines import set_aside_run

logger = logging.getLogger(__name__)

RUN_NAME_PATTERN = re.compile(
    r"(?P<trial_name>.+)_(?P<client_mode>truncating|silent)_r\d+"
)
TRACE_FILE = "continuous.jsonl"
TRANSCRIPTION_FILE = "transcription.json"
PCM16_SCALE = 32768


def compose_inputs(arguments: argparse.Namespace) -> int:
    trial_inputs = write_inputs(arguments.clips, arguments.output, TRIALS)
    print(f"Wrote {len(trial_inputs)} trial inputs to {arguments.output}")
    return 0


async def record_run(
    run_dir: Path,
    pcm: bytes,
    server: dict[str, JsonValue],
    *,
    url: str,
    client_mode: ClientMode,
    turn_detection: TurnDetection,
    timeout_s: float,
) -> None:
    """Record one session into a new run directory and replay its protocol check."""
    run_dir.mkdir(parents=True)
    (run_dir / "input.pcm").write_bytes(pcm)
    manifest = {
        "schema_version": 1,
        "profile": PROFILE,
        "source": source_fingerprint(),
        "server": server,
        "config": {
            "packet_ms": PACKET_MS,
            "paced": True,
            "timeout_s": timeout_s,
            "input_duration_s": len(pcm) / (SAMPLE_RATE * 2),
            "transport": TRANSPORT,
            "turn_based": {
                "turn_detection": turn_detection.model_dump(mode="json"),
                "barge_in_cut": TURN_BASED_BARGE_IN_CUT,
                **TURN_BASED_TRANSPORT,
            },
            "playback": {
                "start_buffer_ms": START_BUFFER_MS,
                "sends_truncate": client_mode == "truncating",
            },
        },
        "input": {
            "file": "input.pcm",
            "sha256": hashlib.sha256(pcm).hexdigest(),
            "sample_rate": SAMPLE_RATE,
        },
        "cases": [
            {"id": "continuous", "scenario": "continuous", "trace_file": TRACE_FILE}
        ],
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    await run_session(
        url,
        pcm,
        trace_path=run_dir / TRACE_FILE,
        timeout_s=timeout_s,
        profile=PROFILE,
        turn_detection=turn_detection,
        truncate_start_buffer_ms=(
            START_BUFFER_MS if client_mode == "truncating" else None
        ),
    )
    (run_dir / "report.json").write_text(
        json.dumps(replay_run(run_dir), indent=2, allow_nan=False) + "\n"
    )


async def record_rounds(
    arguments: argparse.Namespace, server: dict[str, JsonValue]
) -> None:
    """Record each trial once per round with each client; keep finished runs and set the rest aside to record again."""
    trial_inputs = read_trial_inputs(arguments.inputs)
    longest_trial_s = max(trial_input.duration_s for trial_input in trial_inputs)
    if not longest_trial_s < arguments.timeout <= MAX_TIMEOUT_S:
        raise ValueError(
            f"timeout must exceed the {longest_trial_s} s longest trial and be at most {MAX_TIMEOUT_S} s"
        )
    else:
        pass
    for round_index in range(
        arguments.first_round, arguments.first_round + arguments.rounds
    ):
        for trial_input in trial_inputs:
            input_samples, _ = soundfile.read(
                str(arguments.inputs / f"{trial_input.name}.wav"), dtype="int16"
            )
            pcm = input_samples.tobytes()
            if hashlib.sha256(pcm).hexdigest() != trial_input.sha256:
                raise ValueError(
                    f"{trial_input.name}.wav no longer matches {TRIALS_FILE}"
                )
            else:
                pass
            for client_mode in CLIENT_MODES:
                run_dir = (
                    arguments.output
                    / f"{trial_input.name}_{client_mode}_r{round_index}"
                )
                if (run_dir / "report.json").exists() and not run_passed(run_dir):
                    set_aside_run(run_dir, "failed the protocol check")
                elif run_dir.exists() and not (run_dir / "report.json").exists():
                    set_aside_run(run_dir, "was cut off before its report")
                else:
                    pass
                if run_dir.exists():
                    recorded_sha256 = json.loads(
                        (run_dir / "manifest.json").read_text()
                    )["input"]["sha256"]
                    if recorded_sha256 != trial_input.sha256:
                        raise ValueError(f"{run_dir} was recorded from another input")
                    else:
                        logger.info(f"Keeping recorded {run_dir.name}")
                else:
                    await record_run(
                        run_dir,
                        pcm,
                        server,
                        url=arguments.url,
                        client_mode=client_mode,
                        turn_detection=arguments.turn_detection,
                        timeout_s=arguments.timeout,
                    )
                    if run_passed(run_dir):
                        logger.info(f"Recorded {run_dir.name}")
                    else:
                        set_aside_run(run_dir, "failed the protocol check")


def record(arguments: argparse.Namespace) -> int:
    server = server_identity(
        arguments.url,
        revision=arguments.server_revision,
        model=arguments.model,
        model_revision=arguments.model_revision,
        runtime=arguments.runtime,
    )
    asyncio.run(record_rounds(arguments, server))
    return 0


def transcribe(arguments: argparse.Namespace) -> int:
    """Time the words of each passed run's counting reply with Whisper, independent of the server's own alignment."""
    arguments.output.mkdir(parents=True, exist_ok=True)
    transcription_options = {
        **WHISPER_OPTIONS,
        "fp16": arguments.device.startswith("cuda"),
    }
    transcription_path = arguments.output / TRANSCRIPTION_FILE
    transcription = {
        "model": arguments.model,
        "whisper_version": package_version("openai-whisper"),
        "options": transcription_options,
        "audio": f"the counting reply's delivered audio, resampled to {ASR_SAMPLE_RATE} Hz with scipy.signal.resample_poly",
    }
    if (
        transcription_path.exists()
        and json.loads(transcription_path.read_text()) != transcription
    ):
        raise ValueError(f"{arguments.output} holds words from another transcription")
    else:
        transcription_path.write_text(json.dumps(transcription, indent=2) + "\n")
    model = whisper.load_model(arguments.model, device=arguments.device)
    divisor = math.gcd(OUTPUT_SAMPLE_RATE, ASR_SAMPLE_RATE)
    run_dirs = [
        run_dir
        for run_dir in sorted(arguments.runs.iterdir())
        if RUN_NAME_PATTERN.fullmatch(run_dir.name) is not None
        and (run_dir / "report.json").exists()
        and run_passed(run_dir)
        and not (arguments.output / f"{run_dir.name}.json").exists()
    ]
    for run_dir in run_dirs:
        cut_reply = counting_reply(recorded_replies(run_dir / TRACE_FILE))
        if cut_reply is None:
            logger.warning(f"{run_dir.name} has no counting reply to transcribe")
        else:
            reply_samples = (
                np.frombuffer(bytes(cut_reply.audio), dtype="<i2").astype(np.float32)
                / PCM16_SCALE
            )
            recognized = model.transcribe(
                resample_poly(
                    reply_samples,
                    ASR_SAMPLE_RATE // divisor,
                    OUTPUT_SAMPLE_RATE // divisor,
                ).astype(np.float32),
                **transcription_options,
            )
            try:
                word_chunks = normalize_words(
                    recognized, len(reply_samples) / OUTPUT_SAMPLE_RATE
                )
            except ValueError:
                # note (Jisen Li): A run with unusable word times stays untranscribed; the rest still get theirs.
                logger.exception(f"{run_dir.name} has unusable word times")
            else:
                words = [
                    TranscribedWord(
                        word=word_chunk["text"],
                        start_s=word_chunk["timestamp"][0],
                        end_s=word_chunk["timestamp"][1],
                    )
                    for word_chunk in word_chunks
                ]
                (arguments.output / f"{run_dir.name}.json").write_bytes(
                    TRANSCRIBED_WORDS.dump_json(words, indent=2)
                )
                logger.info(f"Transcribed {run_dir.name}: {len(words)} words")
    return 0


def report(arguments: argparse.Namespace) -> int:
    trial_inputs = read_trial_inputs(arguments.inputs)
    sha256_by_trial_name = {
        trial_input.name: trial_input.sha256 for trial_input in trial_inputs
    }
    scores_by_mode: dict[ClientMode, list[PositionScore]] = {
        client_mode: [] for client_mode in CLIENT_MODES
    }
    skipped_names: list[str] = []
    untranscribed_names: list[str] = []
    for run_dir in sorted(arguments.runs.iterdir()):
        run_name_match = RUN_NAME_PATTERN.fullmatch(run_dir.name)
        words_path = arguments.words / f"{run_dir.name}.json"
        if (
            run_name_match is None
            or run_name_match["trial_name"] not in sha256_by_trial_name
            or not (run_dir / "report.json").exists()
            or not run_passed(run_dir)
        ):
            skipped_names.append(run_dir.name)
        else:
            manifest = json.loads((run_dir / "manifest.json").read_text())
            if (
                manifest["input"]["sha256"]
                != sha256_by_trial_name[run_name_match["trial_name"]]
            ):
                raise ValueError(
                    f"{run_dir} was recorded from another input than {arguments.inputs}"
                )
            elif words_path.exists():
                words = TRANSCRIBED_WORDS.validate_json(words_path.read_bytes())
            else:
                words = None
                untranscribed_names.append(run_dir.name)
            scores_by_mode[run_name_match["client_mode"]].append(
                score_run(
                    recorded_replies(run_dir / TRACE_FILE),
                    words,
                    start_buffer_ms=manifest["config"]["playback"]["start_buffer_ms"],
                )
            )
    if not any(scores_by_mode.values()):
        raise ValueError(
            f"No finished runs of the trials in {arguments.inputs / TRIALS_FILE}"
        )
    else:
        pass
    print(position_tables(scores_by_mode))
    print()
    for client_mode, scores in scores_by_mode.items():
        statuses = [score.status for score in scores]
        print(
            f"{client_mode}: {len(scores)} runs, {statuses.count('no count')} without a counting reply, {statuses.count('not interrupted')} whose count was not cut"
        )
    if untranscribed_names:
        print(
            f"Not transcribed, so left out of the tables: {', '.join(untranscribed_names)}"
        )
    else:
        pass
    if skipped_names:
        print(
            f"Skipped {len(skipped_names)} runs that are unfinished, failed or not in {TRIALS_FILE}: {', '.join(skipped_names)}"
        )
    else:
        pass
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inputs_parser = commands.add_parser(
        "inputs", help="Compose every trial's input WAV from the spoken clips"
    )
    inputs_parser.add_argument(
        "--clips",
        type=Path,
        default=CLIPS_DIR,
        help="Directory of trimmed PCM16 mono 16 kHz clips named as in the README (default: the committed clips)",
    )
    inputs_parser.add_argument(
        "--output", type=Path, required=True, help="New inputs directory"
    )
    inputs_parser.set_defaults(handler=compose_inputs)

    record_parser = commands.add_parser(
        "record",
        help="Record every trial once per round with each client, one session at a time",
    )
    record_parser.add_argument("--inputs", type=Path, required=True)
    record_parser.add_argument("--url", required=True, help="/v1/realtime URL")
    record_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Runs directory; finished runs are kept, so a stopped recording resumes",
    )
    record_parser.add_argument(
        "--turn-detection",
        type=turn_detection_argument,
        required=True,
        help="JSON turn_detection object for session.update",
    )
    add_server_identity_args(record_parser)
    record_parser.add_argument("--rounds", type=int, default=10)
    record_parser.add_argument("--first-round", type=int, default=1)
    record_parser.add_argument(
        "--timeout", type=float, default=90.0, help="Per-session deadline in seconds"
    )
    record_parser.set_defaults(handler=record)

    transcribe_parser = commands.add_parser(
        "transcribe", help="Time the words of every counting reply with Whisper"
    )
    transcribe_parser.add_argument("--runs", type=Path, required=True)
    transcribe_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Words directory; transcribed runs are kept",
    )
    transcribe_parser.add_argument(
        "--model", required=True, help="Whisper model name or local .pt checkpoint"
    )
    transcribe_parser.add_argument("--device", required=True)
    transcribe_parser.set_defaults(handler=transcribe)

    report_parser = commands.add_parser(
        "report", help="Print the history and answer tables"
    )
    report_parser.add_argument("--inputs", type=Path, required=True)
    report_parser.add_argument("--runs", type=Path, required=True)
    report_parser.add_argument(
        "--words", type=Path, required=True, help="Words directory from transcribe"
    )
    report_parser.set_defaults(handler=report)

    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    return arguments.handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
else:
    pass
