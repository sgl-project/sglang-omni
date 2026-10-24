# SPDX-License-Identifier: Apache-2.0
"""Speaker consistency on SeedTTS sessions.

cross-turn: consecutive rows share the first row's reference.
repeat: each row is synthesized --repeats times.

Usage:

    python -m benchmarks.eval.benchmark_tts_consistency \
        --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --meta zhaochenyang20/seed-tts-eval-arrow \
        --lang en --turns-per-session 4 --port 8000

    python -m benchmarks.eval.benchmark_tts_consistency \
        --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --mode repeat --repeats 3 --seed 42 \
        --meta zhaochenyang20/seed-tts-eval-arrow \
        --lang en --port 8000

    python -m benchmarks.eval.benchmark_tts_consistency \
        --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
        --no-ref-audio --voice Vivian --task-type CustomVoice \
        --meta zhaochenyang20/seed-tts-eval-arrow \
        --lang zh --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import soundfile as sf

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import save_json_results
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.metrics.speaker_similarity import WavLMSpeakerSimilarity
from benchmarks.metrics.speaker_similarity_assets import (
    ensure_speaker_similarity_assets,
)
from benchmarks.tasks.tts import (
    build_base_url,
    make_tts_send_fn,
    save_generated_audio_metadata,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

DEFAULT_CONCURRENCY = int(os.getenv("TTS_BENCHMARK_CONCURRENCY", "16"))
DEFAULT_TURNS_PER_SESSION = 4
DEFAULT_REPEATS = 3
MODE_CROSS_TURN = "cross-turn"
MODE_REPEAT = "repeat"
SPEAKER_SIMILARITY_BATCH_SIZE = 2
F0_MIN_HZ = 70.0
F0_MAX_HZ = 400.0
TTS_LANGUAGE_BY_SPLIT = {"en": "English", "zh": "Chinese"}


def tts_language_for_split(lang: str) -> str:
    return TTS_LANGUAGE_BY_SPLIT[lang]


def repeat_sessions(
    samples: Sequence[SampleInput],
    repeats: int,
) -> list[list[SampleInput]]:
    sessions: list[list[SampleInput]] = []
    for sample in samples:
        sessions.append(
            [
                SampleInput(
                    sample_id=f"{sample.sample_id}__r{index}",
                    ref_text=sample.ref_text,
                    ref_audio=sample.ref_audio,
                    target_text=sample.target_text,
                )
                for index in range(repeats)
            ]
        )
    return sessions


def group_sessions(
    samples: Sequence[SampleInput],
    turns_per_session: int,
) -> list[list[SampleInput]]:
    """Group rows; a trailing short group is dropped."""
    sessions: list[list[SampleInput]] = []
    for start in range(0, len(samples), turns_per_session):
        group = list(samples[start : start + turns_per_session])
        if len(group) < turns_per_session:
            break
        anchor = group[0]
        sessions.append(
            [
                SampleInput(
                    sample_id=sample.sample_id,
                    ref_text=anchor.ref_text,
                    ref_audio=anchor.ref_audio,
                    target_text=sample.target_text,
                )
                for sample in group
            ]
        )
    return sessions


def adjacent_abs_deltas(values: Sequence[float | None]) -> list[float]:
    deltas: list[float] = []
    for previous, current in zip(values, values[1:]):
        if previous is None or current is None:
            continue
        deltas.append(abs(float(current) - float(previous)))
    return deltas


def summarize_scores(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "p10": None, "min": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "p10": float(np.percentile(array, 10)),
        "min": float(array.min()),
    }


def speaking_rate_chars_per_s(text: str, duration_s: float) -> float | None:
    """Non-space characters per second. Needed when text length changes."""
    if duration_s <= 0:
        return None
    chars = len("".join(text.split()))
    if chars <= 0:
        return None
    return chars / duration_s


def estimate_f0_hz(
    samples: np.ndarray,
    sample_rate: int,
    *,
    fmin: float = F0_MIN_HZ,
    fmax: float = F0_MAX_HZ,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float64)
    if audio.size == 0:
        return np.asarray([], dtype=np.float64)
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak

    frame_size = max(1, int(sample_rate * frame_ms / 1000.0))
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    min_lag = max(1, int(sample_rate / fmax))
    max_lag = max(min_lag + 1, int(sample_rate / fmin))
    f0: list[float] = []
    window = np.hanning(frame_size)
    for start in range(0, max(1, audio.size - frame_size + 1), hop):
        frame = audio[start : start + frame_size]
        if frame.size < frame_size:
            break
        frame = frame * window
        if float(np.sqrt(np.mean(frame * frame))) < 0.02:
            continue
        corr = np.correlate(frame, frame, mode="full")[frame_size - 1 :]
        if max_lag >= corr.size:
            continue
        region = corr[min_lag : max_lag + 1]
        if region.size == 0:
            continue
        lag = int(np.argmax(region)) + min_lag
        if corr[0] <= 0 or corr[lag] / corr[0] < 0.3:
            continue
        f0.append(sample_rate / float(lag))
    return np.asarray(f0, dtype=np.float64)


def f0_mean_hz(samples: np.ndarray, sample_rate: int) -> float | None:
    values = estimate_f0_hz(samples, sample_rate)
    if values.size == 0:
        return None
    return float(values.mean())


def _parse_token_count(value: str) -> int | str:
    if value.strip().lower() == "auto":
        return "auto"
    try:
        token_count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "token count must be a positive integer or 'auto'"
        ) from exc
    if token_count <= 0:
        raise argparse.ArgumentTypeError("token count must be positive")
    return token_count


def load_consistency_samples(
    meta: str,
    lang: str,
    max_samples: int | None,
    sample_offset: int,
) -> list[SampleInput]:
    if sample_offset < 0:
        raise ValueError(
            f"--sample-offset must be non-negative, got {sample_offset}"
        )
    if sample_offset:
        head = sample_offset + (max_samples or 0)
        return load_seedtts_samples(
            meta, head if max_samples else None, split=lang
        )[sample_offset:]
    return load_seedtts_samples(meta, max_samples, split=lang)


def _generation_kwargs(args: argparse.Namespace) -> dict:
    kwargs: dict = {"language": tts_language_for_split(args.lang)}
    if args.max_new_tokens is not None:
        kwargs["max_new_tokens"] = args.max_new_tokens
    if args.token_count is not None:
        kwargs["token_count"] = args.token_count
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        kwargs["top_p"] = args.top_p
    if args.top_k is not None:
        kwargs["top_k"] = args.top_k
    if args.repetition_penalty is not None:
        kwargs["repetition_penalty"] = args.repetition_penalty
    if args.seed is not None:
        kwargs["seed"] = args.seed
    return kwargs


def _build_results_config(args: argparse.Namespace, base_url: str) -> dict:
    return {
        "model": args.model,
        "base_url": base_url,
        "meta": args.meta,
        "lang": args.lang,
        "language": tts_language_for_split(args.lang),
        "voice": args.voice,
        "task_type": args.task_type,
        "voice_clone": not args.no_ref_audio,
        "ref_format": args.ref_format,
        "no_ref_text": args.no_ref_text,
        "mode": args.mode,
        "turns_per_session": args.turns_per_session,
        "repeats": args.repeats,
        "max_samples": args.max_samples,
        "sample_offset": args.sample_offset,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "concurrency": args.concurrency,
        "device": args.device,
    }


def _score_against_references(
    scorer: WavLMSpeakerSimilarity,
    reference_paths: list[str],
    wav_paths: list[str],
) -> list[float]:
    scores: list[float] = []
    for start in range(0, len(wav_paths), SPEAKER_SIMILARITY_BATCH_SIZE):
        end = start + SPEAKER_SIMILARITY_BATCH_SIZE
        scores.extend(
            scorer.score_batch(reference_paths[start:end], wav_paths[start:end])
        )
    return scores


def _load_speaker_scorer(args: argparse.Namespace) -> WavLMSpeakerSimilarity:
    assets = ensure_speaker_similarity_assets(
        finetune_checkpoint_override=args.similarity_checkpoint,
    )
    return WavLMSpeakerSimilarity(
        finetune_checkpoint=assets.finetune_checkpoint,
        wavlm_base=assets.wavlm_base,
        device=args.device,
    )


def _speaker_similarities(
    args: argparse.Namespace,
    samples: list[SampleInput],
    outputs: list[RequestResult],
    success: list[bool],
    scorer: WavLMSpeakerSimilarity,
) -> list[float | None]:
    if args.no_ref_audio:
        return [None] * len(samples)
    scoreable = [
        index
        for index, ok in enumerate(success)
        if ok and os.path.isfile(samples[index].ref_audio)
    ]
    similarities: list[float | None] = [None] * len(samples)
    if not scoreable:
        return similarities
    scored = _score_against_references(
        scorer,
        [samples[index].ref_audio for index in scoreable],
        [outputs[index].wav_path for index in scoreable],
    )
    for index, score in zip(scoreable, scored):
        similarities[index] = score
    return similarities


def _inter_repeat_similarities(
    sessions: list[list[SampleInput]],
    outputs: list[RequestResult],
    success: list[bool],
    scorer: WavLMSpeakerSimilarity,
) -> list[float | None]:
    similarities: list[float | None] = [None] * len(outputs)
    reference_paths: list[str] = []
    wav_paths: list[str] = []
    score_indexes: list[int] = []
    offset = 0
    for session in sessions:
        first_path: str | None = None
        for local_index, _ in enumerate(session):
            index = offset + local_index
            if not (
                success[index]
                and outputs[index].wav_path
                and os.path.isfile(outputs[index].wav_path)
            ):
                continue
            if first_path is None:
                first_path = outputs[index].wav_path
                continue
            reference_paths.append(first_path)
            wav_paths.append(outputs[index].wav_path)
            score_indexes.append(index)
        offset += len(session)
    if not score_indexes:
        return similarities
    scored = _score_against_references(scorer, reference_paths, wav_paths)
    for index, score in zip(score_indexes, scored):
        similarities[index] = score
    return similarities


def _prosody_tracks(
    samples: list[SampleInput],
    outputs: list[RequestResult],
    success: list[bool],
    sessions: list[list[SampleInput]],
) -> tuple[list[float | None], list[float | None], list[float], list[float]]:
    f0_means: list[float | None] = []
    rates: list[float | None] = []
    for output, sample, ok in zip(outputs, samples, success):
        if not ok:
            f0_means.append(None)
            rates.append(None)
            continue
        audio, sample_rate = sf.read(output.wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        f0_means.append(f0_mean_hz(np.asarray(audio), int(sample_rate)))
        rates.append(
            speaking_rate_chars_per_s(sample.target_text, output.audio_duration_s)
        )

    f0_deltas: list[float] = []
    rate_deltas: list[float] = []
    offset = 0
    for session in sessions:
        width = len(session)
        f0_deltas.extend(adjacent_abs_deltas(f0_means[offset : offset + width]))
        rate_deltas.extend(adjacent_abs_deltas(rates[offset : offset + width]))
        offset += width
    return f0_means, rates, f0_deltas, rate_deltas


def _per_sample(
    samples: list[SampleInput],
    sessions: list[list[SampleInput]],
    similarities: list[float | None],
    f0_means: list[float | None],
    rates: list[float | None],
) -> list[dict]:
    rows = []
    offset = 0
    for session_index, session in enumerate(sessions):
        reference_id = session[0].sample_id
        source_id = reference_id.rsplit("__r", 1)[0]
        for turn_index, sample in enumerate(session):
            index = offset + turn_index
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "source_sample_id": source_id,
                    "repeat_index": turn_index,
                    "session_index": session_index,
                    "reference_sample_id": reference_id,
                    "speaker_similarity": similarities[index],
                    "f0_mean_hz": f0_means[index],
                    "speaking_rate": rates[index],
                }
            )
        offset += len(session)
    return rows


async def _synthesize(
    args: argparse.Namespace,
    samples: list[SampleInput],
) -> list[RequestResult]:
    audio_dir = Path(args.output_dir) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    base_url = build_base_url(args)
    send_fn = make_tts_send_fn(
        args.model,
        f"{base_url}/v1/audio/speech",
        response_format=args.response_format,
        no_ref_audio=args.no_ref_audio,
        ref_format=args.ref_format,
        no_ref_text=args.no_ref_text,
        voice=args.voice,
        task_type=args.task_type,
        instructions=args.instructions,
        save_audio_dir=str(audio_dir),
        **_generation_kwargs(args),
    )
    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=args.concurrency,
            warmup=args.warmup,
            disable_tqdm=args.disable_tqdm,
        )
    )
    return await runner.run(samples, send_fn)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Speaker consistency on SeedTTS sessions."
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL (e.g. http://localhost:8000). Overrides --host/--port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Model name for the API request.",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Preset speaker name. Used with --no-ref-audio.",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default=None,
        help="TTS task type, for example Base, CustomVoice, or VoiceDesign.",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        default=None,
        help="Model-specific style or voice-design instructions.",
    )
    parser.add_argument(
        "--meta",
        "--testset",
        dest="meta",
        type=str,
        default="zhaochenyang20/seed-tts-eval-arrow",
        help="HuggingFace Arrow/Parquet dataset repo id or local meta.lst path.",
    )
    parser.add_argument(
        "--no-ref-audio",
        action="store_true",
        help="Skip ref audio/text from testset (TTS without voice cloning).",
    )
    parser.add_argument(
        "--no-ref-text",
        action="store_true",
        help="Keep the reference clip but drop its transcript.",
    )
    parser.add_argument(
        "--ref-format",
        choices=["flat", "references"],
        default="flat",
        help="Reference payload shape. 'flat' sends ref_audio/ref_text.",
    )
    parser.add_argument("--response-format", type=str, default="wav")
    parser.add_argument("--output-dir", type=str, default="results/tts_consistency")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=[MODE_CROSS_TURN, MODE_REPEAT],
        default=MODE_CROSS_TURN,
        help="cross-turn shares one reference; repeat resynthesizes each row.",
    )
    parser.add_argument(
        "--turns-per-session",
        type=int,
        default=DEFAULT_TURNS_PER_SESSION,
        help="Turns per session in cross-turn mode.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Times to synthesize each row in repeat mode.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--token-count", type=_parse_token_count, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Per-request sampler seed for reproducible generation.",
    )
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument(
        "--concurrency",
        "--max-concurrency",
        dest="concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
    )
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--lang", choices=["en", "zh"], default="en")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for the WavLM speaker-similarity scorer.",
    )
    parser.add_argument(
        "--similarity-checkpoint",
        type=str,
        default=None,
        help="Optional path to a custom fine-tuned WavLM checkpoint. "
        "If omitted, the official weights are downloaded into a local cache "
        "directory (override the cache root with SEEDTTS_SIM_CACHE_DIR).",
    )
    return parser


def _print_summary(summary: dict) -> None:
    similarity = summary["speaker_similarity"]
    print(
        "Speaker similarity: "
        f"mean={similarity['mean']} p10={similarity['p10']} "
        f"min={similarity['min']} ({similarity['n']} evaluated)"
    )
    inter_repeat = summary.get("inter_repeat_similarity")
    if inter_repeat is not None:
        print(
            "Inter-repeat similarity: "
            f"mean={inter_repeat['mean']} p10={inter_repeat['p10']} "
            f"min={inter_repeat['min']} ({inter_repeat['n']} evaluated)"
        )


def _build_sessions(
    args: argparse.Namespace, loaded: list[SampleInput]
) -> list[list[SampleInput]]:
    if args.mode == MODE_REPEAT:
        return repeat_sessions(loaded, args.repeats)
    return group_sessions(loaded, args.turns_per_session)


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.turns_per_session < 1:
        parser.error("--turns-per-session must be at least 1")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    loaded = load_consistency_samples(
        args.meta, args.lang, args.max_samples, args.sample_offset
    )
    sessions = _build_sessions(args, loaded)
    if not sessions:
        raise SystemExit("SeedTTS did not yield any sessions")
    samples = [sample for session in sessions for sample in session]
    outputs = asyncio.run(_synthesize(args, samples))
    save_generated_audio_metadata(outputs, samples, args.output_dir)

    success = [
        bool(output.is_success and output.wav_path and os.path.isfile(output.wav_path))
        for output in outputs
    ]
    need_scorer = (not args.no_ref_audio) or args.mode == MODE_REPEAT
    scorer = _load_speaker_scorer(args) if need_scorer else None
    similarities = (
        _speaker_similarities(args, samples, outputs, success, scorer)
        if scorer is not None
        else [None] * len(samples)
    )
    inter_repeat = (
        _inter_repeat_similarities(sessions, outputs, success, scorer)
        if args.mode == MODE_REPEAT and scorer is not None
        else None
    )
    f0_means, rates, f0_deltas, rate_deltas = _prosody_tracks(
        samples, outputs, success, sessions
    )
    similarity_scores = [score for score in similarities if score is not None]
    if not args.no_ref_audio and not similarity_scores:
        raise SystemExit("no utterances could be scored against a session reference")
    inter_repeat_scores = (
        [score for score in inter_repeat if score is not None]
        if inter_repeat is not None
        else []
    )
    if args.mode == MODE_REPEAT and args.repeats >= 2 and not inter_repeat_scores:
        raise SystemExit("no repeat pairs could be scored against the first synthesis")

    dropped = 0 if args.mode == MODE_REPEAT else len(loaded) - len(samples)
    summary = {
        "sessions": len(sessions),
        "dropped_samples": dropped,
        "samples": len(samples),
        "synthesized": sum(success),
        "speaker_similarity": summarize_scores(similarity_scores),
        "f0_mean_hz": summarize_scores(
            [value for value in f0_means if value is not None]
        ),
        "f0_adjacent_delta_hz": summarize_scores(f0_deltas),
        "speaking_rate": summarize_scores(
            [value for value in rates if value is not None]
        ),
        "speaking_rate_adjacent_delta": summarize_scores(rate_deltas),
    }
    if inter_repeat is not None:
        summary["inter_repeat_similarity"] = summarize_scores(inter_repeat_scores)
    results = {
        "config": _build_results_config(args, build_base_url(args)),
        "summary": summary,
        "per_sample": _per_sample(samples, sessions, similarities, f0_means, rates),
    }
    save_json_results(results, args.output_dir, "consistency.json")
    _print_summary(summary)


if __name__ == "__main__":
    main()
