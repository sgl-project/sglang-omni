# SPDX-License-Identifier: Apache-2.0
"""MOSS-Transcribe-Diarize eval.

Evaluate the MOSS-Transcribe-Diarize model on the Movies800Time and AISHELL4
datasets for multi-speaker dialog transcription. These two datasets are private
datasets and can only be accessed with zhaochenyang20's Hugging Face account.

Author:

    Yiyang Zhang https://github.com/CloudRipple
    Chenyang Zhao https://github.com/zhaochenyang20

Usage:

    python -m benchmarks.eval.eval_transcribe_diarize \
        --dataset movies800times \
        --max-concurrency 16 \
        --output-dir results/moss_transcribe_diarize_movies800times

    python -m benchmarks.eval.eval_transcribe_diarize \
        --dataset aishell4_long \
        --max-concurrency 16 \
        --output-dir results/moss_transcribe_diarize_aishell4_long
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import mimetypes
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Final

import aiohttp
import soundfile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig, SendFn
from benchmarks.benchmarker.utils import (
    managed_omni_server,
    save_json_results,
    wait_for_service,
)
from benchmarks.metrics._format import SPEED_LABEL_WIDTH, SPEED_LINE_WIDTH
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.tasks.transcribe_diarize import (
    MOVIES800_REPO_ID,
    EvaluationPayload,
    Movies800Sample,
    build_evaluation_payload,
    extract_prediction_text,
    load_movies800_samples,
)

AISHELL4_REPO_ID: Final[str] = "zhaochenyang20/AISHELL4"
MODEL_PATH: Final[str] = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
RESULTS_FILE: Final[str] = "transcribe_diarize_results.json"
ASR_RESULTS_FILE: Final[str] = "transcribe_diarize_asr_results.json"
SPEED_RESULTS_FILE: Final[str] = "transcribe_diarize_speed_results.json"
DEFAULT_SERVER_MEM_FRACTION_STATIC: Final[float] = 0.80
DEFAULT_MAX_NEW_TOKENS: Final[int] = 65536
MOVIES800TIMES_EXPECTED_SAMPLE_COUNT: Final[int] = 800
AISHELL4_LONG_EXPECTED_SAMPLE_COUNT: Final[int] = 20
MOVIES800TIMES_OUTPUT_DIR: Final[str] = "results/moss_transcribe_diarize_movies800times"
AISHELL4_LONG_OUTPUT_DIR: Final[str] = "results/moss_transcribe_diarize_aishell4_long"
SUMMARY_ORDER: Final[tuple[str, ...]] = (
    "total_samples",
    "evaluated",
    "skipped",
    "exact_matches",
    "mismatches",
    "exact_match_rate",
)
SPEED_ORDER: Final[tuple[str, ...]] = (
    "total_requests",
    "completed_requests",
    "failed_requests",
    "latency_mean_s",
    "latency_median_s",
    "latency_p95_s",
    "latency_p99_s",
    "audio_duration_mean_s",
    "rtf_mean",
    "rtf_median",
    "rtf_p95",
    "rtf_p99",
    "throughput_qps",
    "audio_throughput_s_per_s",
    "output_throughput",
    "output_tok_per_req_s",
    "output_tokens_mean",
    "output_tokens_total",
    "prompt_tokens_mean",
    "prompt_tokens_total",
    "audio_ttfp_mean_s",
    "audio_ttfp_median_s",
    "audio_ttfp_p95_s",
    "audio_ttfp_p99_s",
    "text_ttft_mean_s",
    "text_ttft_median_s",
    "text_ttft_p95_s",
    "text_ttft_p99_s",
    "inter_chunk_mean_s",
    "inter_chunk_p95_s",
    "inter_chunk_p99_s",
    "audio_chunks_mean",
    "audio_chunks_p95",
    "first_audio_payload_bytes_mean",
    "first_audio_payload_bytes_p95",
)
MOVIES800TIMES_DIARIZATION_METRICS_PERCENT_ORDER: Final[tuple[str, ...]] = (
    "cer",
    "cer_no_spk",
    "cer_no_spk_below_50_corpus",
    "n_above_50_pct_cer",
    "pct_above_50_pct_cer",
    "cp_cer",
    "cer_no_spk_cp_valid",
    "delta_cer",
    "speaker_timestamp_der",
    "speaker_timestamp_der_collar",
    "speaker_timestamp_der_valid_samples",
    "speaker_timestamp_der_skipped",
    "speaker_timestamp_der_skipped_parse_error",
    "speaker_timestamp_der_skipped_no_ref_segments",
    "speaker_timestamp_der_skipped_no_pred_segments",
    "speaker_timestamp_der_compute_error",
    "speaker_timestamp_der_total_seconds",
    "speaker_timestamp_der_false_alarm",
    "speaker_timestamp_der_missed_detection",
    "speaker_timestamp_der_confusion",
    "cer_valid_samples",
    "cp_cer_valid_samples",
    "count",
)
AISHELL4_LONG_DIARIZATION_METRICS_PERCENT_ORDER: Final[tuple[str, ...]] = (
    "cer",
    "cer_no_spk",
    "cp_cer",
    "cer_no_spk_cp_valid",
    "delta_cer",
    "speaker_timestamp_der",
    "speaker_timestamp_der_collar",
    "speaker_timestamp_der_valid_samples",
    "speaker_timestamp_der_skipped",
    "speaker_timestamp_der_skipped_parse_error",
    "speaker_timestamp_der_skipped_no_ref_segments",
    "speaker_timestamp_der_skipped_no_pred_segments",
    "speaker_timestamp_der_compute_error",
    "speaker_timestamp_der_total_seconds",
    "speaker_timestamp_der_false_alarm",
    "speaker_timestamp_der_missed_detection",
    "speaker_timestamp_der_confusion",
    "cer_valid_samples",
    "cp_cer_valid_samples",
    "count",
)
MOVIES800TIMES_KEY_METRICS_ORDER: Final[tuple[str, ...]] = (
    "cer_no_spk",
    "cer_no_spk_below_50_corpus",
    "n_above_50_pct_cer",
    "cp_cer",
    "delta_cer",
    "speaker_timestamp_der",
)
AISHELL4_LONG_KEY_METRICS_ORDER: Final[tuple[str, ...]] = (
    "cer_no_spk",
    "cp_cer",
    "delta_cer",
    "speaker_timestamp_der",
)


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    name: str
    description: str
    repo_id: str
    split: str
    audio_column: str
    expected_column: str
    output_dir: str
    expected_sample_count: int | None
    key_metrics_order: tuple[str, ...]
    diarization_metrics_percent_order: tuple[str, ...]


DATASET_CONFIGS: Final[dict[str, DatasetConfig]] = {
    "movies800times": DatasetConfig(
        name="movies800times",
        description="Movies800Time",
        repo_id=MOVIES800_REPO_ID,
        split="validation",
        audio_column="audio",
        expected_column="transcription",
        output_dir=MOVIES800TIMES_OUTPUT_DIR,
        expected_sample_count=MOVIES800TIMES_EXPECTED_SAMPLE_COUNT,
        key_metrics_order=MOVIES800TIMES_KEY_METRICS_ORDER,
        diarization_metrics_percent_order=(
            MOVIES800TIMES_DIARIZATION_METRICS_PERCENT_ORDER
        ),
    ),
    "aishell4_long": DatasetConfig(
        name="aishell4_long",
        description="AISHELL4 long-audio",
        repo_id=AISHELL4_REPO_ID,
        split="validation",
        audio_column="audio",
        expected_column="transcription",
        output_dir=AISHELL4_LONG_OUTPUT_DIR,
        expected_sample_count=AISHELL4_LONG_EXPECTED_SAMPLE_COUNT,
        key_metrics_order=AISHELL4_LONG_KEY_METRICS_ORDER,
        diarization_metrics_percent_order=(
            AISHELL4_LONG_DIARIZATION_METRICS_PERCENT_ORDER
        ),
    ),
}


def make_send_fn(
    api_url: str,
    model_path: str,
    language: str | None,
    max_new_tokens: int | None,
) -> SendFn:
    async def send_fn(
        session: aiohttp.ClientSession,
        sample: Movies800Sample,
    ) -> RequestResult:
        result = RequestResult(request_id=sample.sample_id)
        audio_path = Path(sample.audio_path)
        try:
            audio_bytes = audio_path.read_bytes()
        except OSError as exc:
            result.error = str(exc)
            return result
        result.audio_duration_s = _audio_duration_s(audio_path)
        start = time.perf_counter()
        try:
            async with session.post(
                api_url,
                data=_request_form(
                    audio_bytes,
                    audio_path,
                    model_path,
                    language,
                    max_new_tokens,
                ),
            ) as response:
                if response.status != 200:
                    result.error = f"HTTP {response.status}: {await response.text()}"
                else:
                    result.text = extract_prediction_text(await response.json())
                    result.is_success = True
        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            result.error = str(exc)
        finally:
            result.latency_s = time.perf_counter() - start
        if result.is_success and result.audio_duration_s > 0:
            result.rtf = result.latency_s / result.audio_duration_s
        return result

    return send_fn


async def run_eval(
    samples: list[Movies800Sample],
    *,
    base_url: str,
    model_path: str,
    language: str | None,
    concurrency: int,
    warmup: int,
    request_rate: float,
    disable_tqdm: bool,
    request_timeout_s: int,
    max_new_tokens: int | None = None,
) -> tuple[list[RequestResult], float]:
    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=concurrency,
            request_rate=request_rate,
            warmup=warmup,
            disable_tqdm=disable_tqdm,
            timeout_s=request_timeout_s,
        )
    )
    outputs = await runner.run(
        samples,
        make_send_fn(
            api_url=f"{base_url}/v1/audio/transcriptions",
            model_path=model_path,
            language=language,
            max_new_tokens=max_new_tokens,
        ),
    )
    return outputs, runner.wall_clock_s


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    default_dataset: str = "movies800times",
) -> argparse.Namespace:
    dataset_config = _dataset_config_from_argv(argv, default_dataset)
    parser = argparse.ArgumentParser(
        description=(
            f"Run MOSS-Transcribe-Diarize on {dataset_config.description} "
            "and compare outputs."
        )
    )
    _add_dataset_args(parser, dataset_config)
    _add_request_args(parser)
    _add_server_args(parser)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    default_dataset: str = "movies800times",
) -> int:
    try:
        args = parse_args(argv, default_dataset=default_dataset)
        if args.reuse_asr_results:
            payload, output_path = _run_from_asr_results(args)
        else:
            samples = _load_samples(args)
            payload, output_path = _run_with_or_without_server(args, samples)
    except (FileNotFoundError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _print_results(args, payload, output_path)
    if _failed_request_count(payload):
        print(
            f"Evaluation failed: {_failed_request_count(payload)} request(s) failed.",
            file=sys.stderr,
        )
        return 1
    return 0


def _dataset_config_from_argv(
    argv: Sequence[str] | None,
    default_dataset: str,
) -> DatasetConfig:
    if default_dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown default dataset: {default_dataset}")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_CONFIGS),
        default=default_dataset,
    )
    namespace, _ = parser.parse_known_args(argv)
    return DATASET_CONFIGS[namespace.dataset]


def _add_dataset_args(
    parser: argparse.ArgumentParser,
    dataset_config: DatasetConfig,
) -> None:
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_CONFIGS),
        default=dataset_config.name,
        help="Dataset preset to evaluate.",
    )
    parser.add_argument("--repo-id", default=dataset_config.repo_id)
    parser.add_argument("--split", default=dataset_config.split)
    parser.add_argument("--audio-column", default=dataset_config.audio_column)
    parser.add_argument("--expected-column", default=dataset_config.expected_column)
    parser.add_argument(
        "--max-samples",
        type=_positive_int,
        default=None,
        help="Optional number of samples to evaluate. Defaults to the full split.",
    )
    parser.add_argument("--output-dir", default=dataset_config.output_dir)


def _add_request_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--language")
    parser.add_argument(
        "--concurrency",
        "--max-concurrency",
        dest="concurrency",
        type=int,
        default=16,
        help="Maximum concurrent requests.",
    )
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Requests per second (inf = send all at once).",
    )
    parser.add_argument("--request-timeout-s", type=int, default=1800)
    parser.add_argument(
        "--max-new-tokens",
        type=_positive_int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Optional max_new_tokens forwarded to /v1/audio/transcriptions.",
    )
    parser.add_argument(
        "--asr-results-file",
        default=ASR_RESULTS_FILE,
        help=(
            "Filename under --output-dir where raw ASR request results are saved "
            "immediately after generation finishes."
        ),
    )
    parser.add_argument(
        "--speed-results-file",
        default=SPEED_RESULTS_FILE,
        help=(
            "Filename under --output-dir where speed metrics are saved "
            "immediately after ASR requests finish and before accuracy metrics."
        ),
    )
    parser.add_argument(
        "--reuse-asr-results",
        default=None,
        help=(
            "Load a previously saved raw ASR results JSON and only recompute "
            "post-processing metrics."
        ),
    )


def _add_server_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--server-timeout-s", type=int, default=600)
    parser.add_argument("--use-existing-server", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument(
        "--max-running-requests",
        type=int,
        default=None,
        help=(
            "SGLang generation stage max_running_requests for the managed "
            "server. Defaults to --concurrency."
        ),
    )
    parser.add_argument(
        "--cuda-graph-max-bs",
        type=int,
        default=None,
        help=(
            "SGLang generation stage cuda_graph_max_bs for the managed server. "
            "Defaults to --max-running-requests."
        ),
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=_mem_fraction_static,
        default=DEFAULT_SERVER_MEM_FRACTION_STATIC,
        help=(
            "SGLang static KV-cache memory fraction for the managed server. "
            "MOSS-Transcribe-Diarize keeps headroom for the audio encoder by "
            f"default ({DEFAULT_SERVER_MEM_FRACTION_STATIC})."
        ),
    )
    parser.add_argument(
        "--skip-gpu-cleanup",
        action="store_true",
        help="Do not run the shared GPU cleanup step after a managed server exits.",
    )


def _load_samples(args: argparse.Namespace) -> list[Movies800Sample]:
    dataset_config = DATASET_CONFIGS[args.dataset]
    return load_movies800_samples(
        repo_id=args.repo_id,
        split=args.split,
        audio_column=args.audio_column,
        expected_column=args.expected_column,
        max_samples=args.max_samples,
        expected_sample_count=dataset_config.expected_sample_count,
    )


def _run_with_or_without_server(
    args: argparse.Namespace,
    samples: list[Movies800Sample],
) -> tuple[EvaluationPayload, str]:
    base_url = _base_url(args)
    if args.use_existing_server:
        wait_for_service(base_url, timeout=args.server_timeout_s)
        return _run_requests_and_save(args, samples, base_url)
    log_file = Path(args.output_dir) / "server_logs" / "asr_server.log"
    with managed_omni_server(
        model_path=args.model_path,
        port=args.port,
        host=args.host,
        log_file=log_file,
        max_running_requests=_server_max_running_requests(args),
        cuda_graph_max_bs=_server_cuda_graph_max_bs(args),
        mem_fraction_static=args.mem_fraction_static,
        timeout=args.server_timeout_s,
        wait_for_gpu_release=not args.skip_gpu_cleanup,
    ):
        return _run_requests_and_save(args, samples, base_url)


def _run_requests_and_save(
    args: argparse.Namespace,
    samples: list[Movies800Sample],
    base_url: str,
) -> tuple[EvaluationPayload, str]:
    outputs, wall_clock_s = asyncio.run(
        run_eval(
            samples,
            base_url=base_url,
            model_path=args.model_path,
            language=args.language,
            concurrency=args.concurrency,
            warmup=args.warmup,
            request_rate=args.request_rate,
            disable_tqdm=args.disable_tqdm,
            request_timeout_s=args.request_timeout_s,
            max_new_tokens=args.max_new_tokens,
        )
    )
    _save_asr_results(args, samples, outputs, wall_clock_s)
    _save_and_print_speed_results(args, outputs, wall_clock_s)
    payload = _build_payload(args, samples, outputs, wall_clock_s)
    output_path = _save_payload(args, payload)
    return payload, output_path


def _run_from_asr_results(args: argparse.Namespace) -> tuple[EvaluationPayload, str]:
    samples, outputs, config = _load_asr_results(args.reuse_asr_results)
    wall_clock_s = _wall_clock_s_from_config(config)
    _save_and_print_speed_results(
        args,
        outputs,
        wall_clock_s,
        config_override=config,
    )
    payload = build_evaluation_payload(
        samples=samples,
        outputs=outputs,
        wall_clock_s=wall_clock_s,
        model_path=str(config.get("model_path", args.model_path)),
        concurrency=int(config.get("concurrency", args.concurrency)),
        repo_id=str(config.get("repo_id", args.repo_id)),
        split=str(config.get("split", args.split)),
    )
    output_path = _save_payload(args, payload)
    return payload, output_path


def _base_url(args: argparse.Namespace) -> str:
    return (args.base_url or f"http://{args.host}:{args.port}").rstrip("/")


def _server_max_running_requests(args: argparse.Namespace) -> int:
    if args.max_running_requests is not None:
        return args.max_running_requests
    return args.concurrency


def _server_cuda_graph_max_bs(args: argparse.Namespace) -> int:
    if args.cuda_graph_max_bs is not None:
        return args.cuda_graph_max_bs
    return _server_max_running_requests(args)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _mem_fraction_static(value: str) -> float:
    try:
        fraction = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "mem_fraction_static must be a float in (0, 1)"
        ) from exc
    if not 0.0 < fraction < 1.0:
        raise argparse.ArgumentTypeError(
            "mem_fraction_static must be a float in (0, 1)"
        )
    return fraction


def _build_payload(
    args: argparse.Namespace,
    samples: list[Movies800Sample],
    outputs: list[RequestResult],
    wall_clock_s: float,
) -> EvaluationPayload:
    return build_evaluation_payload(
        samples=samples,
        outputs=outputs,
        wall_clock_s=wall_clock_s,
        model_path=args.model_path,
        concurrency=args.concurrency,
        repo_id=args.repo_id,
        split=args.split,
    )


def _save_payload(args: argparse.Namespace, payload: EvaluationPayload) -> str:
    return save_json_results(
        json.loads(json.dumps(payload)),
        args.output_dir,
        RESULTS_FILE,
    )


def _save_asr_results(
    args: argparse.Namespace,
    samples: list[Movies800Sample],
    outputs: list[RequestResult],
    wall_clock_s: float,
) -> str:
    payload = {
        "schema_version": 1,
        "config": _asr_results_config(args, wall_clock_s),
        "samples": [asdict(sample) for sample in samples],
        "outputs": [asdict(output) for output in outputs],
    }
    path = save_json_results(payload, args.output_dir, args.asr_results_file)
    print(f"Raw ASR results saved before metric computation: {path}", flush=True)
    return path


def _save_and_print_speed_results(
    args: argparse.Namespace,
    outputs: list[RequestResult],
    wall_clock_s: float,
    *,
    config_override: Mapping[str, object] | None = None,
) -> str:
    payload = _build_speed_payload(
        args,
        outputs,
        wall_clock_s,
        config_override=config_override,
    )
    path = save_json_results(payload, args.output_dir, args.speed_results_file)
    _print_speed_results(args, payload, path)
    return path


def _build_speed_payload(
    args: argparse.Namespace,
    outputs: list[RequestResult],
    wall_clock_s: float,
    *,
    config_override: Mapping[str, object] | None = None,
) -> dict[str, object]:
    config = (
        dict(config_override)
        if config_override is not None
        else _asr_results_config(args, wall_clock_s)
    )
    config["wall_clock_s"] = wall_clock_s
    config["timing_scope"] = "asr_requests_only"
    speed = {
        key: value
        for key, value in compute_speed_metrics(
            outputs,
            wall_clock_s=wall_clock_s,
        ).items()
        if isinstance(value, str | int | float | bool) or value is None
    }
    return {
        "schema_version": 1,
        "config": config,
        "speed": speed,
    }


def _print_speed_results(
    args: argparse.Namespace,
    payload: Mapping[str, object],
    output_path: str,
) -> None:
    config = payload.get("config", {})
    speed = payload.get("speed", {})
    if not isinstance(config, Mapping) or not isinstance(speed, Mapping):
        raise TypeError("speed payload must contain mapping 'config' and 'speed'")
    print(f"\n{'=' * SPEED_LINE_WIDTH}")
    print(f"{'ASR Speed Result':^{SPEED_LINE_WIDTH}}")
    print(f"{'=' * SPEED_LINE_WIDTH}")
    print(f"  {'Dataset:':<{SPEED_LABEL_WIDTH}} {config.get('dataset', args.dataset)}")
    print(
        f"  {'Model:':<{SPEED_LABEL_WIDTH}} {config.get('model_path', args.model_path)}"
    )
    print(
        f"  {'Concurrency:':<{SPEED_LABEL_WIDTH}} {config.get('concurrency', args.concurrency)}"
    )
    print(f"  {'Timing:':<{SPEED_LABEL_WIDTH}} ASR requests only")
    print(f"  {'Output:':<{SPEED_LABEL_WIDTH}} {output_path}")
    print(f"{'=' * SPEED_LINE_WIDTH}")
    print(_build_metrics_section("speed", speed, SPEED_ORDER))


def _asr_results_config(
    args: argparse.Namespace,
    wall_clock_s: float,
) -> dict[str, object]:
    return {
        "dataset": getattr(args, "dataset", "custom"),
        "repo_id": args.repo_id,
        "split": args.split,
        "audio_column": args.audio_column,
        "expected_column": args.expected_column,
        "model_path": args.model_path,
        "base_url": _base_url(args),
        "concurrency": args.concurrency,
        "warmup": args.warmup,
        "request_rate": _json_safe_float(args.request_rate),
        "request_timeout_s": args.request_timeout_s,
        "max_new_tokens": args.max_new_tokens,
        "max_samples": args.max_samples,
        "wall_clock_s": wall_clock_s,
        "timing_scope": "asr_requests_only",
    }


def _wall_clock_s_from_config(config: Mapping[str, object]) -> float:
    value = config.get("wall_clock_s", 0.0)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _load_asr_results(
    path: str,
) -> tuple[list[Movies800Sample], list[RequestResult], dict[str, object]]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"ASR results file {path!r} must contain a JSON object")
    samples_payload = payload.get("samples")
    outputs_payload = payload.get("outputs")
    config_payload = payload.get("config", {})
    if not isinstance(samples_payload, list) or not isinstance(outputs_payload, list):
        raise ValueError(
            f"ASR results file {path!r} must contain 'samples' and 'outputs' lists"
        )
    if not isinstance(config_payload, dict):
        raise ValueError(f"ASR results file {path!r} has invalid 'config'")
    samples = _load_sample_records(samples_payload)
    outputs = _load_request_results(outputs_payload)
    return samples, outputs, config_payload


def _load_sample_records(records: list[object]) -> list[Movies800Sample]:
    return [
        Movies800Sample(
            sample_id=str(record["sample_id"]),
            audio_path=str(record["audio_path"]),
            expected_text=str(record["expected_text"]),
        )
        for record in records
        if isinstance(record, Mapping)
    ]


def _load_request_results(records: list[object]) -> list[RequestResult]:
    result_fields = {field.name for field in fields(RequestResult)}
    outputs: list[RequestResult] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        values = {key: record[key] for key in result_fields if key in record}
        outputs.append(RequestResult(**values))
    return outputs


def _json_safe_float(value: float) -> float | str:
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def _print_results(
    args: argparse.Namespace,
    payload: EvaluationPayload,
    output_path: str,
) -> None:
    dataset_config = DATASET_CONFIGS[args.dataset]
    print(f"\n{'=' * SPEED_LINE_WIDTH}")
    print(f"{'ASR Eval Result':^{SPEED_LINE_WIDTH}}")
    print(f"{'=' * SPEED_LINE_WIDTH}")
    print(f"  {'Dataset:':<{SPEED_LABEL_WIDTH}} {args.dataset}")
    print(f"  {'Model:':<{SPEED_LABEL_WIDTH}} {args.model_path}")
    print(f"  {'Concurrency:':<{SPEED_LABEL_WIDTH}} {args.concurrency}")
    print(f"  {'Output:':<{SPEED_LABEL_WIDTH}} {output_path}")
    print(f"{'=' * SPEED_LINE_WIDTH}")
    print(
        _build_key_metrics_section(
            payload["diarization_metrics_percent"],
            dataset_config.key_metrics_order,
        )
    )
    print(_build_metrics_section("summary", payload["summary"], SUMMARY_ORDER))
    print(_build_metrics_section("speed", payload["speed"], SPEED_ORDER))
    print(
        _build_metrics_section(
            "diarization_metrics_percent",
            payload["diarization_metrics_percent"],
            dataset_config.diarization_metrics_percent_order,
        )
    )


def _failed_request_count(payload: EvaluationPayload) -> int:
    return int(payload["speed"].get("failed_requests", 0) or 0)


def _build_metrics_section(
    title: str,
    metrics: Mapping[str, object],
    key_order: tuple[str, ...],
) -> str:
    lines = [f"\n{title}", "-" * SPEED_LINE_WIDTH]
    seen_keys: set[str] = set()
    for key in key_order:
        if key not in metrics:
            continue
        seen_keys.add(key)
        lines.append(
            f"  {key + ':':<{SPEED_LABEL_WIDTH}} {_display_value(title, key, metrics[key])}"
        )
    for key in sorted(metrics):
        if key in seen_keys:
            continue
        lines.append(
            f"  {key + ':':<{SPEED_LABEL_WIDTH}} {_display_value(title, key, metrics[key])}"
        )
    return "\n".join(lines)


def _build_key_metrics_section(
    metrics: Mapping[str, object],
    key_order: tuple[str, ...] = MOVIES800TIMES_KEY_METRICS_ORDER,
) -> str:
    lines = ["\nkey_metrics", "-" * SPEED_LINE_WIDTH]
    for key in key_order:
        if key not in metrics:
            continue
        lines.append(
            f"  {key + ':':<{SPEED_LABEL_WIDTH}} {_display_value('diarization_metrics_percent', key, metrics[key])}"
        )
    return "\n".join(lines)


def _display_value(section: str, key: str, value: object) -> object:
    if isinstance(value, float):
        return _format_float(section, key, value)
    return value


def _format_float(section: str, key: str, value: float) -> float:
    if section == "summary":
        return round(value, 4)
    if section == "diarization_metrics_percent":
        if key.endswith(
            ("_seconds", "_false_alarm", "_missed_detection", "_confusion", "_collar")
        ):
            return round(value, 4)
        return round(value, 2)
    if "rtf" in key:
        return round(value, 4)
    return round(value, 3)


def _audio_duration_s(audio_path: Path) -> float:
    try:
        return float(soundfile.info(str(audio_path)).duration)
    except RuntimeError:
        return 0.0


def _request_form(
    audio_bytes: bytes,
    audio_path: Path,
    model_path: str,
    language: str | None,
    max_new_tokens: int | None,
) -> aiohttp.FormData:
    form = aiohttp.FormData()
    form.add_field("model", model_path)
    form.add_field("response_format", "verbose_json")
    if language:
        form.add_field("language", language)
    if max_new_tokens is not None:
        form.add_field("max_new_tokens", str(max_new_tokens))
    form.add_field(
        "file",
        audio_bytes,
        filename=audio_path.name,
        content_type=mimetypes.guess_type(audio_path.name)[0]
        or "application/octet-stream",
    )
    return form


if __name__ == "__main__":
    raise SystemExit(main())
