# SPDX-License-Identifier: Apache-2.0
"""MOSS-Transcribe-Diarize eval on movies800.

Usage:

    python -m benchmarks.eval.eval_transcribe_diarize \
        --max-concurrency 16 \
        --output-dir results/moss_transcribe_diarize_movies800
"""

from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import sys
import time
from collections.abc import Mapping
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
from benchmarks.tasks.transcribe_diarize import (
    MOVIES800_REPO_ID,
    EvaluationPayload,
    Movies800Sample,
    build_evaluation_payload,
    extract_prediction_text,
    load_movies800_samples,
)

MODEL_PATH: Final[str] = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
RESULTS_FILE: Final[str] = "transcribe_diarize_results.json"
DEFAULT_OUTPUT_DIR: Final[str] = "results/moss_transcribe_diarize_movies800"
DEFAULT_SERVER_MEM_FRACTION_STATIC: Final[float] = 0.80
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
DIARIZATION_METRICS_PERCENT_ORDER: Final[tuple[str, ...]] = (
    "cer",
    "cer_no_spk",
    "cp_cer",
    "cer_no_spk_cp_valid",
    "delta_cer",
    "cer_valid_samples",
    "cp_cer_valid_samples",
    "count",
)


def make_send_fn(api_url: str, model_path: str, language: str | None) -> SendFn:
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
                data=_request_form(audio_bytes, audio_path, model_path, language),
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
        ),
    )
    return outputs, runner.wall_clock_s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MOSS-Transcribe-Diarize on movies800 and compare outputs."
    )
    parser.add_argument("--repo-id", default=MOVIES800_REPO_ID)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--expected-column", default="transcription")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
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
    parser.add_argument("--request-timeout-s", type=int, default=300)
    parser.add_argument("--server-timeout-s", type=int, default=600)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
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
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        max_samples = args.max_samples if args.max_samples > 0 else None
        samples = load_movies800_samples(
            repo_id=args.repo_id,
            split=args.split,
            audio_column=args.audio_column,
            expected_column=args.expected_column,
            max_samples=max_samples,
        )
        payload, output_path = _run_with_or_without_server(args, samples)
    except (FileNotFoundError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"\n{'=' * SPEED_LINE_WIDTH}")
    print(f"{'ASR Eval Result':^{SPEED_LINE_WIDTH}}")
    print(f"{'=' * SPEED_LINE_WIDTH}")
    print(f"  {'Model:':<{SPEED_LABEL_WIDTH}} {args.model_path}")
    print(f"  {'Concurrency:':<{SPEED_LABEL_WIDTH}} {args.concurrency}")
    print(f"  {'Output:':<{SPEED_LABEL_WIDTH}} {output_path}")
    print(f"{'=' * SPEED_LINE_WIDTH}")
    print(_build_metrics_section("summary", payload["summary"], SUMMARY_ORDER))
    print(_build_metrics_section("speed", payload["speed"], SPEED_ORDER))
    print(
        _build_metrics_section(
            "diarization_metrics_percent",
            payload["diarization_metrics_percent"],
            DIARIZATION_METRICS_PERCENT_ORDER,
        )
    )
    failed_requests = int(payload["speed"].get("failed_requests", 0) or 0)
    if failed_requests:
        print(
            f"Evaluation failed: {failed_requests} request(s) failed.", file=sys.stderr
        )
        return 1
    return 0


def _run_with_or_without_server(
    args: argparse.Namespace,
    samples: list[Movies800Sample],
) -> tuple[EvaluationPayload, str]:
    base_url = _base_url(args)
    if args.use_existing_server:
        wait_for_service(base_url, timeout=args.server_timeout_s)
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
            )
        )
        payload = _build_payload(args, samples, outputs, wall_clock_s)
        output_path = save_json_results(
            json.loads(json.dumps(payload)),
            args.output_dir,
            RESULTS_FILE,
        )
        return payload, output_path
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
            )
        )
        payload = _build_payload(args, samples, outputs, wall_clock_s)
        output_path = save_json_results(
            json.loads(json.dumps(payload)),
            args.output_dir,
            RESULTS_FILE,
        )
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


def _display_value(section: str, key: str, value: object) -> object:
    if isinstance(value, float):
        return _format_float(section, key, value)
    return value


def _format_float(section: str, key: str, value: float) -> float:
    if section == "summary":
        return round(value, 4)
    if section == "diarization_metrics_percent":
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
) -> aiohttp.FormData:
    form = aiohttp.FormData()
    form.add_field("model", model_path)
    form.add_field("response_format", "verbose_json")
    if language:
        form.add_field("language", language)
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
