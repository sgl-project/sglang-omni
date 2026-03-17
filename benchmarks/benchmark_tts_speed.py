"""
Benchmark online serving for TTS (text-to-speech) models.

This script profiles the speed of TTS inference via the /v1/audio/speech HTTP
API. It supports two task types:

  1. Voice cloning (default) -- Each request carries a reference audio clip and
     its transcript. The model synthesizes the target text in the voice of the
     reference speaker. This is the standard evaluation mode used by
     seed-tts-eval.

  2. Plain TTS (--no-ref-audio) -- No reference audio is provided. The model
     generates speech with its default voice. Useful for measuring raw
     generation speed without the voice-cloning overhead.

Both modes read samples from a seed-tts-eval meta.lst file. Each line in the
file has the format: id|ref_text|ref_audio_path|text_to_synthesize

Dataset: seed-tts-eval

    The seed-tts-eval testset (from BytedanceSpeech) contains samples from
    public speech corpora for objective TTS evaluation:

      - en/meta.lst : 1000 English samples from CommonVoice
      - zh/meta.lst : 2000 Chinese samples from DiDiSpeech-2

    Download from Google Drive:

        pip install gdown
        gdown 1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP -O seed-tts-eval.tar
        tar xf seed-tts-eval.tar

Usage:

    # Launch a server first:
    uv pip install -v ".[s2pro]"

    python -m sglang_omni.cli.cli serve \
        --model-path fishaudio/s2-pro \
        --config examples/configs/s2pro_tts.yaml \
        --port 8000

    # Benchmark voice cloning (with ref audio, default):

    Note that 20 samples shall take 120s+

    python -m benchmarks.benchmark_tts_speed \
        --model fishaudio/s2-pro --port 8000 \
        --testset seedtts_testset/en/meta.lst \
        --max-samples 10

    # Benchmark plain TTS (without ref audio):
    python -m benchmarks.benchmark_tts_speed \
        --model fishaudio/s2-pro --port 8000 \
        --testset seedtts_testset/en/meta.lst \
         --no-ref-audio --max-samples 20
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import struct
import time
from dataclasses import dataclass

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class RequestFuncInput:
    request_id: str
    text: str
    api_url: str
    model: str
    ref_audio: str | None = None
    ref_text: str | None = None
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None


@dataclass
class RequestFuncOutput:
    request_id: str = ""
    text: str = ""
    success: bool = False
    latency: float = 0.0
    audio_duration_s: float = 0.0
    rtf: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    engine_time_s: float = 0.0
    tok_per_s: float = 0.0
    error: str = ""


def parse_meta_lst(path: str, max_samples: int | None = None) -> list[dict]:
    """Parse a seed-tts-eval meta.lst file into a list of sample dicts.

    Each line in the file has format: ``id|ref_text|ref_audio_path|text``.
    Relative audio paths are resolved against the directory containing the
    meta.lst file.
    """
    base_dir = os.path.dirname(path)
    samples: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            samples.append(
                {
                    "id": parts[0],
                    "ref_text": parts[1],
                    "ref_audio": os.path.join(base_dir, parts[2]),
                    "text": parts[3],
                }
            )
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def _wav_duration(data: bytes) -> float:
    """Return duration in seconds from a WAV byte buffer, or 0.0 on error."""
    if len(data) <= 44:
        return 0.0
    sample_rate = struct.unpack_from("<I", data, 24)[0]
    num_channels = struct.unpack_from("<H", data, 22)[0]
    bits_per_sample = struct.unpack_from("<H", data, 34)[0]
    if sample_rate == 0 or num_channels == 0 or bits_per_sample == 0:
        return 0.0
    bytes_per_sample = num_channels * bits_per_sample // 8
    data_size = len(data) - 44
    return data_size / (sample_rate * bytes_per_sample)


async def send_tts_request(
    request: RequestFuncInput,
    session: aiohttp.ClientSession,
    save_audio_dir: str | None = None,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    output = RequestFuncOutput(request_id=request.request_id, text=request.text[:60])

    payload: dict = {
        "model": request.model,
        "input": request.text,
        "response_format": "wav",
    }
    if request.ref_audio is not None:
        payload["ref_audio"] = request.ref_audio
    if request.ref_text is not None:
        payload["ref_text"] = request.ref_text
    if request.max_new_tokens is not None:
        payload["max_new_tokens"] = request.max_new_tokens
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.top_k is not None:
        payload["top_k"] = request.top_k
    if request.repetition_penalty is not None:
        payload["repetition_penalty"] = request.repetition_penalty

    start_time = time.perf_counter()
    try:
        async with session.post(request.api_url, json=payload) as response:
            if response.status == 200:
                audio_bytes = await response.read()
                output.latency = time.perf_counter() - start_time
                output.success = True
                output.audio_duration_s = _wav_duration(audio_bytes)
                if output.audio_duration_s > 0:
                    output.rtf = output.latency / output.audio_duration_s
                else:
                    output.rtf = float("inf")
                prompt_tok = response.headers.get("X-Prompt-Tokens")
                comp_tok = response.headers.get("X-Completion-Tokens")
                eng_time = response.headers.get("X-Engine-Time")
                if prompt_tok is not None:
                    output.prompt_tokens = int(prompt_tok)
                if comp_tok is not None:
                    output.completion_tokens = int(comp_tok)
                if eng_time is not None:
                    output.engine_time_s = float(eng_time)
                if output.completion_tokens > 0 and output.engine_time_s > 0:
                    output.tok_per_s = output.completion_tokens / output.engine_time_s
                if save_audio_dir and audio_bytes:
                    path = os.path.join(save_audio_dir, f"{request.request_id}.wav")
                    with open(path, "wb") as f:
                        f.write(audio_bytes)
            else:
                output.latency = time.perf_counter() - start_time
                output.error = f"HTTP {response.status}: {await response.text()}"
    except Exception as e:
        output.latency = time.perf_counter() - start_time
        output.error = str(e)

    if pbar:
        pbar.update(1)
    return output


def wait_for_service(base_url: str, timeout: int = 1200) -> None:
    import requests as req_lib

    logger.info("Waiting for service at %s ...", base_url)
    start = time.time()
    while True:
        try:
            resp = req_lib.get(f"{base_url}/health", timeout=1)
            if resp.status_code == 200:
                logger.info("Service is ready.")
                return
        except req_lib.exceptions.RequestException:
            pass
        if time.time() - start > timeout:
            raise TimeoutError(f"Service at {base_url} not ready within {timeout}s")
        time.sleep(1)


def calculate_metrics(outputs: list[RequestFuncOutput]) -> dict:
    successes = [o for o in outputs if o.success]
    if not successes:
        return {"completed_requests": 0, "failed_requests": len(outputs)}

    latencies = [o.latency for o in successes]
    rtfs = [o.rtf for o in successes if o.rtf < float("inf")]
    audio_durs = [o.audio_duration_s for o in successes]
    toks = [o.tok_per_s for o in successes if o.tok_per_s > 0]
    gen_tokens = [o.completion_tokens for o in successes if o.completion_tokens > 0]

    total_wall = sum(latencies)
    total_tokens = sum(gen_tokens)
    total_engine_time = sum(o.engine_time_s for o in successes if o.engine_time_s > 0)
    agg_tok_per_s = (
        total_tokens / total_engine_time
        if total_engine_time > 0 and total_tokens > 0
        else 0
    )

    result = {
        "completed_requests": len(successes),
        "failed_requests": len(outputs) - len(successes),
        "latency_mean_s": round(float(np.mean(latencies)), 3),
        "latency_median_s": round(float(np.median(latencies)), 3),
        "latency_p95_s": round(float(np.percentile(latencies, 95)), 3),
        "latency_p99_s": round(float(np.percentile(latencies, 99)), 3),
        "audio_duration_mean_s": (
            round(float(np.mean(audio_durs)), 3) if audio_durs else 0
        ),
        "rtf_mean": round(float(np.mean(rtfs)), 4) if rtfs else None,
        "rtf_median": round(float(np.median(rtfs)), 4) if rtfs else None,
        "throughput_qps": (
            round(len(successes) / total_wall, 3) if total_wall > 0 else 0
        ),
    }
    if toks:
        result["tok_per_s_mean"] = round(float(np.mean(toks)), 1)
        result["tok_per_s_median"] = round(float(np.median(toks)), 1)
    if agg_tok_per_s > 0:
        result["tok_per_s_agg"] = round(agg_tok_per_s, 1)
    if gen_tokens:
        result["gen_tokens_mean"] = round(float(np.mean(gen_tokens)), 0)
        result["gen_tokens_total"] = total_tokens
    return result


def print_summary(metrics: dict, args: argparse.Namespace) -> None:
    w = 60
    print(f"\n{'=' * w}")
    print(f"{'TTS Benchmark Result':^{w}}")
    print(f"{'=' * w}")
    print(f"  {'Model:':<30} {args.model}")
    print(f"  {'Completed requests:':<30} {metrics['completed_requests']}")
    print(f"  {'Failed requests:':<30} {metrics['failed_requests']}")
    print(f"{'-' * w}")
    print(f"  {'Latency mean (s):':<30} {metrics.get('latency_mean_s', 'N/A')}")
    print(f"  {'Latency median (s):':<30} {metrics.get('latency_median_s', 'N/A')}")
    print(f"  {'Latency p95 (s):':<30} {metrics.get('latency_p95_s', 'N/A')}")
    print(f"  {'Latency p99 (s):':<30} {metrics.get('latency_p99_s', 'N/A')}")
    if metrics.get("rtf_mean") is not None:
        print(f"  {'RTF mean:':<30} {metrics['rtf_mean']}")
        print(f"  {'RTF median:':<30} {metrics['rtf_median']}")
    if metrics.get("audio_duration_mean_s"):
        print(f"  {'Audio duration mean (s):':<30} {metrics['audio_duration_mean_s']}")
    if metrics.get("tok_per_s_mean") is not None:
        print(f"  {'Tok/s (per-req mean):':<30} {metrics['tok_per_s_mean']}")
        print(f"  {'Tok/s (per-req median):':<30} {metrics['tok_per_s_median']}")
    if metrics.get("tok_per_s_agg") is not None:
        print(f"  {'Tok/s (aggregate):':<30} {metrics['tok_per_s_agg']}")
    if metrics.get("gen_tokens_mean") is not None:
        print(f"  {'Gen tokens (mean):':<30} {metrics['gen_tokens_mean']:.0f}")
        print(f"  {'Gen tokens (total):':<30} {metrics['gen_tokens_total']}")
    print(f"  {'Throughput (req/s):':<30} {metrics.get('throughput_qps', 'N/A')}")
    print(f"{'=' * w}")


async def benchmark(args: argparse.Namespace) -> None:
    base_url = args.base_url or f"http://{args.host}:{args.port}"
    api_url = f"{base_url}/v1/audio/speech"

    wait_for_service(base_url)

    if not os.path.isfile(args.testset):
        logger.error("Testset not found: %s", args.testset)
        return

    samples = parse_meta_lst(args.testset, args.max_samples)
    if args.no_ref_audio:
        for s in samples:
            s["ref_audio"] = None
            s["ref_text"] = None

    logger.info("Prepared %d requests", len(samples))

    requests_list = [
        RequestFuncInput(
            request_id=s["id"],
            text=s["text"],
            api_url=api_url,
            model=args.model,
            ref_audio=s.get("ref_audio"),
            ref_text=s.get("ref_text"),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        for s in samples
    ]

    save_audio_dir = None
    if args.save_audio and args.output_dir:
        save_audio_dir = os.path.join(args.output_dir, "audio")
        os.makedirs(save_audio_dir, exist_ok=True)

    if args.warmup > 0:
        logger.info("Warmup (%d requests)...", args.warmup)
        async with aiohttp.ClientSession() as session:
            for i in range(min(args.warmup, len(requests_list))):
                result = await send_tts_request(requests_list[i], session)
                status = "ok" if result.success else result.error
                logger.info("  warmup %d/%d: %s", i + 1, args.warmup, status)

    logger.info(
        "Benchmarking %d requests (max_concurrency=%s)...",
        len(requests_list),
        args.max_concurrency,
    )

    semaphore = (
        asyncio.Semaphore(args.max_concurrency) if args.max_concurrency else None
    )

    async def _limited_request(
        req: RequestFuncInput,
        session: aiohttp.ClientSession,
        pbar: tqdm,
    ) -> RequestFuncOutput:
        if semaphore:
            async with semaphore:
                return await send_tts_request(req, session, save_audio_dir, pbar)
        return await send_tts_request(req, session, save_audio_dir, pbar)

    pbar_obj = tqdm(total=len(requests_list), disable=args.disable_tqdm)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for req in requests_list:
            if args.request_rate != float("inf"):
                interval = np.random.exponential(1.0 / args.request_rate)
                await asyncio.sleep(interval)
            tasks.append(asyncio.create_task(_limited_request(req, session, pbar_obj)))
        outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)

    pbar_obj.close()

    metrics = calculate_metrics(outputs)
    print_summary(metrics, args)

    if args.output_dir:
        _save_results(outputs, metrics, args, base_url)


def _save_results(
    outputs: list[RequestFuncOutput],
    metrics: dict,
    args: argparse.Namespace,
    base_url: str,
) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    results = {
        "summary": metrics,
        "config": {
            "model": args.model,
            "base_url": base_url,
            "testset": args.testset,
            "no_ref_audio": args.no_ref_audio,
            "max_samples": args.max_samples,
            "max_new_tokens": args.max_new_tokens,
            "warmup": args.warmup,
            "max_concurrency": args.max_concurrency,
            "request_rate": args.request_rate,
        },
        "per_request": [
            {
                "id": o.request_id,
                "text": o.text,
                "success": o.success,
                "latency_s": round(o.latency, 4),
                "audio_duration_s": round(o.audio_duration_s, 4),
                "rtf": round(o.rtf, 4) if o.rtf < float("inf") else None,
                "prompt_tokens": o.prompt_tokens or None,
                "completion_tokens": o.completion_tokens or None,
                "tok_per_s": round(o.tok_per_s, 1) if o.tok_per_s > 0 else None,
                "error": o.error or None,
            }
            for o in outputs
        ],
    }
    json_path = os.path.join(args.output_dir, "speed_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", json_path)

    csv_path = os.path.join(args.output_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "text",
                "latency_s",
                "audio_duration_s",
                "rtf",
                "completion_tokens",
                "tok_per_s",
                "success",
                "error",
            ]
        )
        for o in outputs:
            writer.writerow(
                [
                    o.request_id,
                    o.text,
                    f"{o.latency:.4f}",
                    f"{o.audio_duration_s:.4f}",
                    f"{o.rtf:.4f}" if o.rtf < float("inf") else "",
                    o.completion_tokens or "",
                    f"{o.tok_per_s:.1f}" if o.tok_per_s > 0 else "",
                    o.success,
                    o.error or "",
                ]
            )
    logger.info("CSV saved to %s", csv_path)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark online serving for TTS models."
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
        default="default",
        help="Model name for the API request.",
    )
    parser.add_argument(
        "--testset",
        type=str,
        default="seed-tts-eval/en/meta.lst",
        help="Path to seed-tts-eval meta.lst.",
    )
    parser.add_argument(
        "--no-ref-audio",
        action="store_true",
        help="Skip ref audio/text from testset (TTS without voice cloning).",
    )
    parser.add_argument("--output-dir", type=str, default="results/tts_speed")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum concurrent requests.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Requests per second (inf = send all at once).",
    )
    parser.add_argument(
        "--save-audio", action="store_true", help="Save generated WAV files."
    )
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
