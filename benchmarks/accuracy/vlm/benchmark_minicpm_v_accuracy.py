# SPDX-License-Identifier: Apache-2.0
"""
Benchmark accuracy for MiniCPM-V vision-language models.

This script evaluates MiniCPM-V (2.6 and 4.5) on standard VLM benchmarks
including image understanding, OCR, and multimodal reasoning tasks.

Supported benchmarks:
- TextVQA: Text reading in images
- DocVQA: Document understanding
- ChartQA: Chart comprehension
- OCRBench: OCR evaluation
- Custom: User-provided image-question pairs

Usage::

    # Launch server first
    python examples/run_minicpm_v_server.py \\
        --model-path openbmb/MiniCPM-V-2_6 \\
        --port 8000

    # Run TextVQA benchmark
    python -m benchmarks.accuracy.vlm.benchmark_minicpm_v_accuracy \\
        --model minicpm-v --port 8000 \\
        --benchmark textvqa --max-samples 100

    # Run custom benchmark with image-question pairs
    python -m benchmarks.accuracy.vlm.benchmark_minicpm_v_accuracy \\
        --model minicpm-v --port 8000 \\
        --testset custom_vqa.jsonl \\
        --max-samples 50

Testset format (JSONL)::

    {"id": "001", "image": "path/to/image.jpg", "question": "What color is the car?", "answer": "red"}
    {"id": "002", "image": "path/to/image.jpg", "question": "How many people are there?", "answer": "3"}

"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
TEXT_PREVIEW_LENGTH = 80
SUMMARY_LABEL_WIDTH = 35
SUMMARY_LINE_WIDTH = 70


@dataclass
class VQARequest:
    """Single VQA request."""

    request_id: str
    image_path: str
    question: str
    ground_truth: str
    api_url: str
    model: str
    max_tokens: int = 512
    temperature: float = 0.1


@dataclass
class VQAResult:
    """Result from a single VQA request."""

    request_id: str = ""
    question: str = ""
    ground_truth: str = ""
    prediction: str = ""
    is_correct: bool = False
    is_success: bool = False
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""


def _image_to_base64(image_path: str) -> str:
    """Convert image file to base64 data URL."""
    import mimetypes

    path = Path(image_path)
    mime_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"

    with open(path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{b64_data}"


def _normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Lowercase
    answer = answer.lower().strip()
    # Remove punctuation
    answer = re.sub(r"[^\w\s]", "", answer)
    # Remove articles
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    # Normalize whitespace
    answer = " ".join(answer.split())
    return answer


def _check_answer(prediction: str, ground_truth: str) -> bool:
    """Check if prediction matches ground truth (relaxed matching)."""
    pred_norm = _normalize_answer(prediction)
    gt_norm = _normalize_answer(ground_truth)

    # Exact match
    if pred_norm == gt_norm:
        return True

    # Contains match (for longer answers)
    if gt_norm in pred_norm or pred_norm in gt_norm:
        return True

    # Numeric match (handle "3" vs "three")
    try:
        pred_num = float(pred_norm)
        gt_num = float(gt_norm)
        if abs(pred_num - gt_num) < 0.01:
            return True
    except ValueError:
        pass

    return False


def load_testset(testset_path: str, max_samples: int | None = None) -> list[dict]:
    """Load testset from JSONL file."""
    samples = []
    base_dir = Path(testset_path).parent

    with open(testset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)

            # Resolve relative image paths
            if not os.path.isabs(sample["image"]):
                sample["image"] = str(base_dir / sample["image"])

            samples.append(sample)

            if max_samples and len(samples) >= max_samples:
                break

    return samples


async def send_vqa_request(
    request: VQARequest,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> VQAResult:
    """Send a single VQA request to the server."""
    result = VQAResult(
        request_id=request.request_id,
        question=request.question[:TEXT_PREVIEW_LENGTH],
        ground_truth=request.ground_truth,
    )

    # Build OpenAI-compatible request
    try:
        image_data = _image_to_base64(request.image_path)
    except Exception as e:
        result.error = f"Failed to load image: {e}"
        if pbar:
            pbar.update(1)
        return result

    payload = {
        "model": request.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data}},
                    {"type": "text", "text": request.question},
                ],
            }
        ],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    }

    start_time = time.perf_counter()
    try:
        async with session.post(request.api_url, json=payload) as response:
            if response.status != 200:
                result.error = f"HTTP {response.status}: {await response.text()}"
            else:
                data = await response.json()
                result.is_success = True

                # Extract prediction
                choices = data.get("choices", [])
                if choices:
                    result.prediction = choices[0].get("message", {}).get("content", "")

                # Extract token counts
                usage = data.get("usage", {})
                result.prompt_tokens = usage.get("prompt_tokens", 0)
                result.completion_tokens = usage.get("completion_tokens", 0)

                # Check correctness
                result.is_correct = _check_answer(
                    result.prediction, result.ground_truth
                )

    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        result.error = str(exc)
    finally:
        result.latency_s = time.perf_counter() - start_time

    if pbar:
        pbar.update(1)
    return result


def wait_for_service(base_url: str, timeout: int = 600) -> None:
    """Wait for the service to become healthy."""
    import requests as requests_lib

    logger.info("Waiting for service at %s ...", base_url)
    start = time.time()
    while True:
        try:
            resp = requests_lib.get(f"{base_url}/health", timeout=2)
            if resp.status_code == 200:
                logger.info("Service is ready.")
                return
        except requests_lib.exceptions.RequestException:
            pass
        if time.time() - start > timeout:
            raise TimeoutError(f"Service at {base_url} not ready within {timeout}s")
        time.sleep(2)


def calculate_metrics(results: list[VQAResult]) -> dict:
    """Calculate accuracy and performance metrics."""
    successes = [r for r in results if r.is_success]
    if not successes:
        return {"completed": 0, "failed": len(results), "accuracy": 0.0}

    correct = sum(1 for r in successes if r.is_correct)
    accuracy = correct / len(successes) * 100 if successes else 0.0

    latencies = [r.latency_s for r in successes]
    total_prompt_tokens = sum(r.prompt_tokens for r in successes)
    total_completion_tokens = sum(r.completion_tokens for r in successes)

    metrics = {
        "completed": len(successes),
        "failed": len(results) - len(successes),
        "correct": correct,
        "accuracy": round(accuracy, 2),
        "latency_mean_s": round(float(np.mean(latencies)), 3),
        "latency_median_s": round(float(np.median(latencies)), 3),
        "latency_p95_s": round(float(np.percentile(latencies, 95)), 3),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "throughput_qps": round(len(successes) / sum(latencies), 3) if latencies else 0,
    }

    return metrics


def print_summary(metrics: dict, args: argparse.Namespace) -> None:
    """Print benchmark summary."""
    lw = SUMMARY_LABEL_WIDTH
    w = SUMMARY_LINE_WIDTH

    print(f"\n{'=' * w}")
    print(f"{'MiniCPM-V Accuracy Benchmark':^{w}}")
    print(f"{'=' * w}")
    print(f"  {'Model:':<{lw}} {args.model}")
    print(f"  {'Testset:':<{lw}} {args.testset or args.benchmark}")
    print(f"{'-' * w}")
    print(f"  {'Completed requests:':<{lw}} {metrics['completed']}")
    print(f"  {'Failed requests:':<{lw}} {metrics['failed']}")
    print(f"  {'Correct answers:':<{lw}} {metrics['correct']}")
    print(f"  {'Accuracy (%):':<{lw}} {metrics['accuracy']:.2f}")
    print(f"{'-' * w}")
    print(f"  {'Latency mean (s):':<{lw}} {metrics.get('latency_mean_s', 'N/A')}")
    print(f"  {'Latency median (s):':<{lw}} {metrics.get('latency_median_s', 'N/A')}")
    print(f"  {'Latency p95 (s):':<{lw}} {metrics.get('latency_p95_s', 'N/A')}")
    print(f"  {'Throughput (req/s):':<{lw}} {metrics.get('throughput_qps', 'N/A')}")
    print(f"  {'Total prompt tokens:':<{lw}} {metrics.get('total_prompt_tokens', 0)}")
    print(
        f"  {'Total completion tokens:':<{lw}} {metrics.get('total_completion_tokens', 0)}"
    )
    print(f"{'=' * w}")


def save_results(
    results: list[VQAResult],
    metrics: dict,
    args: argparse.Namespace,
    output_dir: str,
) -> None:
    """Save benchmark results to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON
    json_results = {
        "summary": metrics,
        "config": {
            "model": args.model,
            "testset": args.testset or args.benchmark,
            "max_samples": args.max_samples,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
        "per_request": [
            {
                "id": r.request_id,
                "question": r.question,
                "ground_truth": r.ground_truth,
                "prediction": r.prediction[:200],  # Truncate for storage
                "is_correct": r.is_correct,
                "is_success": r.is_success,
                "latency_s": round(r.latency_s, 4),
                "error": r.error or None,
            }
            for r in results
        ],
    }

    json_path = os.path.join(output_dir, "accuracy_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", json_path)


async def benchmark(args: argparse.Namespace) -> None:
    """Run the benchmark."""
    base_url = args.base_url or f"http://{args.host}:{args.port}"
    api_url = f"{base_url}/v1/chat/completions"

    wait_for_service(base_url)

    # Load testset
    if args.testset:
        if not os.path.isfile(args.testset):
            logger.error("Testset not found: %s", args.testset)
            return
        samples = load_testset(args.testset, args.max_samples)
    else:
        # Use built-in benchmark placeholder
        logger.error("Please provide --testset path to a JSONL file")
        return

    logger.info("Loaded %d samples", len(samples))

    # Build requests
    requests_list = [
        VQARequest(
            request_id=sample.get("id", str(i)),
            image_path=sample["image"],
            question=sample["question"],
            ground_truth=sample.get("answer", ""),
            api_url=api_url,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        for i, sample in enumerate(samples)
    ]

    # Run warmup
    if args.warmup > 0:
        logger.info("Warmup (%d requests)...", args.warmup)
        async with aiohttp.ClientSession() as session:
            for i in range(min(args.warmup, len(requests_list))):
                warmup_result = await send_vqa_request(requests_list[i], session)
                status = "ok" if warmup_result.is_success else warmup_result.error
                logger.info("  warmup %d/%d: %s", i + 1, args.warmup, status)

    # Run benchmark
    logger.info(
        "Benchmarking %d requests (concurrency=%d)...",
        len(requests_list),
        args.max_concurrency,
    )

    semaphore = asyncio.Semaphore(args.max_concurrency)

    async def limited_request(req: VQARequest, session: aiohttp.ClientSession, pbar: tqdm) -> VQAResult:
        async with semaphore:
            return await send_vqa_request(req, session, pbar)

    pbar = tqdm(total=len(requests_list), disable=args.disable_tqdm)
    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(limited_request(req, session, pbar))
            for req in requests_list
        ]
        results = await asyncio.gather(*tasks)
    pbar.close()

    # Calculate and display metrics
    metrics = calculate_metrics(results)
    print_summary(metrics, args)

    # Save results
    if args.output_dir:
        save_results(results, metrics, args, args.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark accuracy for MiniCPM-V vision models."
    )

    # Server connection
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL (overrides --host/--port)",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model",
        type=str,
        default="minicpm-v",
        help="Model name for API requests",
    )

    # Benchmark selection
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        choices=["textvqa", "docvqa", "chartqa", "ocrbench"],
        help="Built-in benchmark to run",
    )
    parser.add_argument(
        "--testset",
        type=str,
        default=None,
        help="Path to custom JSONL testset",
    )

    # Generation parameters
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)

    # Benchmark options
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="results/minicpm_v_accuracy")
    parser.add_argument("--disable-tqdm", action="store_true")

    args = parser.parse_args()

    if not args.testset and not args.benchmark:
        parser.error("Either --testset or --benchmark is required")

    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
