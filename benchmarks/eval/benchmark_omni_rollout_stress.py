# SPDX-License-Identifier: Apache-2.0
# note (luojiaxuan):
"""Closed-loop concurrency / rollout stress runner for Qwen3-Omni.

Reuses the exact same prompt N times across a sweep of concurrency levels so
operators can measure how same-prompt multi-rollout (and, more generally,
many-concurrent-session) traffic scales: per-level requests/s, output tokens/s,
prompt-token counts, and p50/p95/p99 latency, plus any profiler events the
server emits. This is a serving pressure / diagnostics tool (e.g. issue #760).

Run from the repo root as a module so the ``benchmarks.`` imports resolve::

    python -m benchmarks.eval.benchmark_omni_rollout_stress \\
        --base-url http://localhost:8000 \\
        --rollout-counts 1,2,4,8,16,32 \\
        --max-tokens 256 \\
        --output-dir results/rollout_stress

By default it drives a multimodal MMMU sample (``--repo-id`` / ``--sample-index``)
with audio output enabled; pass ``--text-only`` for a text-only prompt or
``--prompt-override`` to supply the text. Results are written to
``<output-dir>/rollout_stress_results.json``; pass ``--no-profile`` to skip the
server-side event profiler.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import aiohttp

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import save_json_results, wait_for_service
from benchmarks.dataset.mmmu import MMMUSample, load_mmmu_samples
from benchmarks.metrics.performance import compute_speed_metrics, print_speed_summary

logger = logging.getLogger(__name__)


def _build_base_url(args: argparse.Namespace) -> str:
    return args.base_url or f"http://{args.host}:{args.port}"


def _parse_counts(raw: str) -> list[int]:
    counts = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not counts or any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError("rollout counts must be positive integers")
    return counts


def _duplicate_prompt(sample: MMMUSample, count: int) -> list[MMMUSample]:
    return [
        replace(
            sample,
            sample_id=f"{sample.sample_id}:rollout-n{count}-{index}",
        )
        for index in range(count)
    ]


def _make_rollout_send_fn(
    *,
    model_name: str,
    api_url: str,
    rollout_group_id: str,
    max_tokens: int,
    temperature: float,
    enable_audio: bool,
    talker_max_new_tokens: int | None,
) -> Any:
    modalities = ["text", "audio"] if enable_audio else ["text"]

    async def send_fn(
        session: aiohttp.ClientSession,
        sample: MMMUSample,
    ) -> RequestResult:
        result = RequestResult(
            request_id=sample.sample_id,
            text=sample.prompt[:60],
        )
        payload: dict[str, Any] = {
            "model": model_name,
            "request_id": sample.sample_id,
            "messages": [{"role": "user", "content": sample.prompt}],
            "images": list(sample.image_data_uris),
            "modalities": modalities,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "user": rollout_group_id,
        }
        if enable_audio:
            payload["audio"] = {"format": "wav"}
        if talker_max_new_tokens is not None:
            payload["talker_max_new_tokens"] = talker_max_new_tokens

        start_time = time.perf_counter()
        try:
            async with session.post(api_url, json=payload) as response:
                body = await response.json()
                if response.status >= 400:
                    result.error = body.get("detail", str(body))
                    return result

            message = body.get("choices", [{}])[0].get("message", {})
            result.text = message.get("content", "") or ""
            if enable_audio:
                audio_obj = message.get("audio")
                audio_b64 = (audio_obj or {}).get("data", "")
                if not audio_b64:
                    result.error = "No audio data in response"
                    return result
                # Decode once so transport/base64 corruption is surfaced in the
                # stress result without forcing a WAV write for every rollout.
                base64.b64decode(audio_b64)

            result.is_success = True
            usage = body.get("usage", {})
            result.prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            result.completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            result.engine_time_s = time.perf_counter() - start_time
            if result.completion_tokens > 0 and result.engine_time_s > 0:
                result.tok_per_s = result.completion_tokens / result.engine_time_s
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            result.error = str(exc)
        finally:
            result.latency_s = time.perf_counter() - start_time
        return result

    return send_fn


async def _get_json(session: aiohttp.ClientSession, url: str) -> dict[str, Any]:
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.json()


async def _post_json(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    async with session.post(url, json=payload) as response:
        body = await response.json()
        response.raise_for_status()
        return body


async def _start_request_profile(
    session: aiohttp.ClientSession,
    base_url: str,
    *,
    run_id: str,
    event_dir: str,
) -> dict[str, Any]:
    return await _post_json(
        session,
        f"{base_url}/start_request_profile",
        {"run_id": run_id, "event_dir": event_dir},
    )


async def _stop_request_profile(
    session: aiohttp.ClientSession,
    base_url: str,
    *,
    run_id: str,
) -> dict[str, Any]:
    return await _post_json(
        session,
        f"{base_url}/stop_request_profile",
        {"run_id": run_id},
    )


async def run_rollout_stress(args: argparse.Namespace) -> dict[str, Any]:
    base_url = _build_base_url(args)
    api_url = f"{base_url}/v1/chat/completions"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_mmmu_samples(
        args.sample_index + 1,
        repo_id=args.repo_id,
        instruction_override=args.prompt_override,
    )
    if len(samples) <= args.sample_index:
        raise ValueError(
            f"sample_index={args.sample_index} unavailable; loaded {len(samples)}"
        )
    base_sample = samples[args.sample_index]
    rollout_group_id = args.rollout_group_id or f"rollout:{base_sample.sample_id}"
    run_id = args.profile_run_id or f"rollout-stress-{int(time.time())}"
    event_dir = str(Path(args.profile_event_dir or (output_dir / "events")).resolve())

    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        health_before = await _get_json(session, f"{base_url}/health")
        profile_info = None
        if not args.no_profile:
            profile_info = await _start_request_profile(
                session,
                base_url,
                run_id=run_id,
                event_dir=event_dir,
            )

        results_by_count: list[dict[str, Any]] = []
        try:
            for count in args.rollout_counts:
                duplicated = _duplicate_prompt(base_sample, count)
                send_fn = _make_rollout_send_fn(
                    model_name=args.model,
                    api_url=api_url,
                    rollout_group_id=rollout_group_id,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    enable_audio=args.enable_audio,
                    talker_max_new_tokens=args.talker_max_new_tokens,
                )
                runner = BenchmarkRunner(
                    RunConfig(
                        max_concurrency=count,
                        request_rate=float("inf"),
                        warmup=0,
                        disable_tqdm=args.disable_tqdm,
                        timeout_s=args.timeout_s,
                    )
                )
                request_results = await runner.run(duplicated, send_fn)
                speed = compute_speed_metrics(
                    request_results,
                    wall_clock_s=runner.wall_clock_s,
                )
                failed = [result for result in request_results if not result.is_success]
                print_speed_summary(
                    speed,
                    args.model,
                    count,
                    title=f"Qwen3-Omni Rollout Stress N={count}",
                )
                print(f"  Failed requests: {len(failed)}/{len(request_results)}")
                results_by_count.append(
                    {
                        "rollout_count": count,
                        "wall_clock_s": runner.wall_clock_s,
                        "success": len(request_results) - len(failed),
                        "failed": len(failed),
                        "speed": speed,
                        "requests": [asdict(result) for result in request_results],
                    }
                )
        finally:
            if not args.no_profile:
                await _stop_request_profile(session, base_url, run_id=run_id)

        health_after = await _get_json(session, f"{base_url}/health")

    output = {
        "config": {
            "base_url": base_url,
            "model": args.model,
            "repo_id": args.repo_id,
            "sample_id": base_sample.sample_id,
            "rollout_group_id": rollout_group_id,
            "rollout_counts": args.rollout_counts,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "enable_audio": args.enable_audio,
            "talker_max_new_tokens": args.talker_max_new_tokens,
            "profile_run_id": run_id,
            "profile_event_dir": event_dir if not args.no_profile else None,
        },
        "profile": profile_info,
        "health_before": health_before,
        "health_after": health_after,
        "results": results_by_count,
    }
    save_json_results(output, str(output_dir), "rollout_stress_results.json")
    return output


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Stress Qwen3-Omni with same-prompt multi-rollout traffic."
    )
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="qwen3-omni")
    parser.add_argument("--repo-id", type=str, default="zhaochenyang20/mmmu-ci-50")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--prompt-override", type=str, default=None)
    parser.add_argument(
        "--rollout-counts",
        type=_parse_counts,
        default=[1, 2, 4, 8, 16],
        help="Comma-separated rollout counts, e.g. 1,2,4,8,16.",
    )
    parser.add_argument("--rollout-group-id", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--text-only", dest="enable_audio", action="store_false")
    parser.set_defaults(enable_audio=True)
    parser.add_argument("--talker-max-new-tokens", type=int, default=None)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--output-dir", type=str, default="results/rollout_stress")
    parser.add_argument("--profile-run-id", type=str, default=None)
    parser.add_argument("--profile-event-dir", type=str, default=None)
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

    base_url = _build_base_url(args)
    wait_for_service(base_url)
    asyncio.run(run_rollout_stress(args))


if __name__ == "__main__":
    main()
