from __future__ import annotations

import asyncio
import logging
import math
import os
import random
import time

import aiohttp
from tqdm.asyncio import tqdm

from benchmarks.adapters.request_family import get_request_family_adapter
from benchmarks.core.metrics import compute_run_summary
from benchmarks.core.reporters import print_summary, save_results
from benchmarks.core.types import BenchmarkResults, BenchmarkRunConfig, PreparedRequest
from benchmarks.profiles.registry import PROFILE_REGISTRY
from benchmarks.profiles.resolver import resolve_profile_selection

logger = logging.getLogger(__name__)


async def run_from_config(
    run_config: BenchmarkRunConfig,
) -> BenchmarkResults:
    _validate_run_config(run_config)
    await wait_for_service(run_config.base_url)
    selection = resolve_profile_selection(
        run_config=run_config,
        registry=PROFILE_REGISTRY,
    )
    profile = selection.model_profile
    case_spec = selection.case_spec

    if case_spec.requires_dataset_path and run_config.dataset_path is None:
        raise ValueError(
            f"Case '{case_spec.case_id}' for profile '{selection.model_profile_name}' "
            "requires --dataset"
        )
    if run_config.stream and not case_spec.supports_stream:
        raise ValueError(
            f"Case '{case_spec.case_id}' for profile '{selection.model_profile_name}' "
            "does not support streaming"
        )

    samples = profile.dataset_loader.load(
        dataset_path=run_config.dataset_path,
        max_samples=run_config.max_samples,
    )
    if not samples:
        raise ValueError(
            f"Dataset loader '{profile.dataset_loader.name}' returned no samples"
        )
    requests = [
        profile.model_adapter.build_request(
            sample=sample,
            case_spec=case_spec,
            run_config=run_config,
            served_model_name=selection.served_model_name,
        )
        for sample in samples
    ]
    logger.info(
        "Prepared %d requests for profile=%s case=%s",
        len(requests),
        selection.model_profile_name,
        case_spec.case_id,
    )

    request_family = get_request_family_adapter(selection.request_family)

    async with aiohttp.ClientSession() as session:
        if run_config.warmup > 0 and requests:
            await _run_warmup(
                requests=requests,
                warmup_count=run_config.warmup,
                base_url=run_config.base_url,
                request_family=request_family,
                session=session,
            )

        started_at = time.perf_counter()
        per_request = await _run_requests(
            requests=requests,
            base_url=run_config.base_url,
            request_family=request_family,
            session=session,
            max_concurrency=run_config.max_concurrency,
            request_rate=run_config.request_rate,
            disable_tqdm=run_config.disable_tqdm,
        )
        run_wall_time_s = time.perf_counter() - started_at

    if run_config.save_audio and run_config.output_dir:
        _save_audio_files(per_request, run_config.output_dir)

    summary = compute_run_summary(
        results=per_request,
        run_wall_time_s=run_wall_time_s,
        model_profile=selection.model_profile_name,
        case_id=case_spec.case_id,
        scenario_id=case_spec.scenario_id,
        request_family=request_family.family_name,
    )
    results = BenchmarkResults(
        summary=summary,
        config={
            "model_profile": selection.model_profile_name,
            "served_model_name": selection.served_model_name,
            "case": case_spec.case_id,
            "scenario": case_spec.scenario_id,
            "request_family": request_family.family_name,
            "dataset_loader": selection.dataset_loader_name,
            "model_adapter": selection.model_adapter_name,
            "base_url": run_config.base_url,
            "model": run_config.model,
            "dataset": run_config.dataset_path,
            "stream": run_config.stream,
            "max_samples": run_config.max_samples,
            "max_output_tokens": run_config.max_output_tokens
            or case_spec.default_max_output_tokens,
            "temperature": run_config.temperature,
            "top_p": run_config.top_p,
            "top_k": run_config.top_k,
            "repetition_penalty": run_config.repetition_penalty,
            "warmup": run_config.warmup,
            "max_concurrency": run_config.max_concurrency,
            "request_rate": (
                "inf" if math.isinf(run_config.request_rate) else run_config.request_rate
            ),
            "save_audio": run_config.save_audio,
        },
        per_request=per_request,
    )
    print_summary(results)
    save_results(results, run_config.output_dir)
    return results


def _validate_run_config(run_config: BenchmarkRunConfig) -> None:
    if not run_config.model_profile:
        raise ValueError("--model-profile is required")
    if run_config.max_samples is not None and run_config.max_samples <= 0:
        raise ValueError("--max-samples must be greater than 0 when provided")
    if run_config.warmup < 0:
        raise ValueError("--warmup must be greater than or equal to 0")
    if run_config.max_concurrency <= 0:
        raise ValueError("--max-concurrency must be greater than 0")
    if not math.isinf(run_config.request_rate) and run_config.request_rate <= 0:
        raise ValueError("--request-rate must be positive or inf")


async def wait_for_service(
    base_url: str,
    timeout_s: int = 1200,
) -> None:
    logger.info("Waiting for service at %s ...", base_url)
    started_at = time.perf_counter()
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        body = await resp.json()
                        if body.get("status") == "healthy":
                            logger.info("Service is ready.")
                            return
        except (aiohttp.ClientError, Exception):
            pass
        if time.perf_counter() - started_at > timeout_s:
            raise TimeoutError(f"Service at {base_url} not ready within {timeout_s}s")
        await asyncio.sleep(1)


async def _run_warmup(
    *,
    requests: list[PreparedRequest],
    warmup_count: int,
    base_url: str,
    request_family,
    session: aiohttp.ClientSession,
) -> None:
    logger.info("Warmup (%d requests)...", warmup_count)
    for index, request in enumerate(requests[:warmup_count], start=1):
        result = await request_family.execute(
            session=session,
            base_url=base_url,
            request=request,
        )
        status = "ok" if result.success else result.error
        logger.info("  warmup %d/%d: %s", index, warmup_count, status)


async def _run_requests(
    *,
    requests: list[PreparedRequest],
    base_url: str,
    request_family,
    session: aiohttp.ClientSession,
    max_concurrency: int,
    request_rate: float,
    disable_tqdm: bool = False,
) -> list:
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    pbar = tqdm(total=len(requests), disable=disable_tqdm)

    async def _execute(request: PreparedRequest):
        if semaphore is not None:
            async with semaphore:
                result = await request_family.execute(
                    session=session,
                    base_url=base_url,
                    request=request,
                )
        else:
            result = await request_family.execute(
                session=session,
                base_url=base_url,
                request=request,
            )
        pbar.update(1)
        return result

    tasks = []
    for request in requests:
        if not math.isinf(request_rate):
            await asyncio.sleep(random.expovariate(request_rate))
        tasks.append(asyncio.create_task(_execute(request)))
    results = await asyncio.gather(*tasks)
    pbar.close()
    return results


def _save_audio_files(per_request: list, output_dir: str) -> None:
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    for result in per_request:
        if result.audio_bytes:
            audio_path = os.path.join(audio_dir, f"{result.request_id}.wav")
            with open(audio_path, "wb") as f:
                f.write(result.audio_bytes)
    logger.info("Audio files saved to %s", audio_dir)
