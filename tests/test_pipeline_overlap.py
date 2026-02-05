# SPDX-License-Identifier: Apache-2.0
"""Zero-overlap pipeline scheduling benchmark.

Compares Serial (CPU->GPU->CPU->GPU) vs Zero-Overlap (CPU[N+1] || GPU[N]).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logging.getLogger("sglang_omni").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkArgs:
    model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    dtype: str = "bfloat16"
    prompt: str = "What do you see in the video?"
    num_requests: int = 2
    video_fps: float = 2.0
    image_device: str = "cuda:0"
    max_pending: int = 2
    serial_only: bool = False
    overlap_only: bool = False


@dataclass
class BenchmarkResult:
    mode: str
    num_requests: int
    total_time: float
    latencies: list[float] = field(default_factory=list)


def get_video_path() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local = os.path.join(os.path.dirname(script_dir), "tests/data/draw.mp4")
    return local


def create_request(
    video_path: str, prompt: str, idx: int, fps: float = 2.0
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": f"{prompt} [req{idx}-{uuid.uuid4().hex[:6]}]"}
        ],
        "videos": [video_path],
        "video_fps": fps,
    }


async def run_encoder_benchmark(
    args: BenchmarkArgs, enable_overlap: bool
) -> BenchmarkResult:
    """Benchmark encoder stage only (frontend + image encoder).

    This directly tests the IntraStageOverlapExecutor without thinker overhead.
    Submits all requests concurrently to maximize overlap opportunity.
    """
    from sglang_omni.engines.omni import create_encoder_engine
    from sglang_omni.executors import EngineExecutor, FrontendExecutor
    from sglang_omni.executors.intra_stage_overlap_executor import (
        IntraStageOverlapExecutor,
    )
    from sglang_omni.models.qwen3_omni.components.frontend import Qwen3OmniFrontend
    from sglang_omni.models.qwen3_omni.components.image_encoder import (
        Qwen3OmniImageEncoder,
    )
    from sglang_omni.models.qwen3_omni.pipeline.engine_io import build_encoder_request
    from sglang_omni.models.qwen3_omni.pipeline.next_stage import IMAGE_STAGE
    from sglang_omni.models.qwen3_omni.pipeline.state_io import load_state
    from sglang_omni.proto import OmniRequest, StagePayload

    video_path = get_video_path()
    mode = "Zero-Overlap" if enable_overlap else "Serial"

    # Create frontend
    frontend = Qwen3OmniFrontend(model_id=args.model_id)

    def frontend_fn(payload: StagePayload) -> StagePayload:
        return frontend(payload)

    # Create image encoder
    image_model = Qwen3OmniImageEncoder(
        model_id=args.model_id, device=args.image_device, dtype=args.dtype
    )

    def request_builder(payload: StagePayload):
        state = load_state(payload)
        return build_encoder_request(state, stage_name=IMAGE_STAGE)

    def result_builder(payload: StagePayload, result: Any) -> StagePayload:
        # Just return the payload, we don't need to process the result
        return payload

    engine = create_encoder_engine(image_model, device=args.image_device)
    gpu_executor = EngineExecutor(
        engine=engine, request_builder=request_builder, result_builder=result_builder
    )

    if enable_overlap:
        # Use IntraStageOverlapExecutor for CPU/GPU overlap
        cpu_executor = FrontendExecutor(
            frontend_fn, use_thread_pool=True, max_workers=4
        )
        executor = IntraStageOverlapExecutor(
            cpu_executor=cpu_executor,
            gpu_executor=gpu_executor,
            max_pending=args.max_pending,
        )
    else:
        # Serial: frontend then GPU
        cpu_executor = FrontendExecutor(frontend_fn, use_thread_pool=False)
        # We'll run them manually in sequence
        executor = None

    # Prepare requests
    requests = [
        create_request(video_path, args.prompt, i, args.video_fps)
        for i in range(args.num_requests)
    ]
    payloads = [
        StagePayload(
            request_id=f"req-{mode}-{i}-{uuid.uuid4().hex[:6]}",
            request=OmniRequest(inputs=req, params={}),
            data={},
        )
        for i, req in enumerate(requests)
    ]

    logger.info(f"[{mode}] Running {args.num_requests} requests concurrently...")
    start = time.perf_counter()

    if enable_overlap:
        # Concurrent submission with overlap executor
        await executor.start()

        # Submit all requests concurrently
        async def submit_and_wait(payload: StagePayload, idx: int) -> float:
            t0 = time.perf_counter()
            await executor.add_request(payload)
            return t0

        # Submit all
        submit_times = await asyncio.gather(
            *[submit_and_wait(p, i) for i, p in enumerate(payloads)]
        )

        # Wait for all results
        latencies = []
        for i in range(len(payloads)):
            await executor.get_result()
            latencies.append(time.perf_counter() - submit_times[i])

        await executor.stop()
    else:
        # Serial execution: one request at a time
        await cpu_executor.start()
        await gpu_executor.start()

        latencies = []
        for i, payload in enumerate(payloads):
            t0 = time.perf_counter()
            await cpu_executor.add_request(payload)
            cpu_result = await cpu_executor.get_result()
            await gpu_executor.add_request(cpu_result)
            await gpu_executor.get_result()
            latencies.append(time.perf_counter() - t0)
            if (i + 1) % 5 == 0:
                logger.info(f"  Progress: {i+1}/{len(payloads)}")

        await gpu_executor.stop()
        await cpu_executor.stop()

    total = time.perf_counter() - start
    logger.info(f"[{mode}] Complete: {total:.3f}s")

    return BenchmarkResult(
        mode=mode, num_requests=len(payloads), total_time=total, latencies=latencies
    )


async def _run_benchmark_pair(
    args: BenchmarkArgs,
) -> tuple[BenchmarkResult | None, BenchmarkResult | None]:
    baseline = (
        None
        if args.overlap_only
        else await run_encoder_benchmark(args, enable_overlap=False)
    )
    optimized = (
        None
        if args.serial_only
        else await run_encoder_benchmark(args, enable_overlap=True)
    )
    return baseline, optimized


def test_encoder_overlap_real_smoke():
    args = BenchmarkArgs()
    baseline, optimized = asyncio.run(_run_benchmark_pair(args))
    if baseline:
        assert baseline.total_time > 0 and baseline.num_requests > 0
    if optimized:
        assert optimized.total_time > 0 and optimized.num_requests > 0
