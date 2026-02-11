# SPDX-License-Identifier: Apache-2.0
"""Three-stage pipeline demo with dynamic Relay selection (Nixl/Shm/Nccl)."""

import argparse
import asyncio
import logging
import multiprocessing as mp
import time
from typing import Any, List

from sglang_omni import Coordinator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Endpoints
STAGE1_ENDPOINT = "tcp://127.0.0.1:16001"
STAGE2_ENDPOINT = "tcp://127.0.0.1:16002"
STAGE3_ENDPOINT = "tcp://127.0.0.1:16003"
COORDINATOR_ENDPOINT = "tcp://127.0.0.1:16000"
ABORT_ENDPOINT = "tcp://127.0.0.1:16099"

# All endpoints for routing
ENDPOINTS = {
    "preprocessor": STAGE1_ENDPOINT,
    "encoder": STAGE2_ENDPOINT,
    "decoder": STAGE3_ENDPOINT,
}

# --- NCCL Topology Constants ---
WORLD_SIZE = 3
RANK_PREPROCESSOR = 0
RANK_ENCODER = 1
RANK_DECODER = 2


def stage1_get_next(request_id: str, output: Any) -> str | None:
    return "encoder"


def stage2_get_next(request_id: str, output: Any) -> str | None:
    # Extract value from payload.data (output is always StagePayload with dict data)
    data = output.data
    value = data.get("value")

    if isinstance(value, (int, float)) and value < 0:
        logger.info("Encoder: output=%s is negative, early exit!", value)
        return None
    return "decoder"


def stage3_get_next(request_id: str, output: Any) -> str | None:
    return None


def run_stage(
    name: str,
    endpoint: str,
    transform,
    delay: float,
    get_next,
    relay_type: str = "shm",
    gpu_id: int | None = None,
    rank: int = 0,
    world_size: int = 1,
    send_to_ranks: List[int] = [],
    recv_from_ranks: List[int] = [],
):
    """Generic stage runner with unified Relay configuration."""
    # Move imports here to avoid multiprocessing pickling issues
    import asyncio
    import logging

    from sglang_omni import Stage, Worker
    from sglang_omni.executors import PreprocessingExecutor
    from sglang_omni.proto import StagePayload

    # Configure logging for child process
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [%(levelname)s] {name}: %(message)s",
    )

    def processor(payload: StagePayload) -> StagePayload:
        if delay > 0:
            time.sleep(delay)

        value = payload.data
        if isinstance(value, dict):
            value = value.get("value", value.get("raw_inputs"))

        result = transform(value)
        if isinstance(payload.data, dict):
            payload.data["value"] = result
        else:
            payload.data = {"raw_inputs": value, "value": result}

        return payload

    engine = PreprocessingExecutor(processor)
    worker = Worker(engine)

    # --- Build Unified Relay Config ---
    relay_config = {
        "relay_type": relay_type,
        "worker_id": f"worker_{name}",
        "slot_size_mb": 64,
        "credits": 4,
        "gpu_id": gpu_id,
        # NCCL topology parameters (safe to pass even if relay_type is not nccl)
        "rank": rank,
        "world_size": world_size,
        "send_to_ranks": send_to_ranks,
        "recv_from_ranks": recv_from_ranks,
    }

    logger.info(
        "Stage %s initializing with %s (gpu_id=%s, rank=%s)",
        name,
        relay_type.upper(),
        gpu_id,
        rank if relay_type == "nccl" else "N/A",
    )

    # Initialize Stage (Stage will create the correct Relay based on config)
    stage = Stage(
        name=name,
        get_next=get_next,
        recv_endpoint=endpoint,
        coordinator_endpoint=COORDINATOR_ENDPOINT,
        abort_endpoint=ABORT_ENDPOINT,
        endpoints=ENDPOINTS,
        relay_config=relay_config,
    )
    stage.add_worker(worker)

    asyncio.run(stage.run())


# --- Stage Runners Wrappers ---


def run_preprocessor(relay_type: str, gpu_ids: list[int]):
    gpu = gpu_ids[0] if gpu_ids else None

    # NCCL topology: Rank 0 sends to Encoder (Rank 1), doesn't receive via NCCL
    send_to = [RANK_ENCODER]
    recv_from = []

    run_stage(
        name="preprocessor",
        endpoint=STAGE1_ENDPOINT,
        transform=lambda x: x * 10 - 5,
        delay=0.05,
        get_next=stage1_get_next,
        relay_type=relay_type,
        gpu_id=gpu,
        # NCCL params
        rank=RANK_PREPROCESSOR,
        world_size=WORLD_SIZE,
        send_to_ranks=send_to,
        recv_from_ranks=recv_from,
    )


def run_encoder(relay_type: str, gpu_ids: list[int]):
    gpu = gpu_ids[1] if len(gpu_ids) > 1 else (gpu_ids[0] if gpu_ids else None)

    # NCCL topology: Rank 1 receives from Preprocessor (0), sends to Decoder (2)
    send_to = [RANK_DECODER]
    recv_from = [RANK_PREPROCESSOR]

    run_stage(
        name="encoder",
        endpoint=STAGE2_ENDPOINT,
        transform=lambda x: x * x if x >= 0 else x,
        delay=0.1,
        get_next=stage2_get_next,
        relay_type=relay_type,
        gpu_id=gpu,
        # NCCL params
        rank=RANK_ENCODER,
        world_size=WORLD_SIZE,
        send_to_ranks=send_to,
        recv_from_ranks=recv_from,
    )


def run_decoder(relay_type: str, gpu_ids: list[int]):
    gpu = gpu_ids[2] if len(gpu_ids) > 2 else (gpu_ids[0] if gpu_ids else None)

    # NCCL topology: Rank 2 receives from Encoder (1), doesn't send via NCCL (returns to Coordinator)
    send_to = []
    recv_from = [RANK_ENCODER]

    run_stage(
        name="decoder",
        endpoint=STAGE3_ENDPOINT,
        transform=lambda x: x + 1000,
        delay=0.1,
        get_next=stage3_get_next,
        relay_type=relay_type,
        gpu_id=gpu,
        # NCCL params
        rank=RANK_DECODER,
        world_size=WORLD_SIZE,
        send_to_ranks=send_to,
        recv_from_ranks=recv_from,
    )


# --- Coordinator & Test Logic ---


async def run_coordinator_main(relay_type: str):
    """Run the coordinator and test the pipeline."""
    coordinator = Coordinator(
        completion_endpoint=COORDINATOR_ENDPOINT,
        abort_endpoint=ABORT_ENDPOINT,
        entry_stage="preprocessor",
    )

    coordinator.register_stage("preprocessor", STAGE1_ENDPOINT)
    coordinator.register_stage("encoder", STAGE2_ENDPOINT)
    coordinator.register_stage("decoder", STAGE3_ENDPOINT)

    await coordinator.start()
    completion_task = asyncio.create_task(coordinator.run_completion_loop())

    try:
        # Give time for stages to complete NCCL warmup
        wait_time = 4.0 if relay_type == "nccl" else 2.0
        logger.info(f"Waiting {wait_time}s for stages to initialize and warm up...")
        await asyncio.sleep(wait_time)

        logger.info("=" * 60)
        logger.info(f"Running Tests with Relay: {relay_type.upper()}")
        logger.info("=" * 60)

        # Test 1: Normal Flow
        input_val = 5
        # 5 -> 45 -> 2025 -> 3025
        expected = 3025
        result = await coordinator.submit("req-1", input_val)
        # Extract value from result dict
        result_value = result.get("value")
        assert result_value == expected
        logger.info(f"Test 1 Passed: Input {input_val} -> Output {result_value}")

        # Test 2: Early Exit
        input_val = 0
        # 0 -> -5 -> -5 (Early Exit)
        expected = -5
        result = await coordinator.submit("req-2", input_val)
        # Extract value from result dict
        result_value = result.get("value")
        assert result_value == expected
        logger.info(
            f"Test 2 Passed (Early Exit): Input {input_val} -> Output {result_value}"
        )

        logger.info("All tests passed!")

    finally:
        completion_task.cancel()
        try:
            await completion_task
        except asyncio.CancelledError:
            pass
        await coordinator.stop()


def parse_args():
    parser = argparse.ArgumentParser(description="Three-stage pipeline demo")
    parser.add_argument(
        "--relay",
        type=str,
        choices=["nixl", "shm", "nccl"],
        default="nccl",
        help="Relay backend to use (default: nccl)",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0,1,2",
        help="Comma-separated GPU IDs (e.g. '0,1,2'). Use -1 for CPU-only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    relay_type = args.relay.lower()

    # Parse GPU IDs
    try:
        raw_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
        gpu_ids = [gid for gid in raw_ids if gid >= 0]
        if not gpu_ids:
            if relay_type == "nccl":
                logger.warning(
                    "NCCL requires GPUs! Attempting to use generic cuda device if available."
                )
            else:
                logger.info("No valid GPU IDs provided, running on CPU.")
                gpu_ids = []
    except ValueError:
        logger.warning("Invalid GPU IDs, defaulting to CPU")
        gpu_ids = []

    # Start Processes
    procs = [
        mp.Process(target=run_preprocessor, args=(relay_type, gpu_ids)),
        mp.Process(target=run_encoder, args=(relay_type, gpu_ids)),
        mp.Process(target=run_decoder, args=(relay_type, gpu_ids)),
    ]

    for p in procs:
        p.start()

    try:
        asyncio.run(run_coordinator_main(relay_type))
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down...")
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join()


if __name__ == "__main__":
    # Ensure spawn for CUDA compatibility
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
