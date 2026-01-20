# SPDX-License-Identifier: Apache-2.0
"""Three-stage pipeline demo with dynamic Relay selection (Nixl/Shm)."""

import argparse
import asyncio
import logging
import multiprocessing as mp
from typing import Any

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


def stage1_get_next(request_id: str, output: Any) -> str | None:
    return "encoder"


def stage2_get_next(request_id: str, output: Any) -> str | None:
    if isinstance(output, (int, float)) and output < 0:
        logger.info("Encoder: output=%s is negative, early exit!", output)
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
):
    """Generic stage runner with unified Relay configuration."""
    # Move imports here to avoid multiprocessing pickling issues
    import asyncio
    import logging

    from sglang_omni import Stage, Worker
    from sglang_omni.executors import FrontendExecutor
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

    engine = FrontendExecutor(processor)
    worker = Worker(engine)

    # --- Build Unified Relay Config ---
    # This dictionary matches the new logic in Stage.__init__
    relay_config = {
        "relay_type": relay_type,  # "nixl" or "shm"
        "worker_id": f"worker_{name}",  # Unique ID for Nixl/Shm
        "slot_size_mb": 64,
        "credits": 4,
        "gpu_id": gpu_id,  # Pass GPU ID (or None for CPU)
    }

    logger.info(
        "Stage %s initializing with %s (gpu_id=%s)", name, relay_type.upper(), gpu_id
    )

    # Initialize Stage (Stage will create the correct Relay based on config)
    stage = Stage(
        name=name,
        get_next=get_next,
        recv_endpoint=endpoint,
        coordinator_endpoint=COORDINATOR_ENDPOINT,
        abort_endpoint=ABORT_ENDPOINT,
        endpoints=ENDPOINTS,
        relay_config=relay_config,  # Pass the config dict
    )
    stage.add_worker(worker)

    asyncio.run(stage.run())


# --- Stage Runners Wrappers ---


def run_preprocessor(relay_type: str, gpu_ids: list[int]):
    # Stage 1 usually lighter, might map to first GPU or CPU
    gpu = gpu_ids[0] if gpu_ids else None
    run_stage(
        name="preprocessor",
        endpoint=STAGE1_ENDPOINT,
        transform=lambda x: x * 10 - 5,
        delay=0.05,
        get_next=stage1_get_next,
        relay_type=relay_type,
        gpu_id=gpu,
    )


def run_encoder(relay_type: str, gpu_ids: list[int]):
    gpu = gpu_ids[1] if len(gpu_ids) > 1 else (gpu_ids[0] if gpu_ids else None)
    run_stage(
        name="encoder",
        endpoint=STAGE2_ENDPOINT,
        transform=lambda x: x * x if x >= 0 else x,
        delay=0.1,
        get_next=stage2_get_next,
        relay_type=relay_type,
        gpu_id=gpu,
    )


def run_decoder(relay_type: str, gpu_ids: list[int]):
    gpu = gpu_ids[2] if len(gpu_ids) > 2 else (gpu_ids[0] if gpu_ids else None)
    run_stage(
        name="decoder",
        endpoint=STAGE3_ENDPOINT,
        transform=lambda x: x + 1000,
        delay=0.1,
        get_next=stage3_get_next,
        relay_type=relay_type,
        gpu_id=gpu,
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
        await asyncio.sleep(2.0)  # Wait for stages to connect

        logger.info("=" * 60)
        logger.info(f"Running Tests with Relay: {relay_type.upper()}")
        logger.info("=" * 60)

        # Test 1: Normal Flow
        input_val = 5
        # 5 -> 45 -> 2025 -> 3025
        expected = 3025
        result = await coordinator.submit("req-1", input_val)
        assert result == expected
        logger.info(f"Test 1 Passed: Input {input_val} -> Output {result}")

        # Test 2: Early Exit
        input_val = 0
        # 0 -> -5 -> -5 (Early Exit)
        expected = -5
        result = await coordinator.submit("req-2", input_val)
        assert result == expected
        logger.info(f"Test 2 Passed (Early Exit): Input {input_val} -> Output {result}")

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
        choices=["nixl", "shm"],
        default="nixl",  # <--- Modify here: change "shm" to "nixl"
        help="Relay backend to use (default: nixl)",  # <--- Also consider updating the help text
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
        # Filter out -1 to represent None (CPU) if desired,
        # or keep valid IDs. Here we assume valid IDs are >= 0.
        gpu_ids = [gid for gid in raw_ids if gid >= 0]
        if not gpu_ids:
            logger.info("No valid GPU IDs provided, running on CPU.")
            gpu_ids = []  # Empty list implies CPU for run_stage logic
    except ValueError:
        logger.warning("Invalid GPU IDs, defaulting to CPU")
        gpu_ids = []

    # Start Processes
    # Note: We pass relay_type and gpu_ids, not pre-built configs or objects
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
            p.terminate()
            p.join()


if __name__ == "__main__":
    # Ensure spawn for CUDA compatibility if using GPUs
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
