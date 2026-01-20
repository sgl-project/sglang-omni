# SPDX-License-Identifier: Apache-2.0
"""Two-stage pipeline demo with dynamic Relay selection (Nixl/Shm).

This demonstrates:
- Stage 1: Receives input, doubles it, sends to Stage 2
- Stage 2: Receives from Stage 1, adds 100, completes
"""

import argparse
import asyncio
import logging
import multiprocessing as mp
import time
from typing import Any

from sglang_omni import Coordinator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Endpoints
STAGE1_ENDPOINT = "tcp://127.0.0.1:15001"
STAGE2_ENDPOINT = "tcp://127.0.0.1:15002"
COORDINATOR_ENDPOINT = "tcp://127.0.0.1:15000"
ABORT_ENDPOINT = "tcp://127.0.0.1:15099"

# All endpoints for routing
ENDPOINTS = {
    "stage1": STAGE1_ENDPOINT,
    "stage2": STAGE2_ENDPOINT,
}


def stage1_get_next(request_id: str, output: Any) -> str | None:
    """Stage 1 always routes to Stage 2."""
    return "stage2"


def stage2_get_next(request_id: str, output: Any) -> str | None:
    """Stage 2 is the final stage."""
    return None  # END


def run_stage(
    name: str,
    endpoint: str,
    transform,
    delay: float,
    get_next,
    relay_type: str = "shm",
    gpu_id: int | None = None,
):
    """Generic stage runner.

    Args:
        name: Stage name
        endpoint: ZMQ endpoint for receiving work
        transform: Data transformation function
        delay: Processing delay (simulation)
        get_next: Routing function
        relay_type: "nixl" or "shm"
        gpu_id: GPU ID to use (or None for CPU)
    """
    import asyncio
    import logging

    from sglang_omni import Stage, Worker
    from sglang_omni.executors import FrontendExecutor
    from sglang_omni.proto import StagePayload

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
    relay_config = {
        "relay_type": relay_type,
        "worker_id": f"worker_{name}",
        "slot_size_mb": 64,
        "credits": 4,
        "gpu_id": gpu_id,  # Pass GPU ID (or None for CPU fallback)
    }

    logger.info(
        "Stage %s initializing with %s (gpu_id=%s)",
        name,
        relay_type.upper(),
        gpu_id,
    )

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


def run_stage1(relay_type: str, gpu_ids: list[int]):
    """Run Stage 1 in a separate process."""
    gpu = gpu_ids[0] if gpu_ids else None
    run_stage(
        name="stage1",
        endpoint=STAGE1_ENDPOINT,
        transform=lambda x: x * 2,
        delay=0.1,
        get_next=stage1_get_next,
        relay_type=relay_type,
        gpu_id=gpu,
    )


def run_stage2(relay_type: str, gpu_ids: list[int]):
    """Run Stage 2 in a separate process."""
    gpu = gpu_ids[1] if len(gpu_ids) > 1 else (gpu_ids[0] if gpu_ids else None)
    run_stage(
        name="stage2",
        endpoint=STAGE2_ENDPOINT,
        transform=lambda x: x + 100,
        delay=0.5,
        get_next=stage2_get_next,
        relay_type=relay_type,
        gpu_id=gpu,
    )


async def run_coordinator_main(relay_type: str):
    """Run the coordinator and test the pipeline."""
    coordinator = Coordinator(
        completion_endpoint=COORDINATOR_ENDPOINT,
        abort_endpoint=ABORT_ENDPOINT,
        entry_stage="stage1",
    )

    # Register stages
    coordinator.register_stage("stage1", STAGE1_ENDPOINT)
    coordinator.register_stage("stage2", STAGE2_ENDPOINT)

    # Start coordinator
    await coordinator.start()

    # Start completion loop in background
    completion_task = asyncio.create_task(coordinator.run_completion_loop())

    try:
        # Give stages time to start
        await asyncio.sleep(2.0)

        logger.info("=" * 60)
        logger.info(f"Running Tests with Relay: {relay_type.upper()}")
        logger.info("=" * 60)

        # Test 1: Normal flow
        logger.info("--- Test 1: Normal flow ---")
        input_value = 10
        # (10 * 2) + 100 = 120
        expected = 120
        result = await coordinator.submit("req-001", input_value)
        assert result == expected
        logger.info(f"Test 1 PASSED: Input {input_value} -> Output {result}")

        # Test 2: Multiple requests
        logger.info("--- Test 2: Multiple sequential requests ---")
        for i in range(3):
            input_val = (i + 1) * 5
            expected = (input_val * 2) + 100
            result = await coordinator.submit(f"req-multi-{i}", input_val)
            assert result == expected
        logger.info("Test 2 PASSED!")

        # Test 3: Abort
        logger.info("--- Test 3: Abort request ---")

        async def submit_and_abort():
            submit_task = asyncio.create_task(coordinator.submit("req-abort-1", 999))
            await asyncio.sleep(0.2)
            aborted = await coordinator.abort("req-abort-1")
            assert aborted
            try:
                await submit_task
                assert False, "Should have been cancelled"
            except asyncio.CancelledError:
                pass

        await submit_and_abort()
        logger.info("Test 3 PASSED!")

        # Test 4: Graceful shutdown
        logger.info("--- Test 4: Graceful shutdown ---")
        await coordinator.shutdown_stages()
        await asyncio.sleep(0.5)
        logger.info("Test 4 PASSED!")

        logger.info("=" * 60)
        logger.info("ALL TESTS PASSED!")
        logger.info("=" * 60)

    finally:
        # Cleanup
        completion_task.cancel()
        try:
            await completion_task
        except asyncio.CancelledError:
            pass
        await coordinator.stop()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Two-stage pipeline demo with configurable relay backend"
    )
    parser.add_argument(
        "--relay",
        type=str,
        choices=["nixl", "shm"],
        default="nixl",  # Default to Nixl (Change to 'shm' if you prefer)
        help="Relay backend to use (default: nixl)",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0,1",
        help="Comma-separated GPU IDs (e.g. '0,1'). Use -1 for CPU-only.",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    relay_type = args.relay.lower()

    # Parse GPU IDs
    try:
        raw_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
        gpu_ids = [gid for gid in raw_ids if gid >= 0]
        if not gpu_ids:
            logger.info("No valid GPU IDs provided, running on CPU.")
            gpu_ids = []
    except ValueError:
        logger.warning("Invalid GPU IDs format, using default: 0,1")
        gpu_ids = [0, 1]

    # Start stage processes
    stage1_proc = mp.Process(
        target=run_stage1,
        name="Stage1",
        args=(relay_type, gpu_ids),
    )
    stage2_proc = mp.Process(
        target=run_stage2,
        name="Stage2",
        args=(relay_type, gpu_ids),
    )

    stage1_proc.start()
    stage2_proc.start()

    logger.info(
        "Stage processes started: stage1=%d, stage2=%d",
        stage1_proc.pid,
        stage2_proc.pid,
    )

    try:
        # Give stages time to initialize
        time.sleep(1.5)
        asyncio.run(run_coordinator_main(relay_type))

    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.error("Error: %s", e)
        raise
    finally:
        # Cleanup - wait for graceful shutdown first
        logger.info("Waiting for stage processes to exit...")
        stage1_proc.join(timeout=2)
        stage2_proc.join(timeout=2)

        # Force kill if still alive
        if stage1_proc.is_alive():
            stage1_proc.terminate()
        if stage2_proc.is_alive():
            stage2_proc.terminate()

        logger.info("Done")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
