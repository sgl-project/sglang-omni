# SPDX-License-Identifier: Apache-2.0
"""Three-stage pipeline demo.

Simulates an omni-model pipeline:
- Stage 1 (Preprocessor): Input validation + transformation (CPU-like)
- Stage 2 (Encoder): Encode/process data (GPU-like)
- Stage 3 (Decoder): Decode and produce final output (GPU-like)

Also demonstrates:
- Early exit (Stage 2 can skip Stage 3 based on output)
- DAG extensibility
- Uses NixlRelay for data transfer

Usage:
    # Use NixlRelay (default)
    python run_three_stage_demo.py

    # Use NixlRelay
    python run_three_stage_demo.py --relay nixl

    # Use NixlRelay with custom config
    python run_three_stage_demo.py --relay nixl --nixl-host 192.168.1.100 --nixl-metadata-server http://192.168.1.100:8080/metadata
"""

import argparse
import asyncio
import logging
import multiprocessing as mp
import time
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

# Default NixlRelay configuration
DEFAULT_NIXL_CONFIG = {
    "host": "127.0.0.1",
    "metadata_server": "http://127.0.0.1:8080/metadata",
    "device_name": "",
}


def stage1_get_next(request_id: str, output: Any) -> str | None:
    """Preprocessor always routes to Encoder."""
    return "encoder"


def stage2_get_next(request_id: str, output: Any) -> str | None:
    """Encoder routes to Decoder, or early-exits if output < 0."""
    # Early exit condition: if output is negative, skip decoder
    if isinstance(output, (int, float)) and output < 0:
        logger.info("Encoder: output=%s is negative, early exit!", output)
        return None  # END
    return "decoder"


def stage3_get_next(request_id: str, output: Any) -> str | None:
    """Decoder is the final stage."""
    return None  # END


def run_stage(
    name: str,
    endpoint: str,
    transform,
    delay: float,
    get_next,
    relay_type: str = "nixl",
    nixl_config: dict[str, Any] | None = None,
    gpu_id: int = 0,
):
    """Generic stage runner.

    Args:
        name: Stage name
        endpoint: ZMQ endpoint for receiving work
        transform: Data transformation function
        delay: Processing delay (simulation)
        get_next: Routing function
        relay_type: Relay type ("nixl")
        nixl_config: NixlRelay configuration dict
        gpu_id: GPU ID to use (for NixlRelay, default: 0)
    """
    import asyncio
    import logging

    from sglang_omni import EchoEngine, Stage, Worker

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    engine = EchoEngine(transform=transform, delay=delay)
    worker = Worker(engine)

    # Configure relay - always use NixlRelay
    if nixl_config is None:
        nixl_config = DEFAULT_NIXL_CONFIG
    # Add worker_id and gpu_id to config
    relay_config = {
        **nixl_config,
        "worker_id": f"worker_{name}",
        "gpu_id": gpu_id,
    }
    logger.info(
        "Stage %s: Initializing with NixlRelay (worker_id=%s, gpu_id=%d)",
        name,
        relay_config["worker_id"],
        gpu_id,
    )

    stage = Stage(
        name=name,
        get_next=get_next,
        recv_endpoint=endpoint,
        coordinator_endpoint=COORDINATOR_ENDPOINT,
        abort_endpoint=ABORT_ENDPOINT,
        endpoints=ENDPOINTS,
        relay_config=relay_config,  # Configuration dict for NixlRelay
    )
    stage.add_worker(worker)

    asyncio.run(stage.run())


def run_preprocessor(relay_type: str, nixl_config: dict[str, Any] | None, gpu_id: int):
    """Stage 1: Validate and normalize input."""
    # Preprocessor: multiply by 10 and subtract 5
    run_stage(
        name="preprocessor",
        endpoint=STAGE1_ENDPOINT,
        transform=lambda x: x * 10 - 5,
        delay=0.05,
        get_next=stage1_get_next,
        relay_type=relay_type,
        nixl_config=nixl_config,
        gpu_id=gpu_id,
    )


def run_encoder(relay_type: str, nixl_config: dict[str, Any] | None, gpu_id: int):
    """Stage 2: Encode data."""
    # Encoder: square the value
    run_stage(
        name="encoder",
        endpoint=STAGE2_ENDPOINT,
        transform=lambda x: x * x if x >= 0 else x,  # Keep negative for early exit
        delay=0.1,
        get_next=stage2_get_next,
        relay_type=relay_type,
        nixl_config=nixl_config,
        gpu_id=gpu_id,
    )


def run_decoder(relay_type: str, nixl_config: dict[str, Any] | None, gpu_id: int):
    """Stage 3: Decode and finalize."""
    # Decoder: add 1000
    run_stage(
        name="decoder",
        endpoint=STAGE3_ENDPOINT,
        transform=lambda x: x + 1000,
        delay=0.1,
        get_next=stage3_get_next,
        relay_type=relay_type,
        nixl_config=nixl_config,
        gpu_id=gpu_id,
    )


async def run_coordinator_main(relay_type: str):
    """Run the coordinator and test the pipeline."""
    coordinator = Coordinator(
        completion_endpoint=COORDINATOR_ENDPOINT,
        abort_endpoint=ABORT_ENDPOINT,
        entry_stage="preprocessor",
    )

    # Register stages
    coordinator.register_stage("preprocessor", STAGE1_ENDPOINT)
    coordinator.register_stage("encoder", STAGE2_ENDPOINT)
    coordinator.register_stage("decoder", STAGE3_ENDPOINT)

    await coordinator.start()
    completion_task = asyncio.create_task(coordinator.run_completion_loop())

    try:
        await asyncio.sleep(1.0)

        # Test 1: Normal 3-stage flow
        logger.info("=" * 60)
        relay_label = " (NixlRelay)"
        logger.info("Test 1: Normal 3-stage flow%s", relay_label)
        logger.info("=" * 60)

        # Input: 5
        # Preprocessor: 5 * 10 - 5 = 45
        # Encoder: 45 * 45 = 2025
        # Decoder: 2025 + 1000 = 3025
        input_val = 5
        expected = ((input_val * 10 - 5) ** 2) + 1000
        logger.info("Input: %d", input_val)
        logger.info("Expected: ((%d * 10 - 5)^2) + 1000 = %d", input_val, expected)

        result = await coordinator.submit("req-normal-1", input_val)
        logger.info("Result: %d", result)
        assert result == expected, f"Expected {expected}, got {result}"
        logger.info("Test 1 PASSED!")

        # Test 2: Early exit (skip decoder)
        logger.info("=" * 60)
        logger.info("Test 2: Early exit (negative value skips decoder)")
        logger.info("=" * 60)

        # Input: 0
        # Preprocessor: 0 * 10 - 5 = -5
        # Encoder: -5 (kept negative for early exit) -> returns -5, skips decoder
        input_val = 0
        expected = -5  # Early exit at encoder
        logger.info("Input: %d", input_val)
        logger.info("Expected: preprocessor outputs -5, encoder early-exits with -5")

        result = await coordinator.submit("req-early-exit-1", input_val)
        logger.info("Result: %d", result)
        assert result == expected, f"Expected {expected}, got {result}"
        logger.info("Test 2 PASSED!")

        # Test 3: Multiple concurrent requests
        logger.info("=" * 60)
        logger.info("Test 3: Multiple concurrent requests")
        logger.info("=" * 60)

        async def submit_request(req_id: str, value: int) -> tuple[str, int, int]:
            result = await coordinator.submit(req_id, value)
            return req_id, value, result

        tasks = [submit_request(f"req-concurrent-{i}", i + 1) for i in range(5)]
        results = await asyncio.gather(*tasks)

        for req_id, input_val, result in results:
            expected = ((input_val * 10 - 5) ** 2) + 1000
            logger.info(
                "%s: input=%d, expected=%d, got=%d", req_id, input_val, expected, result
            )
            assert result == expected, f"{req_id}: Expected {expected}, got {result}"

        logger.info("Test 3 PASSED!")

        # Test 4: Abort in 3-stage pipeline
        logger.info("=" * 60)
        logger.info("Test 4: Abort in 3-stage pipeline")
        logger.info("=" * 60)

        async def submit_and_abort():
            task = asyncio.create_task(coordinator.submit("req-abort-3stage", 100))
            await asyncio.sleep(0.15)
            aborted = await coordinator.abort("req-abort-3stage")
            logger.info("Abort successful: %s", aborted)
            try:
                await task
                assert False, "Should have been cancelled"
            except asyncio.CancelledError:
                logger.info("Request correctly cancelled")

        await submit_and_abort()
        logger.info("Test 4 PASSED!")

        # Test 5: Health and graceful shutdown
        logger.info("=" * 60)
        logger.info("Test 5: Health and graceful shutdown")
        logger.info("=" * 60)

        health = coordinator.health()
        logger.info("Health: %s", health)
        assert len(health["stages"]) == 3

        await coordinator.shutdown_stages()
        await asyncio.sleep(0.5)
        logger.info("Test 5 PASSED!")

        relay_label = " (NixlRelay)"
        logger.info("=" * 60)
        logger.info("ALL TESTS PASSED!%s", relay_label)
        logger.info("=" * 60)

    finally:
        completion_task.cancel()
        try:
            await completion_task
        except asyncio.CancelledError:
            pass
        await coordinator.stop()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Three-stage pipeline demo with configurable relay backend"
    )
    parser.add_argument(
        "--relay",
        type=str,
        choices=["nixl"],
        default="nixl",
        help="Relay backend to use (default: nixl)",
    )
    parser.add_argument(
        "--nixl-host",
        type=str,
        default="127.0.0.1",
        help="NIXL host address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--nixl-metadata-server",
        type=str,
        default="http://127.0.0.1:8080/metadata",
        help="NIXL metadata server URL (default: http://127.0.0.1:8080/metadata)",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0,1,2",
        help="Comma-separated GPU IDs for each stage (default: 0,1,2)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    relay_type = args.relay.lower()

    # Always use NixlRelay
    if True:
        logger.info("Starting three-stage pipeline demo with NixlRelay...")
        # Check if NixlRelay is available
        try:
            pass

            # Build NIXL config
            nixl_config = {
                "host": args.nixl_host,
                "metadata_server": args.nixl_metadata_server,
                "device_name": "",
            }

        except ImportError as e:
            logger.error("Failed to import NixlRelay: %s", e)
            logger.error(
                "Please ensure dynamo.nixl_connect is available and NIXL metadata server is running."
            )
            raise
    # nixl_config is already set above

    # Parse GPU IDs
    try:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
        if len(gpu_ids) < 3:
            logger.warning(
                "Only %d GPU IDs provided, using first GPU for remaining stages",
                len(gpu_ids),
            )
            gpu_ids.extend([gpu_ids[0]] * (3 - len(gpu_ids)))
    except ValueError:
        logger.warning("Invalid GPU IDs format, using default: 0,1,2")
        gpu_ids = [0, 1, 2]

    # Start stage processes
    procs = [
        mp.Process(
            target=run_preprocessor,
            name="Preprocessor",
            args=(relay_type, nixl_config, gpu_ids[0]),
        ),
        mp.Process(
            target=run_encoder,
            name="Encoder",
            args=(relay_type, nixl_config, gpu_ids[1]),
        ),
        mp.Process(
            target=run_decoder,
            name="Decoder",
            args=(relay_type, nixl_config, gpu_ids[2]),
        ),
    ]

    for p in procs:
        p.start()

    logger.info("Stage processes started: %s", [p.pid for p in procs])

    try:
        time.sleep(1.5)
        asyncio.run(run_coordinator_main(relay_type))
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.error("Error: %s", e)
        raise
    finally:
        logger.info("Waiting for stage processes to exit...")
        for p in procs:
            p.join(timeout=2)
            if p.is_alive():
                logger.warning("Force killing %s", p.name)
                p.terminate()
                p.join(timeout=1)
        logger.info("Done")


if __name__ == "__main__":
    main()
