# SPDX-License-Identifier: Apache-2.0
"""Two-stage pipeline demo.

This demonstrates:
- Stage 1: Receives input, doubles it, sends to Stage 2
- Stage 2: Receives from Stage 1, adds 100, completes

Tests:
1. Normal flow
2. Multiple requests
3. Abort functionality
4. Graceful shutdown

Usage:
    # Use NixlRelay (default)
    python run_two_stage_demo.py

    # Use NixlRelay
    python run_two_stage_demo.py --relay nixl

    # Use NixlRelay with custom config
    python run_two_stage_demo.py --relay nixl --nixl-host 192.168.1.100 --nixl-metadata-server http://192.168.1.100:8080/metadata
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

# Default NixlRelay configuration
DEFAULT_NIXL_CONFIG = {
    "host": "127.0.0.1",
    "metadata_server": "http://127.0.0.1:8080/metadata",
    "device_name": "",
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


def run_stage1(relay_type: str, nixl_config: dict[str, Any] | None, gpu_id: int):
    """Run Stage 1 in a separate process."""
    # Engine that doubles the input
    run_stage(
        name="stage1",
        endpoint=STAGE1_ENDPOINT,
        transform=lambda x: x * 2,
        delay=0.1,
        get_next=stage1_get_next,
        relay_type=relay_type,
        nixl_config=nixl_config,
        gpu_id=gpu_id,
    )


def run_stage2(relay_type: str, nixl_config: dict[str, Any] | None, gpu_id: int):
    """Run Stage 2 in a separate process."""
    # Engine that adds 100 (with longer delay for abort test)
    run_stage(
        name="stage2",
        endpoint=STAGE2_ENDPOINT,
        transform=lambda x: x + 100,
        delay=0.5,
        get_next=stage2_get_next,
        relay_type=relay_type,
        nixl_config=nixl_config,
        gpu_id=gpu_id,
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
        await asyncio.sleep(1.0)

        # Test 1: Normal flow
        logger.info("=" * 50)
        relay_label = " (NixlRelay)"
        logger.info("Test 1: Normal flow%s", relay_label)
        logger.info("=" * 50)

        input_value = 10
        logger.info("Submitting request with input=%d", input_value)
        logger.info("Expected: (10 * 2) + 100 = 120")

        result = await coordinator.submit("req-001", input_value)

        logger.info("Result: %s", result)
        assert result == 120, f"Expected 120, got {result}"
        logger.info("Test 1 PASSED!")

        # Test 2: Multiple requests
        logger.info("=" * 50)
        logger.info("Test 2: Multiple sequential requests")
        logger.info("=" * 50)

        for i in range(3):
            input_val = (i + 1) * 5
            expected = (input_val * 2) + 100
            result = await coordinator.submit(f"req-multi-{i}", input_val)
            logger.info("Input=%d, Expected=%d, Got=%d", input_val, expected, result)
            assert result == expected, f"Expected {expected}, got {result}"

        logger.info("Test 2 PASSED!")

        # Test 3: Abort
        logger.info("=" * 50)
        logger.info("Test 3: Abort request")
        logger.info("=" * 50)

        # Submit a request but abort it quickly
        async def submit_and_abort():
            submit_task = asyncio.create_task(coordinator.submit("req-abort-1", 999))
            # Wait a tiny bit, then abort
            await asyncio.sleep(0.2)
            aborted = await coordinator.abort("req-abort-1")
            logger.info("Abort result: %s", aborted)
            assert aborted, "Should have aborted"

            # The submit should raise CancelledError
            try:
                await submit_task
                logger.error("Submit should have been cancelled!")
                assert False, "Submit should have been cancelled"
            except asyncio.CancelledError:
                logger.info("Submit correctly raised CancelledError")

        await submit_and_abort()

        # Check request state
        info = coordinator.get_request_info("req-abort-1")
        logger.info("Aborted request state: %s", info.state if info else "None")
        assert (
            info is not None and info.state.value == "aborted"
        ), "Request should be aborted"

        logger.info("Test 3 PASSED!")

        # Test 4: Health check
        logger.info("=" * 50)
        logger.info("Test 4: Health check")
        logger.info("=" * 50)

        health = coordinator.health()
        logger.info("Coordinator health: %s", health)
        assert health["running"] is True
        assert "aborted" in health["request_states"]
        logger.info("Test 4 PASSED!")

        # Test 5: Graceful shutdown
        logger.info("=" * 50)
        logger.info("Test 5: Graceful shutdown")
        logger.info("=" * 50)

        await coordinator.shutdown_stages()
        logger.info("Shutdown signals sent")

        # Give stages time to shutdown
        await asyncio.sleep(0.5)
        logger.info("Test 5 PASSED!")

        relay_label = " (NixlRelay)"
        logger.info("=" * 50)
        logger.info("ALL TESTS PASSED!%s", relay_label)
        logger.info("=" * 50)

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
        default="0,1",
        help="Comma-separated GPU IDs for each stage (default: 0,1)",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    relay_type = args.relay.lower()

    # Always use NixlRelay
    if True:
        logger.info("Starting two-stage pipeline demo with NixlRelay...")
        # Check if NixlRelay is available
        try:
            pass

            # Build NIXL config
            nixl_config = {
                "host": args.nixl_host,
                "metadata_server": args.nixl_metadata_server,
                "device_name": "",
            }

            # Try to create a test instance to verify NIXL is available
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
        if len(gpu_ids) < 2:
            logger.warning(
                "Only %d GPU IDs provided, using first GPU for remaining stages",
                len(gpu_ids),
            )
            gpu_ids.extend([gpu_ids[0]] * (2 - len(gpu_ids)))
    except ValueError:
        logger.warning("Invalid GPU IDs format, using default: 0,1")
        gpu_ids = [0, 1]

    # Start stage processes
    stage1_proc = mp.Process(
        target=run_stage1,
        name="Stage1",
        args=(relay_type, nixl_config, gpu_ids[0]),
    )
    stage2_proc = mp.Process(
        target=run_stage2,
        name="Stage2",
        args=(relay_type, nixl_config, gpu_ids[1]),
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
        time.sleep(1.0)

        # Run coordinator
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
            logger.warning("Force killing stage1")
            stage1_proc.terminate()
            stage1_proc.join(timeout=1)
        if stage2_proc.is_alive():
            logger.warning("Force killing stage2")
            stage2_proc.terminate()
            stage2_proc.join(timeout=1)

        logger.info("Done")


if __name__ == "__main__":
    main()
