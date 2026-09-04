# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest

from sglang_omni.config import (
    EndpointsConfig,
    FactoryArgs,
    PipelineConfig,
    ProcessConfig,
    StageConfig,
)
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from tests.unit_test.fixtures.pipeline_fakes import fake_factory_path


@pytest.mark.asyncio
async def test_replicated_multistage_process_dispatch_stream_and_shutdown(
    tmp_path,
) -> None:
    config = PipelineConfig(
        model_path="model",
        entry_stage="producer",
        stages=[
            StageConfig(
                name="producer",
                process="pair",
                factory_path=fake_factory_path("make_replica_process_probe_scheduler"),
                factory=FactoryArgs(marker="producer", emit_stream=True),
                next="consumer",
                stream_to=["consumer"],
            ),
            StageConfig(
                name="consumer",
                process="pair",
                factory_path=fake_factory_path("make_replica_process_probe_scheduler"),
                factory=FactoryArgs(marker="consumer"),
                terminal=True,
                can_accept_stream_before_payload=True,
            ),
        ],
        processes={"pair": ProcessConfig(num_replicas=2)},
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
    )
    runner = MultiProcessPipelineRunner(config)

    await runner.start(timeout=30.0)
    processes = [process for group in runner._groups for process in group.processes]
    try:
        results = [
            await asyncio.wait_for(
                runner.coordinator.submit(f"req-{replica_id}", "hello"),
                timeout=10.0,
            )
            for replica_id in range(2)
        ]
    finally:
        await runner.stop()

    for replica_id, result in enumerate(results):
        expected_process = f"process-pair@r{replica_id}"
        assert result["producer_process"] == expected_process
        assert result["consumer_process"] == expected_process
        assert result["stream_process"] == expected_process
        assert result["producer_pid"] == result["consumer_pid"]
        assert result["stream_pid"] == result["consumer_pid"]
        assert result["same_payload_object"] is True

    assert results[0]["producer_pid"] != results[1]["producer_pid"]
    assert all(not process.is_alive() for process in processes)
    assert [process.exitcode for process in processes] == [0, 0]
    assert list(tmp_path.iterdir()) == []
