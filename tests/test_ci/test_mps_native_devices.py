# SPDX-License-Identifier: Apache-2.0
"""Model-free, two-GPU validation of a serve-local MPS daemon and CUDA mapping."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.accelerator,
    pytest.mark.skipif(
        shutil.which("nvidia-cuda-mps-control") is None,
        reason="requires NVIDIA MPS tooling and two available GPUs",
    ),
]

_WORKER = """
import json
import sys
import torch
x = torch.ones(16, device='cuda:0')
assert x.sum().item() == 16
print(json.dumps({'uuid': str(torch.cuda.get_device_properties(0).uuid),
                  'count': torch.cuda.device_count()}), flush=True)
sys.stdin.readline()
torch.cuda.synchronize()
"""


@pytest.mark.asyncio
async def test_multiple_gpu_workers_share_one_server_with_correct_local_cuda_zero():
    import torch

    from sglang_omni.mps.runtime import create_for_pipeline
    from sglang_omni.pipeline.stage_workers import (
        StageLaunchConfig,
        StageWorkerProcessSpec,
    )

    if torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs visible to the parent")
    specs = [
        StageWorkerProcessSpec(
            process_name=f"worker-{index}",
            stage_specs=[
                StageLaunchConfig(
                    stage_name=f"worker-{index}",
                    factory="unused",
                    gpu_id=index,
                    placement_gpu_id=index,
                )
            ],
        )
        for index in range(2)
    ]
    # Note (kaige): keep failed run directories and native logs for inspection.
    root = Path(tempfile.mkdtemp(prefix="mps-devices-", dir="/tmp"))
    runtime, devices = create_for_pipeline(
        mode="on",
        process_specs=specs,
        state_root=root,
    )
    assert runtime is not None
    assert len(set(devices.values())) == 2
    workers = {}
    try:
        await runtime.start(devices.values())
        for name, gpu_uuid in devices.items():
            workers[name] = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                _WORKER,
                env={
                    **os.environ,
                    **runtime.worker_env,
                    "CUDA_VISIBLE_DEVICES": gpu_uuid,
                },
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
            )
        for name, worker in workers.items():
            line = await asyncio.wait_for(worker.stdout.readline(), timeout=60)
            assert line, f"{name} exited before CUDA initialization"
            observed = json.loads(line)
            assert observed == {"uuid": devices[name], "count": 1}
        await runtime.verify(worker.pid for worker in workers.values())
        refs = runtime.client.snapshot(runtime.pipe_dir)
        assert {ref.server_pid for ref in refs} == {runtime.server_pid}
        assert {ref.client_pid for ref in refs} == {
            worker.pid for worker in workers.values()
        }
        assert await runtime.probe() is None
    finally:
        for name, worker in workers.items():
            if worker.returncode is None:
                worker.stdin.close()
                try:
                    await asyncio.wait_for(worker.wait(), 10)
                except asyncio.TimeoutError:
                    await runtime.retire_process_clients(worker.pid)
                    worker.kill()
                    await asyncio.wait_for(worker.wait(), 10)
        await runtime.close()
        root.rmdir()

    assert all(worker.returncode == 0 for worker in workers.values())
