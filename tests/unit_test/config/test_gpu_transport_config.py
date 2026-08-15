# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
from pydantic import ValidationError

from sglang_omni.config.schema import CommConfig, StageConfig


def test_remote_backend_is_closed_enum() -> None:
    assert CommConfig().remote_backend == "auto"
    assert CommConfig(remote_backend="nixl").remote_backend == "nixl"
    assert CommConfig(remote_backend="mooncake").remote_backend == "mooncake"

    with pytest.raises(ValidationError, match="remote_backend"):
        CommConfig(remote_backend="ucx")


def test_existing_cuda_ipc_config_contract_is_unchanged() -> None:
    comm = CommConfig(cuda_ipc_slot_size_kb=128, cuda_ipc_pool_size_mb=256)
    stage = StageConfig(
        name="decode",
        factory="example.create",
        terminal=True,
        disable_direct_cuda_ipc_payload=True,
    )

    assert comm.cuda_ipc_slot_size_kb == 128
    assert comm.cuda_ipc_pool_size_mb == 256
    assert stage.disable_direct_cuda_ipc_payload is True
    assert "gpu_ipc_slot_size_kb" not in comm.model_dump()
    assert "disable_direct_gpu_ipc_payload" not in stage.model_dump()
