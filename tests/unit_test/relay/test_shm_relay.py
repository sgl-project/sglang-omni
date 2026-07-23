# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import os

import pytest
import torch

from sglang_omni.relay.shm import ShmRelay


def test_shm_put_timeout_unlinks_block_and_releases_credit() -> None:
    async def run() -> None:
        relay = ShmRelay(engine_id="sender", device="cpu", credits=1)
        tensor = torch.arange(16, dtype=torch.uint8)
        op = await relay.put_async(tensor, request_id="r0")
        shm_name = op.metadata["transfer_info"]["shm_name"]
        shm_path = f"/dev/shm/{shm_name}"

        assert os.path.exists(shm_path)
        with pytest.raises(TimeoutError, match="was not consumed in time"):
            await op.wait_for_completion(timeout=0.0)
        assert not os.path.exists(shm_path)

        # The timeout path released the semaphore credit; another put should not
        # block even though the first transfer failed.
        op2 = await asyncio.wait_for(
            relay.put_async(tensor, request_id="r1"),
            timeout=1.0,
        )
        shm_name2 = op2.metadata["transfer_info"]["shm_name"]
        shm_path2 = f"/dev/shm/{shm_name2}"
        try:
            assert os.path.exists(shm_path2)
        finally:
            with pytest.raises(TimeoutError):
                await op2.wait_for_completion(timeout=0.0)
            assert not os.path.exists(shm_path2)

    asyncio.run(run())
