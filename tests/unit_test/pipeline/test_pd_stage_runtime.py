# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.scheduling.pd_continuation import DecodeContinuation


class _BlockingComm:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.kwargs = None

    async def send_kv_pages(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.started.set()
        await self.release.wait()


def _continuation() -> DecodeContinuation:
    return DecodeContinuation(
        request_id="request-1",
        transfer_id="transfer-1",
        origin_input_ids=[1, 2, 3],
        output_ids=[4],
        vocab_size=16,
        sampling_params={"max_new_tokens": 8},
        cached_tokens=0,
    )


def test_pd_handoff_waits_for_ack_in_background() -> None:
    async def run() -> None:
        stage = object.__new__(Stage)
        stage._comm = _BlockingComm()
        stage._receive_tasks = set()
        stage._clear_request_state = lambda _: None
        stage._on_background_task_done = lambda *_: None
        stage._send_failure = lambda *_: None
        handoff = SimpleNamespace(
            continuation=_continuation(),
            source_pool_id="prefill:kv",
            source_page_indices=(7, 8, 9),
            target_pool_id="decode:kv",
            to_stage="decode",
            lease=object(),
        )

        stage._launch_pd_handoff("request-1", handoff)
        await stage._comm.started.wait()

        task = next(iter(stage._receive_tasks))
        assert not task.done()
        assert stage._comm.kwargs["transfer_id"] == "transfer-1"
        assert stage._comm.kwargs["metadata"]["pd_continuation_present"] is True

        stage._comm.release.set()
        await task

    asyncio.run(run())
