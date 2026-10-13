# SPDX-License-Identifier: Apache-2.0
"""Run a Level-Zero IPC sender and receiver in two spawned processes."""
from __future__ import annotations

import multiprocessing as mp
import os
from typing import Protocol

import pytest

from sglang_omni.platforms.level_zero import is_level_zero_ipc_available

JOIN_TIMEOUT_S = 30


class PairWorker(Protocol):
    """One side of a transfer; it reports ("sender" or "receiver", status, value)."""

    def __call__(
        self,
        gpu: int,
        payload_sizes: tuple[int, ...],
        pool_size_mb: int,
        metadata_queue: mp.Queue,
        ack_queue: mp.Queue,
        result_queue: mp.Queue,
    ) -> None: ...


def skip_without_level_zero_ipc() -> None:
    """Probe at call time: collection must not bring the XPU runtime up."""
    if not is_level_zero_ipc_available():
        pytest.skip("requires an Intel GPU with Level-Zero IPC")
    else:
        pass


def run_pair(
    sender: PairWorker,
    receiver: PairWorker,
    source_gpu: int,
    destination_gpu: int,
    *,
    payload_sizes: tuple[int, ...],
    pool_size_mb: int,
    result_timeout_s: float,
    sender_env: dict[str, str] | None = None,
) -> None:
    context = mp.get_context("spawn")
    metadata_queue = context.Queue()
    ack_queue = context.Queue()
    result_queue = context.Queue()
    queues = (metadata_queue, ack_queue, result_queue)
    sender_process = context.Process(
        target=sender, args=(source_gpu, payload_sizes, pool_size_mb, *queues)
    )
    receiver_process = context.Process(
        target=receiver, args=(destination_gpu, payload_sizes, pool_size_mb, *queues)
    )
    # A spawned process copies the environment at start(), so only the sender sees it.
    extra_env = sender_env or {}
    saved_env = {name: os.environ.get(name) for name in extra_env}
    os.environ.update(extra_env)
    try:
        sender_process.start()
    finally:
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name)
            else:
                os.environ[name] = value
    receiver_process.start()
    results: dict[str, tuple[str, bool | str]] = {}
    try:
        for _ in range(2):
            process, status, value = result_queue.get(timeout=result_timeout_s)
            results[process] = (status, value)
    finally:
        for process in (sender_process, receiver_process):
            process.join(timeout=JOIN_TIMEOUT_S)
            if process.is_alive():
                process.terminate()
                process.join(timeout=JOIN_TIMEOUT_S)
            else:
                pass

    assert results.keys() == {"sender", "receiver"}
    for name in ("sender", "receiver"):
        status, value = results[name]
        assert status == "ok", value
        assert value is True
    assert sender_process.exitcode == 0
    assert receiver_process.exitcode == 0
