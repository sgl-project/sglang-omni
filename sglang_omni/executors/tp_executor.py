# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel executor wrapper for SGLang-backed stages."""

from __future__ import annotations

import asyncio
import multiprocessing
import socket
from collections.abc import Callable
from typing import Any

import torch

from sglang_omni.executors.interface import Executor
from sglang_omni.proto import StagePayload
from sglang_omni.utils import import_string


def _allocate_nccl_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


async def _discard_results(executor: Executor) -> None:
    while True:
        await executor.get_result()


def _tp_worker_entry(
    conn,
    *,
    factory_path: str,
    factory_kwargs: dict[str, Any],
    tp_rank: int,
    gpu_id: int,
    nccl_port: int,
) -> None:
    async def _run() -> None:
        executor = None
        discard_task = None
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > gpu_id:
                torch.cuda.set_device(gpu_id)

            factory = import_string(factory_path)
            executor = factory(
                **factory_kwargs,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                nccl_port=nccl_port,
            )
            await executor.start()
            discard_task = asyncio.create_task(_discard_results(executor))
            conn.send(("ready", None))

            while True:
                cmd, payload = await asyncio.to_thread(conn.recv)
                if cmd == "add_request":
                    await executor.add_request(payload)
                elif cmd == "abort":
                    await executor.abort(payload)
                elif cmd == "stop":
                    break
                else:
                    raise ValueError(f"Unknown TP worker command: {cmd}")
        except Exception as exc:
            try:
                conn.send(("error", repr(exc)))
            except Exception:
                pass
            raise
        finally:
            if discard_task is not None:
                discard_task.cancel()
                try:
                    await discard_task
                except asyncio.CancelledError:
                    pass
            if executor is not None:
                await executor.stop()
            conn.close()

    asyncio.run(_run())


class TensorParallelExecutor(Executor):
    """Wrap a local executor with auxiliary TP worker subprocesses."""

    def __init__(
        self,
        *,
        factory_path: str,
        factory_kwargs: dict[str, Any],
        tp_size: int,
        base_gpu_id: int,
    ):
        if tp_size < 2:
            raise ValueError("TensorParallelExecutor requires tp_size >= 2")

        self._factory_path = factory_path
        self._factory_kwargs = dict(factory_kwargs)
        self._tp_size = int(tp_size)
        self._base_gpu_id = int(base_gpu_id)

        self._local_executor: Executor | None = None
        self._stream_fn: Callable | None = None
        self._feedback_mailbox: Any | None = None
        self._children: list[tuple[multiprocessing.Process, Any]] = []
        self._nccl_port: int | None = None

    async def start(self) -> None:
        if self._local_executor is not None:
            return

        try:
            self._nccl_port = _allocate_nccl_port()
            ctx = multiprocessing.get_context("spawn")

            for tp_rank in range(1, self._tp_size):
                parent_conn, child_conn = ctx.Pipe()
                proc = ctx.Process(
                    target=_tp_worker_entry,
                    kwargs={
                        "conn": child_conn,
                        "factory_path": self._factory_path,
                        "factory_kwargs": self._factory_kwargs,
                        "tp_rank": tp_rank,
                        "gpu_id": self._base_gpu_id + tp_rank,
                        "nccl_port": self._nccl_port,
                    },
                    name=f"tp-worker-{tp_rank}",
                    daemon=True,
                )
                proc.start()
                self._children.append((proc, parent_conn))

            factory = import_string(self._factory_path)
            self._local_executor = factory(
                **self._factory_kwargs,
                gpu_id=self._base_gpu_id,
                tp_rank=0,
                nccl_port=self._nccl_port,
                stream_fn=self._stream_fn,
            )
            if self._feedback_mailbox is not None:
                set_feedback_mailbox = getattr(
                    self._local_executor, "set_feedback_mailbox", None
                )
                if callable(set_feedback_mailbox):
                    set_feedback_mailbox(self._feedback_mailbox)
            await self._local_executor.start()

            loop = asyncio.get_running_loop()
            for proc, conn in self._children:
                status, payload = await loop.run_in_executor(None, conn.recv)
                if status != "ready":
                    raise RuntimeError(
                        f"TP worker {proc.name} failed to start: {payload}"
                    )
        except Exception:
            await self.stop()
            raise

    async def stop(self) -> None:
        for proc, conn in self._children:
            if proc.is_alive():
                try:
                    conn.send(("stop", None))
                except Exception:
                    pass

        if self._local_executor is not None:
            await self._local_executor.stop()
            self._local_executor = None

        for proc, conn in self._children:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
            conn.close()
        self._children.clear()

    async def add_request(self, payload: StagePayload) -> None:
        if self._local_executor is None:
            raise RuntimeError("TensorParallelExecutor not started")

        try:
            for proc, conn in self._children:
                if not proc.is_alive():
                    raise RuntimeError(f"TP worker {proc.name} exited unexpectedly")
                conn.send(("add_request", payload))
            await self._local_executor.add_request(payload)
        except Exception:
            await self.abort(payload.request_id)
            raise

    async def get_result(self) -> StagePayload:
        if self._local_executor is None:
            raise RuntimeError("TensorParallelExecutor not started")
        return await self._local_executor.get_result()

    async def abort(self, request_id: str) -> None:
        if self._local_executor is None:
            return
        for proc, conn in self._children:
            if proc.is_alive():
                conn.send(("abort", request_id))
        await self._local_executor.abort(request_id)

    def set_stream_fn(self, fn) -> None:
        self._stream_fn = fn
        if self._local_executor is not None:
            self._local_executor.set_stream_fn(fn)

    def set_feedback_mailbox(self, mailbox: Any) -> None:
        self._feedback_mailbox = mailbox
        if self._local_executor is not None:
            set_feedback_mailbox = getattr(
                self._local_executor, "set_feedback_mailbox", None
            )
            if callable(set_feedback_mailbox):
                set_feedback_mailbox(mailbox)
