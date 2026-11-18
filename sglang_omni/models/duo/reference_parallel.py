"""Persistent DCP ranks behind the realtime session owner."""

import atexit
import gc
import logging
import os
import threading
import time
import traceback
from argparse import Namespace
from dataclasses import dataclass, field, replace
from datetime import timedelta
from multiprocessing.connection import Connection, wait
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from sglang.multimodal_gen.runtime.models.dits.duo.model import RealizationSegment
from sglang.multimodal_gen.runtime.models.dits.duo.native_execution import (
    NativeDCPOptions,
)
from sglang.srt.utils import freeze_gc

from sglang_omni.models.duo.reference import DCPRealizationSessionHooks
from sglang_omni.models.duo.streaming import discard_latents
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    ResourceUsage,
    SessionIdentity,
    SessionOperation,
    TimedChunk,
)
from sglang_omni.scheduling.session import SessionContext, SessionHooks

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class RealizationRankReply:
    payload: StagePayload | None = None
    usage: ResourceUsage = field(default_factory=ResourceUsage)
    error: str | None = None
    stage_seconds: dict[str, float] = field(default_factory=dict)
    graph_counts: tuple[int, int] = (0, 0)


def run_realization_rank(
    rank: int,
    devices: tuple[int, ...],
    arguments: Namespace,
    rendezvous: str,
    connection: Connection,
    use_sglang_kernels: bool,
    native_backbone: bool,
    sp_mode: Literal["ulysses", "ring"],
    tp_size: int,
    kv_window_tokens: int | None,
    reference_window_chunks: int,
    native_options: NativeDCPOptions | None,
) -> None:
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(devices[rank]),
        WORLD_SIZE=str(len(devices)),
        LOCAL_WORLD_SIZE=str(len(devices)),
    )
    device = torch.device("cuda", devices[rank])
    torch.cuda.set_device(device)
    hooks: DCPRealizationSessionHooks | None = None
    try:
        if len(devices) > 1:
            distributed.init_process_group(
                "cpu:gloo,cuda:nccl",
                init_method=rendezvous,
                rank=rank,
                world_size=len(devices),
                timeout=timedelta(minutes=10),
            )
        else:
            pass
        hooks = DCPRealizationSessionHooks(
            arguments,
            device=device,
            use_sglang_kernels=use_sglang_kernels,
            native_backbone=native_backbone,
            rendezvous=rendezvous,
            sp_mode=sp_mode,
            tp_size=tp_size,
            kv_window_tokens=kv_window_tokens,
            reference_window_chunks=reference_window_chunks,
            native_options=native_options,
        )
        gc.collect()
        freeze_gc(f"Realization rank {rank}")
        connection.send(RealizationRankReply())
        while True:
            command: tuple[SessionOperation, StagePayload | None] | None = (
                connection.recv()
            )
            if command is None:
                break
            else:
                pass
            operation, payload = command
            identity = operation.session_identity
            result = None
            stage_seconds = {}
            if operation.operation == "open":
                assert payload is not None
                hooks.open(identity, payload.request)
            elif operation.operation == "append":
                assert payload is not None and operation.chunk is not None
                context = SessionContext(
                    session_identity=identity,
                    cancelled=threading.Event(),
                    emit=discard_latents,
                )
                started = time.monotonic()
                result = hooks.append(operation.chunk, payload, context)
                result.data["metrics"]["worker_seconds"] = time.monotonic() - started
                stage_seconds = {
                    name: value
                    for name, value in result.data["metrics"].items()
                    if name
                    in (
                        "cache_prefill_seconds",
                        "cache_evict_seconds",
                        "denoise_seconds",
                        "cache_commit_seconds",
                        "worker_seconds",
                    )
                }
                if rank != 0:
                    result = None
                else:
                    pass
            elif operation.operation == "close":
                hooks.close(identity)
            else:
                raise ValueError(
                    f"Unsupported Realization operation: {operation.operation}"
                )
            connection.send(
                RealizationRankReply(
                    payload=result,
                    usage=hooks.usage(identity),
                    stage_seconds=stage_seconds,
                    graph_counts=hooks.graph_counts,
                )
            )
    except BaseException:
        connection.send(RealizationRankReply(error=traceback.format_exc()))
        raise
    finally:
        # Release captured NCCL references before destroying their process groups.
        if hooks is not None:
            for identity in tuple(hooks.states):
                hooks.close(identity)
        else:
            pass
        connection.close()
        if distributed.is_initialized():
            distributed.destroy_process_group()
        else:
            pass


class ParallelDCPRealizationSessionHooks(SessionHooks):
    def __init__(
        self,
        arguments: Namespace,
        *,
        devices: tuple[int, ...],
        device: torch.device,
        use_sglang_kernels: bool = True,
        native_backbone: bool = False,
        sp_mode: Literal["ulysses", "ring"] = "ulysses",
        tp_size: int = 1,
        kv_window_tokens: int | None = None,
        reference_window_chunks: int = 0,
        native_options: NativeDCPOptions | None = None,
    ) -> None:
        self.device = device
        self.sp_mode = sp_mode
        self.tp_size = tp_size
        self.rank0_graph_counts = (0, 0)
        self.connections: list[Connection] = []
        self.workers: list[multiprocessing.Process] = []
        self.usages: dict[SessionIdentity, ResourceUsage] = {}
        self.rendezvous = TemporaryDirectory(prefix="realization-realtime-ranks-")
        context = multiprocessing.get_context("spawn")
        arguments = Namespace(**vars(arguments))
        arguments.context_parallel_degree = len(devices)
        atexit.register(self.shutdown)
        try:
            for rank in range(len(devices)):
                parent, child = context.Pipe()
                process = context.Process(
                    target=run_realization_rank,
                    args=(
                        rank,
                        devices,
                        arguments,
                        (Path(self.rendezvous.name) / "group").as_uri(),
                        child,
                        use_sglang_kernels,
                        native_backbone,
                        sp_mode,
                        tp_size,
                        kv_window_tokens,
                        reference_window_chunks,
                        native_options,
                    ),
                    daemon=True,
                )
                process.start()
                child.close()
                self.connections.append(parent)
                self.workers.append(process)
            self.receive()
        except BaseException:
            self.shutdown(abort=True)
            raise

    @property
    def graph_counts(self) -> tuple[int, int]:
        return self.rank0_graph_counts

    def receive(self) -> list[RealizationRankReply]:
        replies = []
        for connection in self.connections:
            ready = wait(
                [connection, *(worker.sentinel for worker in self.workers)],
                timeout=600,
            )
            if connection not in ready:
                raise RuntimeError("Realization rank exited or timed out")
            else:
                pass
            reply: RealizationRankReply = connection.recv()
            if reply.error is not None:
                raise RuntimeError(reply.error)
            else:
                replies.append(reply)
        self.rank0_graph_counts = replies[0].graph_counts
        return replies

    def exchange(
        self, operation: SessionOperation, payload: StagePayload | None
    ) -> list[RealizationRankReply]:
        if not self.workers:
            raise RuntimeError("Realization rank group is stopped; restart the server")
        else:
            pass
        try:
            for connection in self.connections:
                connection.send((operation, payload))
            return self.receive()
        except BaseException:
            self.shutdown(abort=True)
            raise

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        operation = SessionOperation("open", session_identity, ("realization",))
        payload = StagePayload(session_identity.id, request, None)
        self.exchange(operation, payload)
        self.usages[session_identity] = ResourceUsage()

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        if context.cancelled.is_set():
            raise RuntimeError("Realization session cancelled")
        else:
            pass
        # CPU tensors avoid exporting live CUDA allocations between rank owners.
        data = dict(payload.data)
        for name in ("reference", "shared", "plan", "video", "audio"):
            segment: RealizationSegment = data[name]
            data[name] = replace(
                segment,
                tokens=segment.tokens.cpu(),
                positions=segment.positions.cpu(),
                roles=segment.roles.cpu(),
            )
        # Input RGB/PCM has already been encoded into the conditioning tensors.
        operation = SessionOperation(
            "append",
            context.session_identity,
            ("realization",),
            replace(chunk, payload=None),
        )
        started = time.monotonic()
        replies = self.exchange(operation, replace(payload, data=data))
        dispatch_seconds = time.monotonic() - started
        result = replies[0].payload
        assert result is not None
        self.usages[context.session_identity] = ResourceUsage(
            kv_tokens=replies[0].usage.kv_tokens,
            bytes=sum(reply.usage.bytes for reply in replies),
        )
        # Aggregate completed rank-local measurements without GPU timing collectives.
        result.data["metrics"]["rank0_worker_seconds"] = replies[0].stage_seconds[
            "worker_seconds"
        ]
        for name in replies[0].stage_seconds:
            result.data["metrics"][name] = max(
                reply.stage_seconds[name] for reply in replies
            )
        result.data["metrics"].update(
            timing_mode="cuda_event_rank_max",
            rank_worker_seconds=[
                reply.stage_seconds["worker_seconds"] for reply in replies
            ],
            sp_degree=len(self.workers) // self.tp_size,
            tp_degree=self.tp_size,
            sp_mode=self.sp_mode,
            dispatch_seconds=dispatch_seconds,
            rank_cache_bytes=[reply.usage.bytes for reply in replies],
        )
        logger.info(
            "Realization parallel block: %s",
            dict(
                session=str(context.session_identity),
                block=chunk.seq,
                **result.data["metrics"],
            ),
        )
        return result

    def close(self, session_identity: SessionIdentity) -> None:
        if self.workers:
            operation = SessionOperation("close", session_identity, ("realization",))
            self.exchange(operation, None)
        else:
            pass
        self.usages.pop(session_identity, None)

    def usage(self, session_identity: SessionIdentity) -> ResourceUsage:
        return self.usages.get(session_identity, ResourceUsage())

    def shutdown(self, *, abort: bool = False) -> None:
        if not abort:
            for connection in self.connections:
                try:
                    connection.send(None)
                except (BrokenPipeError, EOFError):
                    pass
            for worker in self.workers:
                worker.join(timeout=5)
        else:
            pass
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()
            else:
                pass
        for worker in self.workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.kill()
                worker.join()
            else:
                pass
        self.workers.clear()
        for connection in self.connections:
            connection.close()
        self.connections.clear()
        self.rendezvous.cleanup()
