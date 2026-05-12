# SPDX-License-Identifier: Apache-2.0
"""Small test doubles for the V1 pipeline unit tests.

These fakes model the public contracts between Coordinator, Stage, Scheduler,
and Relay. They intentionally avoid real ZMQ, CUDA, SGLang, and relay backends.
"""

from __future__ import annotations

import asyncio
import queue
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from sglang_omni_v1.proto import OmniRequest, StagePayload
from sglang_omni_v1.scheduling.messages import IncomingMessage, OutgoingMessage


@dataclass
class EventLog:
    events: list[tuple[Any, ...]] = field(default_factory=list)

    def append(self, *event: Any) -> None:
        self.events.append(event)


class FakeOp:
    def __init__(self, metadata: dict[str, Any], log: EventLog | None = None):
        self._metadata = metadata
        self.log = log or EventLog()
        self.waited = False

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        del timeout
        self.waited = True
        self.log.append("op_wait", self._metadata.get("key"))


class FakeRelay:
    def __init__(self, *, device: str = "cpu", log: EventLog | None = None):
        self.device = device
        self.log = log or EventLog()
        self.storage: dict[str, torch.Tensor] = {}
        self.cleaned: list[str] = []
        self.closed = False
        self.fail_get: BaseException | None = None

    async def put_async(
        self,
        tensor: torch.Tensor,
        request_id: str | None = None,
        dst_rank: int | None = None,
    ) -> FakeOp:
        del dst_rank
        key = str(request_id)
        stored = tensor.detach().clone()
        self.storage[key] = stored
        self.log.append("relay_put", key, int(stored.numel()))
        return FakeOp(
            {"transfer_info": {"size": int(stored.numel())}, "key": key},
            self.log,
        )

    async def get_async(
        self,
        metadata: dict[str, Any],
        dest_tensor: torch.Tensor,
        request_id: str | None = None,
    ) -> FakeOp:
        if self.fail_get is not None:
            raise self.fail_get
        key = str(metadata.get("key", request_id))
        stored = self.storage[key]
        dest_tensor.reshape(-1)[: stored.numel()].copy_(stored.reshape(-1))
        self.log.append("relay_get", key, int(stored.numel()))
        return FakeOp(metadata, self.log)

    def cleanup(self, request_id: str) -> None:
        self.cleaned.append(request_id)
        self.log.append("relay_cleanup", request_id)

    def close(self) -> None:
        self.closed = True
        self.log.append("relay_close")


class FakeScheduler:
    def __init__(self, *, fail_start: BaseException | None = None):
        self.inbox: queue.Queue[IncomingMessage] = queue.Queue()
        self.outbox: queue.Queue[OutgoingMessage] = queue.Queue()
        self.fail_start = fail_start
        self.started = False
        self.stopped = False
        self.aborted: list[str] = []

    def start(self) -> None:
        self.started = True
        if self.fail_start is not None:
            raise self.fail_start

    def stop(self) -> None:
        self.stopped = True

    def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class RecordingStageControlPlane:
    def __init__(self, recv_endpoint: str = "inproc://stage"):
        self.recv_endpoint = recv_endpoint
        self.inbox: asyncio.Queue[Any] = asyncio.Queue()
        self.abort_inbox: asyncio.Queue[Any] = asyncio.Queue()
        self.sent_to_stage: list[tuple[str, str, Any]] = []
        self.completions: list[Any] = []
        self.started = False
        self.closed = False
        self.log = EventLog()

    async def start(self) -> None:
        self.started = True
        self.log.append("stage_cp_start")

    async def recv(self) -> Any:
        return await self.inbox.get()

    async def recv_abort(self) -> Any:
        return await self.abort_inbox.get()

    async def send_to_stage(self, target: str, endpoint: str, msg: Any) -> None:
        self.sent_to_stage.append((target, endpoint, msg))
        self.log.append("stage_cp_send_to_stage", target, type(msg).__name__)

    async def send_complete(self, msg: Any) -> None:
        self.completions.append(msg)
        self.log.append("stage_cp_send_complete", msg.request_id, msg.success)

    def close(self) -> None:
        self.closed = True
        self.log.append("stage_cp_close")


class RecordingCoordinatorControlPlane:
    def __init__(self):
        self.events: asyncio.Queue[Any] = asyncio.Queue()
        self.submitted: list[tuple[str, str, Any]] = []
        self.aborts: list[Any] = []
        self.shutdowns: list[tuple[str, str]] = []
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def submit_to_stage(self, stage: str, endpoint: str, msg: Any) -> None:
        self.submitted.append((stage, endpoint, msg))

    async def broadcast_abort(self, msg: Any) -> None:
        self.aborts.append(msg)

    async def send_shutdown(self, stage: str, endpoint: str) -> None:
        self.shutdowns.append((stage, endpoint))

    async def recv_event(self) -> Any:
        return await self.events.get()

    def close(self) -> None:
        self.closed = True


class FakeMpContext:
    def Queue(self) -> queue.Queue:
        return queue.Queue()


def make_scheduler(**_: Any) -> FakeScheduler:
    return FakeScheduler()


def make_scheduler_accepting_model_path(
    model_path: str, **kwargs: Any
) -> FakeScheduler:
    scheduler = FakeScheduler()
    scheduler.model_path = model_path
    scheduler.factory_kwargs = kwargs
    return scheduler


def make_scheduler_accepting_gpu_id(gpu_id: int = -1, **kwargs: Any) -> FakeScheduler:
    scheduler = FakeScheduler()
    scheduler.gpu_id = gpu_id
    scheduler.factory_kwargs = kwargs
    return scheduler


def merge_payloads(payloads: dict[str, StagePayload]) -> StagePayload:
    first = next(iter(payloads.values()))
    return StagePayload(
        request_id=first.request_id,
        request=first.request,
        data={
            "merged_sources": sorted(payloads),
            "values": {name: payload.data for name, payload in payloads.items()},
        },
    )


def project_payload(payload: StagePayload) -> StagePayload:
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data={"projected": payload.data},
    )


def make_stage_payload(
    data: Any | None = None,
    *,
    request_id: str = "req-1",
    inputs: Any | None = None,
    params: dict[str, Any] | None = None,
) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(
            inputs={"text": "hello"} if inputs is None else inputs,
            params=params or {},
        ),
        data={} if data is None else data,
    )


def make_tensor_payload(request_id: str = "req-tensor") -> StagePayload:
    return make_stage_payload(
        request_id=request_id,
        data={
            "float": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "nested": {
                "long": torch.tensor([1, 2, 3], dtype=torch.int64),
                "tuple": (torch.tensor([True, False]), "kept"),
            },
            "list": [torch.tensor([[1.5]], dtype=torch.float16)],
            "plain": "value",
        },
    )


def make_result_message(
    request_id: str = "req-1",
    data: Any | None = None,
    *,
    target: str | None = None,
) -> OutgoingMessage:
    return OutgoingMessage(
        request_id=request_id,
        type="result",
        data=make_stage_payload(
            request_id=request_id,
            data={"result": "ok"} if data is None else data,
        ),
        target=target,
    )


def make_error_message(
    request_id: str = "req-1", error: BaseException | str = "boom"
) -> OutgoingMessage:
    return OutgoingMessage(request_id=request_id, type="error", data=error)


def make_stream_message(
    request_id: str = "req-1",
    data: Any | None = None,
    *,
    target: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> OutgoingMessage:
    return OutgoingMessage(
        request_id=request_id,
        type="stream",
        data=torch.tensor([1, 2, 3]) if data is None else data,
        target=target,
        metadata=metadata,
    )


def fake_factory_path(name: str) -> str:
    return f"tests.unit_test.fixtures.pipeline_fakes.{name}"


def tensor_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            tensor_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            tensor_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def drain_queue(q: queue.Queue) -> list[Any]:
    items: list[Any] = []
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            return items


def collect_event_names(log: EventLog) -> list[Any]:
    return [event[0] for event in log.events]


def make_noop_projector(marker: str) -> Callable[[StagePayload], StagePayload]:
    def _project(payload: StagePayload) -> StagePayload:
        return make_stage_payload(
            request_id=payload.request_id,
            inputs=payload.request.inputs,
            params=payload.request.params,
            data={"marker": marker, "data": payload.data},
        )

    return _project
