# SPDX-License-Identifier: Apache-2.0
"""CPU transport coverage for the existing metrics admin control channel."""

import asyncio
import multiprocessing
from multiprocessing.connection import Connection
from types import SimpleNamespace

import zmq
from prometheus_client import CollectorRegistry, Counter

from sglang_omni.pipeline.control_plane import (
    ControlPlaneContext,
    CoordinatorControlPlane,
    StageControlPlane,
)
from sglang_omni.proto import (
    AdminMessage,
    AdminOperation,
    AdminResult,
    AdminResultMessage,
)
from sglang_omni.scheduling.metrics_snapshot import backend_metrics_snapshot


def metrics_worker(
    connection: Connection, completion_endpoint: str, abort_endpoint: str
) -> None:

    async def run() -> None:
        plane = StageControlPlane(
            "decode", "tcp://127.0.0.1:*", completion_endpoint, abort_endpoint
        )
        try:
            await plane.start()
            connection.send(
                plane.recv_socket.socket.getsockopt_string(zmq.LAST_ENDPOINT)
            )
            message = await asyncio.wait_for(plane.recv(), 15)
            assert isinstance(message, AdminMessage)
            assert message.operation.action == "metrics"
            counter = Counter(
                "sglang:evicted_tokens_total", "Eviction", registry=CollectorRegistry()
            )
            counter.inc(7)
            scheduler = SimpleNamespace(
                enable_metrics=True,
                metrics_reporter=SimpleNamespace(
                    metrics_collector=SimpleNamespace(counter=counter)
                ),
            )
            await plane.send_admin_result(
                AdminResultMessage(
                    AdminResult(
                        message.operation.op_id,
                        "decode",
                        "metrics",
                        True,
                        data={
                            "num_running_reqs": 1,
                            "num_queue_reqs": 2,
                            "backend_metrics": backend_metrics_snapshot(scheduler),
                        },
                    )
                )
            )
        finally:
            plane.close()
            ControlPlaneContext.close()
            connection.close()

    loop = asyncio.SelectorEventLoop()
    try:
        loop.run_until_complete(run())
    finally:
        loop.close()


def test_metrics_admin_round_trip_over_tcp_to_spawned_worker() -> None:
    async def run() -> None:
        plane = CoordinatorControlPlane("tcp://127.0.0.1:*", "tcp://127.0.0.1:*")
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = None
        try:
            await plane.start()
            completion = plane.completion_socket.socket.getsockopt_string(
                zmq.LAST_ENDPOINT
            )
            abort = plane.abort_socket.socket.getsockopt_string(zmq.LAST_ENDPOINT)
            process = context.Process(
                target=metrics_worker, args=(sender, completion, abort)
            )
            process.start()
            sender.close()
            assert await asyncio.to_thread(receiver.poll, 15), "Worker did not bind"
            endpoint = receiver.recv()
            operation = AdminOperation(op_id="metrics-tcp", action="metrics")
            await plane.send_admin("decode", endpoint, AdminMessage(operation))
            message = await asyncio.wait_for(plane.recv_event(), 15)
            assert isinstance(message, AdminResultMessage)
            assert message.result.op_id == operation.op_id
            assert message.result.success
            assert message.result.data["num_running_reqs"] == 1
            assert message.result.data["num_queue_reqs"] == 2
            family = message.result.data["backend_metrics"][0]
            assert family["name"] == "sglang:evicted_tokens"
            assert family["type"] == "counter"
            assert family["samples"][0]["value"] == 7
            process.join(timeout=5)
            assert process.exitcode == 0
        finally:
            receiver.close()
            sender.close()
            if process is not None and process.is_alive():
                process.terminate()
                process.join(timeout=5)
            plane.close()
            ControlPlaneContext.close()

    loop = asyncio.SelectorEventLoop()
    try:
        loop.run_until_complete(run())
    finally:
        loop.close()
