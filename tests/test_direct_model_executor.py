# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from sglang_omni.config.compiler import _wire_stream_targets
from sglang_omni.config.schema import ExecutorConfig, StageConfig, StreamTargetConfig
from sglang_omni.executors.direct_model_executor import DirectModelExecutor


class _EchoModel(torch.nn.Module):
    def forward(self, **kwargs):
        del kwargs
        return {"tensor": torch.tensor([7]), "metadata": {"kind": "chunk"}}


class _DummyReceiverExecutor:
    def __init__(self):
        self._stream_queue = None


class _DummyWorker:
    def __init__(self, executor):
        self.executor = executor
        self._stream_targets = []
        self._bootstrap_targets = set()
        self._same_gpu_targets = set()
        self.emitted = []

    def _enqueue_stream(self, request_id, data, target_stage, metadata=None):
        self.emitted.append((request_id, data, target_stage, metadata))


class _DummyStage:
    def __init__(self, name: str, workers: list[_DummyWorker]):
        self.name = name
        self.workers = workers
        self._stream_queue = None


def test_wire_stream_targets_sets_direct_executor_target_stage() -> None:
    sender_executor = DirectModelExecutor(
        model=_EchoModel(),
        device="cpu",
        request_builder=lambda payload: {"payload": payload.request_id},
        result_builder=lambda payload, output: payload,
        streaming=True,
    )
    sender_worker = _DummyWorker(sender_executor)
    receiver_worker = _DummyWorker(_DummyReceiverExecutor())
    sender_stage = _DummyStage("sender", [sender_worker])
    receiver_stage = _DummyStage("receiver", [receiver_worker])

    sender_cfg = StageConfig(
        name="sender",
        executor=ExecutorConfig(factory="tests.sender", args={}),
        get_next="tests.next",
        stream_to=[StreamTargetConfig(to_stage="receiver")],
    )

    _wire_stream_targets(
        sender_stage=sender_stage,
        sender_cfg=sender_cfg,
        stage_map={"sender": sender_stage, "receiver": receiver_stage},
    )

    assert sender_executor._target_stage == "receiver"
    assert sender_executor._stream_fn is not None
    assert receiver_stage._stream_queue is not None
    assert receiver_worker.executor._stream_queue is receiver_stage._stream_queue
