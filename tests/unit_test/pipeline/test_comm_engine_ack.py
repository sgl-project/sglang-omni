# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest
import torch

from sglang_omni.comm.data_ref import DataRef, TransportKind
from sglang_omni.comm.engine import CommEngine
from sglang_omni.comm.router import CommRouter
from sglang_omni.proto import DataAckMessage, DataReadyMessage
from tests.unit_test.fixtures.pipeline_fakes import (
    FakeRelay,
    RecordingStageControlPlane,
    make_stage_payload,
    wait_until,
)


def test_comm_engine_releases_sender_op_after_data_ack() -> None:
    async def _run() -> None:
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        engine = CommEngine(
            CommRouter(
                stage_name="sender",
                gpu_id=None,
                same_process_targets=set(),
                gpu_stage_names=set(),
                comm_config={"ack_timeout_s": 1.0},
                injected_relay=relay,
            )
        )
        payload = make_stage_payload(request_id="req-1", data={"x": torch.ones(2)})

        data_ref = await engine.send_payload(
            relay=relay,
            control_plane=control_plane,
            request_id="req-1",
            payload=payload,
            transport=TransportKind.SHM,
            from_stage="sender",
            to_stage="receiver",
            target_endpoint="inproc://receiver",
        )

        op = relay.ops[0]
        assert not op.waited
        target, _, msg = control_plane.sent_to_stage[0]
        assert target == "receiver"
        assert relay.receiver_ids == ["receiver"]
        assert DataRef.from_dict(msg.data_ref).object_id == data_ref.object_id

        engine.ack_transfer(
            DataAckMessage(
                request_id="req-1",
                from_stage="receiver",
                to_stage="sender",
                object_id=data_ref.object_id,
            )
        )
        await wait_until(lambda: op.waited)
        assert op.acked

    asyncio.run(_run())


def test_comm_engine_ignores_unknown_data_ack() -> None:
    engine = CommEngine(
        CommRouter(
            stage_name="sender",
            gpu_id=None,
            same_process_targets=set(),
            gpu_stage_names=set(),
        )
    )

    engine.ack_transfer(
        DataAckMessage(
            request_id="req-1",
            from_stage="receiver",
            to_stage="sender",
            object_id="missing",
        )
    )


def test_comm_engine_ignores_duplicate_data_ack() -> None:
    async def _run() -> None:
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        engine = CommEngine(
            CommRouter(
                stage_name="sender",
                gpu_id=None,
                same_process_targets=set(),
                gpu_stage_names=set(),
                comm_config={"ack_timeout_s": 1.0},
                injected_relay=relay,
            )
        )
        payload = make_stage_payload(request_id="req-1", data={"x": torch.ones(2)})
        data_ref = await engine.send_payload(
            relay=relay,
            control_plane=control_plane,
            request_id="req-1",
            payload=payload,
            transport=TransportKind.SHM,
            from_stage="sender",
            to_stage="receiver",
            target_endpoint="inproc://receiver",
        )
        ack = DataAckMessage(
            request_id="req-1",
            from_stage="receiver",
            to_stage="sender",
            object_id=data_ref.object_id,
        )

        engine.ack_transfer(ack)
        await wait_until(lambda: relay.ops[0].waited)

        engine.ack_transfer(ack)

    asyncio.run(_run())


def test_data_messages_reject_missing_data_ref() -> None:
    with pytest.raises(TypeError, match="data_ref"):
        DataReadyMessage(
            request_id="req-1",
            from_stage="a",
            to_stage="b",
            data_ref=None,
        ).to_dict()

    with pytest.raises(TypeError, match="success"):
        DataAckMessage.from_dict(
            {
                "type": "data_ack",
                "request_id": "req-1",
                "from_stage": "b",
                "to_stage": "a",
                "object_id": "obj",
            }
        )

    with pytest.raises(TypeError, match="is_done"):
        DataReadyMessage.from_dict(
            {
                "type": "data_ready",
                "request_id": "req-1",
                "from_stage": "a",
                "to_stage": "b",
                "is_done": "false",
            }
        )

    with pytest.raises(TypeError, match="chunk_id"):
        DataReadyMessage.from_dict(
            {
                "type": "data_ready",
                "request_id": "req-1",
                "from_stage": "a",
                "to_stage": "b",
                "data_ref": {"version": 1},
                "chunk_id": True,
            }
        )

    with pytest.raises(ValueError, match="both done and error"):
        DataReadyMessage(
            request_id="req-1",
            from_stage="a",
            to_stage="b",
            data_ref=None,
            is_done=True,
            error="boom",
        ).to_dict()


def test_data_ref_rejects_bool_int_fields() -> None:
    data_ref = {
        "_type": "DataRef",
        "version": 1,
        "kind": "stream_chunk",
        "object_id": "obj",
        "transport": "shm",
        "layout": "raw_tensor",
        "buffer": {"transport": "shm", "info": {}, "length": 1},
        "tensors": [],
        "metadata_tensors": [],
        "shape": [1],
        "dtype": "torch.uint8",
        "offset": 0,
    }

    data_ref["version"] = True
    with pytest.raises(TypeError, match="version must be int"):
        DataRef.from_dict(data_ref)

    data_ref["version"] = 1
    data_ref["shape"] = [True]
    with pytest.raises(TypeError, match="shape must be list\\[int\\]"):
        DataRef.from_dict(data_ref)
