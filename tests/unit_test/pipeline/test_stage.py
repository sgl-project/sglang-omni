# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import queue

import pytest
import torch

from sglang_omni.pipeline import relay_io
from sglang_omni.pipeline.local_dispatch import LocalStageDispatcher
from sglang_omni.pipeline.stage.input import AggregatedInput
from sglang_omni.pipeline.stage.stream_queue import StreamQueue
from sglang_omni.pipeline.stage_workers import StageLaunchConfig, _construct_stage
from sglang_omni.pipeline.tensor_ref import (
    TensorRef,
    TensorRefPolicy,
    is_tensor_ref_dict,
)
from sglang_omni.pipeline.tp_control import TPLeaderFanout
from sglang_omni.proto import DataReadyMessage
from tests.unit_test.fixtures.pipeline_fakes import (
    EventLog,
    FakeRelay,
    FakeScheduler,
    RecordingStageControlPlane,
    collect_event_names,
    fake_factory_path,
    make_noop_projector,
    make_result_message,
    make_stage_payload,
    make_stream_message,
    make_tensor_payload,
    tensor_equal,
)
from tests.unit_test.pipeline.helpers import make_stage


class _CloseAwareControlPlane(RecordingStageControlPlane):
    async def recv(self):
        while not self.closed:
            await asyncio.sleep(0)
        raise RuntimeError("control plane closed")


def test_aggregated_input_waits_per_request_without_cross_talk() -> None:
    """Preserves per-request fan-in isolation when requests interleave."""
    handler = AggregatedInput(
        {"preprocess", "image"},
        lambda payloads: make_stage_payload(data={"sources": sorted(payloads)}),
    )

    assert handler.receive("req-1", "preprocess", make_stage_payload()) is None
    assert handler.receive("req-2", "preprocess", make_stage_payload()) is None
    req2 = handler.receive("req-2", "image", make_stage_payload())
    req1 = handler.receive("req-1", "image", make_stage_payload())

    assert req2.data == {"sources": ["image", "preprocess"]}
    assert req1.data == {"sources": ["image", "preprocess"]}


def test_aggregated_input_supports_request_dynamic_source_sets() -> None:
    """Preserves early-arriving payloads while narrowing fan-in per request."""

    def _expected_sources(request_id, from_stage, payload):
        del request_id
        if from_stage != "preprocess":
            return None
        return payload.data["expected"]

    handler = AggregatedInput(
        {"preprocess", "image", "audio"},
        lambda payloads: make_stage_payload(data={"sources": sorted(payloads)}),
        expected_sources_fn=_expected_sources,
    )

    assert handler.receive("req-audio", "audio", make_stage_payload()) is None
    audio = handler.receive(
        "req-audio",
        "preprocess",
        make_stage_payload(data={"expected": ["preprocess", "audio"]}),
    )
    assert audio.data == {"sources": ["audio", "preprocess"]}

    text = handler.receive(
        "req-text",
        "preprocess",
        make_stage_payload(data={"expected": ["preprocess"]}),
    )
    assert text.data == {"sources": ["preprocess"]}


def test_aggregated_input_rejects_dynamic_sources_outside_static_fanin() -> None:
    def _invalid_sources(request_id, from_stage, payload):
        del request_id, from_stage, payload
        return ["preprocess", "audio"]

    handler = AggregatedInput(
        {"preprocess", "image"},
        lambda payloads: make_stage_payload(data={"sources": sorted(payloads)}),
        expected_sources_fn=_invalid_sources,
    )

    with pytest.raises(ValueError, match="outside static wait_for"):
        handler.receive("req-1", "preprocess", make_stage_payload())


def test_stage_routes_results_streams_and_clears_abort_state() -> None:
    """Preserves result routing, stream forwarding, and abort cleanup."""

    async def _run() -> None:
        relay = FakeRelay()
        scheduler = FakeScheduler()
        control_plane = RecordingStageControlPlane()
        stage_obj = make_stage(
            name="thinker",
            get_next=lambda request_id, output: "decode",
            endpoints={"decode": "inproc://decode", "talker": "inproc://talker"},
            project_payload={"decode": make_noop_projector("decode-only")},
            stream_targets=["talker"],
            relay=relay,
            scheduler=scheduler,
            control_plane=control_plane,
        )
        stage_obj._active_requests.add("req-1")
        scheduler.outbox.put(make_stream_message("req-1", data=torch.tensor([7])))
        scheduler.outbox.put(make_result_message("req-1", data={"answer": 1}))

        await stage_obj._drain_outbox()

        decode_msg = next(
            msg for target, _, msg in control_plane.sent_to_stage if target == "decode"
        )
        restored = await relay_io.read_payload(relay, "req-1", decode_msg.shm_metadata)
        assert restored.data == {"marker": "decode-only", "data": {"answer": 1}}
        stream_msg = next(
            msg
            for target, _, msg in control_plane.sent_to_stage
            if target == "talker" and msg.chunk_id == 0
        )
        assert stream_msg.chunk_id == 0

        stage_obj._stream_queue = StreamQueue()
        stage_obj._stream_queue.open("req-1")
        stage_obj._on_abort("req-1")

        assert "req-1" in stage_obj._aborted
        assert relay.cleaned[-1] == "req-1"
        assert scheduler.aborted == ["req-1"]
        assert not stage_obj._stream_queue.has("req-1")

    asyncio.run(_run())


def test_stage_process_rejects_dynamic_targets_outside_static_topology() -> None:
    spec = StageLaunchConfig(
        stage_name="thinker",
        factory=fake_factory_path("make_scheduler"),
        next_stages=["decode"],
        route_fn=fake_factory_path("route_to_undeclared_talker"),
        stream_targets=["decode"],
        stream_done_to_fn=fake_factory_path("stream_done_to_undeclared_talker"),
        recv_endpoint="inproc://thinker",
        coordinator_endpoint="inproc://coordinator",
        abort_endpoint="inproc://abort",
        stage_endpoints={
            "decode": "inproc://decode",
            "talker": "inproc://talker",
        },
        relay_config={"relay_type": "shm", "slot_size_mb": 1},
    )
    stage_obj = _construct_stage(spec, logging.getLogger(__name__))
    payload = make_stage_payload()

    with pytest.raises(ValueError, match="route_fn.*outside the static topology"):
        stage_obj.get_next("req-1", payload)

    with pytest.raises(
        ValueError, match="stream_done_to_fn.*outside the static topology"
    ):
        stage_obj.get_stream_done_targets("req-1", payload)


def test_stage_process_rejects_dynamic_wait_sources_outside_static_fanin() -> None:
    spec = StageLaunchConfig(
        stage_name="aggregate",
        factory=fake_factory_path("make_scheduler"),
        next_stages="decode",
        wait_for=["preprocess", "thinker"],
        wait_for_fn=fake_factory_path("wait_sources_to_undeclared_stage"),
        merge_fn=fake_factory_path("merge_payloads"),
        recv_endpoint="inproc://aggregate",
        coordinator_endpoint="inproc://coordinator",
        abort_endpoint="inproc://abort",
        stage_endpoints={"decode": "inproc://decode"},
        relay_config={"relay_type": "shm", "slot_size_mb": 1},
    )
    stage_obj = _construct_stage(spec, logging.getLogger(__name__))

    with pytest.raises(ValueError, match="outside static wait_for"):
        stage_obj.input_handler.receive("req-1", "preprocess", make_stage_payload())


def test_stage_process_accepts_iterable_dynamic_wait_sources() -> None:
    spec = StageLaunchConfig(
        stage_name="aggregate",
        factory=fake_factory_path("make_scheduler"),
        next_stages="decode",
        wait_for=["preprocess", "thinker"],
        wait_for_fn=fake_factory_path("tuple_wait_sources"),
        merge_fn=fake_factory_path("merge_payloads"),
        recv_endpoint="inproc://aggregate",
        coordinator_endpoint="inproc://coordinator",
        abort_endpoint="inproc://abort",
        stage_endpoints={"decode": "inproc://decode"},
        relay_config={"relay_type": "shm", "slot_size_mb": 1},
    )
    stage_obj = _construct_stage(spec, logging.getLogger(__name__))

    assert (
        stage_obj.input_handler.receive("req-1", "preprocess", make_stage_payload())
        is None
    )
    merged = stage_obj.input_handler.receive("req-1", "thinker", make_stage_payload())

    assert merged is not None
    assert merged.data["merged_sources"] == ["preprocess", "thinker"]


def test_stage_run_raises_when_scheduler_thread_crashes() -> None:
    async def _run() -> None:
        scheduler = FakeScheduler(fail_start=RuntimeError("boom"))
        stage_obj = make_stage(
            scheduler=scheduler,
            control_plane=_CloseAwareControlPlane(),
        )

        with pytest.raises(RuntimeError, match="Scheduler thread"):
            await asyncio.wait_for(stage_obj.run(), timeout=2.0)

        assert scheduler.stopped is True

    asyncio.run(_run())


def test_relay_payload_and_cross_gpu_stream_contracts() -> None:
    """Preserves tensor payload round-trips and stream control-before-wait ordering."""

    async def _run() -> None:
        relay = FakeRelay()
        payload = make_tensor_payload()
        metadata, op = await relay_io.write_payload(relay, payload.request_id, payload)
        await op.wait_for_completion()
        restored = await relay_io.read_payload(relay, payload.request_id, metadata)
        assert tensor_equal(restored.data, payload.data)

        log = EventLog()
        stream_relay = FakeRelay(log=log)
        control_plane = RecordingStageControlPlane()
        control_plane.log = log
        await relay_io.send_stream_chunk(
            stream_relay,
            control_plane,
            request_id="req-1",
            data=torch.tensor([1, 2, 3]),
            target_stage="talker",
            target_endpoint="inproc://talker",
            from_stage="thinker",
            chunk_id=0,
            metadata={"token_id": 1, "hidden": torch.tensor([4])},
        )

        names = collect_event_names(log)
        assert names.index("stage_cp_send_to_stage") < names.index("op_wait")
        msg = control_plane.sent_to_stage[0][2]
        assert msg.shm_metadata["chunk_metadata"]["token_id"] == 1
        assert "hidden" in msg.shm_metadata["chunk_metadata_tensors"]

    asyncio.run(_run())


def test_stage_execute_fanouts_unresolved_payload_before_materializing_refs(
    monkeypatch,
) -> None:
    """TP followers get the unresolved ref; the local scheduler gets the resolved tensor."""
    monkeypatch.setenv("SGLANG_OMNI_ENABLE_TENSOR_REFS", "1")

    async def _run() -> None:
        relay = FakeRelay()
        tensor = torch.arange(6, dtype=torch.float32)
        blob_metadata, blob_op = await relay_io.write_blob(relay, "req-1:blob", tensor)
        await blob_op.wait_for_completion()
        ref = TensorRef(
            ref_id="req-1:blob",
            request_id="req-1",
            producer_stage="image_encoder",
            consumer_stage="thinker",
            path="video_embeds",
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            nbytes=tensor.numel() * tensor.element_size(),
            blob_key="req-1:blob",
            blob_metadata=blob_metadata,
        )
        payload = make_stage_payload(
            request_id="req-1", data={"video_embeds": ref.to_dict()}
        )

        scheduler = FakeScheduler()
        scheduler.requires_tp_work_fanout = True
        follower_queue: queue.Queue = queue.Queue()
        tp_fanout = TPLeaderFanout(
            "thinker",
            follower_work_queues=[follower_queue],
            follower_abort_queues=[],
        )
        stage_obj = make_stage(
            name="thinker",
            role="leader",
            scheduler=scheduler,
            relay=relay,
            tp_fanout=tp_fanout,
        )

        await stage_obj._execute(payload)

        follower_msg = follower_queue.get_nowait()
        assert is_tensor_ref_dict(follower_msg.data.data["video_embeds"])

        scheduled = scheduler.inbox.get_nowait()
        assert torch.equal(scheduled.data.data["video_embeds"], tensor)

    asyncio.run(_run())


def test_send_to_stage_does_not_block_on_externalized_tensor_ref_blob(
    monkeypatch,
) -> None:
    """The small envelope op completes without waiting on the deferred blob op."""
    monkeypatch.setenv("SGLANG_OMNI_ENABLE_TENSOR_REFS", "1")
    monkeypatch.setenv(
        "SGLANG_OMNI_TENSOR_REF_EDGES", "image_encoder:mm_aggregate:thinker"
    )
    monkeypatch.setenv("SGLANG_OMNI_TENSOR_REF_PATHS", "big")
    monkeypatch.setenv("SGLANG_OMNI_TENSOR_REF_THRESHOLD_MB", "0")

    async def _run() -> None:
        gate = asyncio.Event()

        class GatedRelay(FakeRelay):
            async def put_async(self, tensor, request_id=None, dst_rank=None):
                op = await super().put_async(
                    tensor, request_id=request_id, dst_rank=dst_rank
                )
                if request_id and ":tensor_ref:" in request_id:
                    real_wait = op.wait_for_completion

                    async def gated_wait(timeout: float = 30.0):
                        await gate.wait()
                        await real_wait(timeout=timeout)

                    op.wait_for_completion = gated_wait
                return op

        relay = GatedRelay()
        stage_obj = make_stage(
            name="image_encoder",
            endpoints={"mm_aggregate": "inproc://mm"},
            relay=relay,
        )
        payload = make_stage_payload(request_id="req-1", data={"big": torch.randn(4)})

        await asyncio.wait_for(
            stage_obj._send_to_stage("req-1", "mm_aggregate", payload),
            timeout=1.0,
        )

        assert len(relay_io._BACKGROUND_REF_TASKS) == 1
        task = next(iter(relay_io._BACKGROUND_REF_TASKS))
        gate.set()
        await asyncio.wait_for(task, timeout=1.0)
        await asyncio.sleep(0)
        assert relay_io._BACKGROUND_REF_TASKS == set()

    asyncio.run(_run())


def test_stage_discards_aborted_tensor_ref_payload_releases_blob() -> None:
    async def _run() -> None:
        relay = FakeRelay()
        tensor = torch.arange(8, dtype=torch.float32)
        payload = make_stage_payload(request_id="req-1", data={"video_embeds": tensor})
        policy = TensorRefPolicy(
            threshold_bytes=1,
            from_stage="image_encoder",
            to_stage="mm_aggregate",
            consumer_stage="thinker",
            path_allowlist=("video_embeds",),
        )
        metadata, op = await relay_io.write_payload(
            relay,
            payload.request_id,
            payload,
            from_stage="image_encoder",
            to_stage="mm_aggregate",
            tensor_ref_policy=policy,
        )
        await op.wait_for_completion()
        blob_key = metadata["tensor_ref_blobs"][0]["blob_key"]
        assert blob_key in relay.storage

        stage_obj = make_stage(name="mm_aggregate", relay=relay)
        stage_obj._record_aborted_request_id("req-1")
        await stage_obj._on_data_ready(
            DataReadyMessage("req-1", "image_encoder", "mm_aggregate", metadata)
        )

        assert blob_key not in relay.storage

        for task in list(relay_io._BACKGROUND_REF_TASKS):
            await task

    asyncio.run(_run())


def test_stage_abort_releases_tracked_unresolved_tensor_ref_payload() -> None:
    async def _run() -> None:
        relay = FakeRelay()
        tensor = torch.arange(8, dtype=torch.float32)
        payload = make_stage_payload(request_id="req-1", data={"video_embeds": tensor})
        policy = TensorRefPolicy(
            threshold_bytes=1,
            from_stage="image_encoder",
            to_stage="mm_aggregate",
            consumer_stage="thinker",
            path_allowlist=("video_embeds",),
        )
        metadata, op = await relay_io.write_payload(
            relay,
            payload.request_id,
            payload,
            from_stage="image_encoder",
            to_stage="mm_aggregate",
            tensor_ref_policy=policy,
        )
        await op.wait_for_completion()
        blob_key = metadata["tensor_ref_blobs"][0]["blob_key"]
        mid_payload = await relay_io.read_payload(relay, payload.request_id, metadata)
        assert blob_key in relay.storage

        stage_obj = make_stage(name="mm_aggregate", relay=relay)
        await stage_obj._receive_payload_from_stage(
            "req-1", "image_encoder", mid_payload
        )
        stage_obj._on_abort("req-1")

        assert blob_key not in relay.storage

        for task in list(relay_io._BACKGROUND_REF_TASKS):
            await task

    asyncio.run(_run())


def test_stage_abort_preserves_tracked_refs_across_ref_free_fanin(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_OMNI_ENABLE_TENSOR_REFS", "1")

    async def _run() -> None:
        relay = FakeRelay()
        tensor = torch.arange(8, dtype=torch.float32)
        blob_metadata, blob_op = await relay_io.write_blob(relay, "req-1:blob", tensor)
        await blob_op.wait_for_completion()
        ref = TensorRef(
            ref_id="req-1:blob",
            request_id="req-1",
            producer_stage="image_encoder",
            consumer_stage="thinker",
            path="video_embeds",
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            nbytes=tensor.numel() * tensor.element_size(),
            blob_key="req-1:blob",
            blob_metadata=blob_metadata,
        )

        def _merge(payloads):
            image_payload = payloads["image_encoder"]
            return make_stage_payload(
                request_id="req-1",
                data={"video_embeds": image_payload.data["video_embeds"]},
            )

        stage_obj = make_stage(
            name="mm_aggregate",
            relay=relay,
            input_handler=AggregatedInput({"image_encoder", "preprocessing"}, _merge),
        )
        started_materialize = asyncio.Event()
        release_materialize = asyncio.Event()
        original_materialize = relay_io.materialize_payload_tensor_refs

        async def _blocking_materialize(relay_arg, payload, *, current_stage):
            started_materialize.set()
            await release_materialize.wait()
            return await original_materialize(
                relay_arg, payload, current_stage=current_stage
            )

        monkeypatch.setattr(
            relay_io, "materialize_payload_tensor_refs", _blocking_materialize
        )
        await stage_obj._receive_payload_from_stage(
            "req-1",
            "image_encoder",
            make_stage_payload(
                request_id="req-1", data={"video_embeds": ref.to_dict()}
            ),
        )
        assert "req-1:blob" in relay.storage

        second_payload = asyncio.create_task(
            stage_obj._receive_payload_from_stage(
                "req-1",
                "preprocessing",
                make_stage_payload(request_id="req-1", data={"plain": 1}),
            )
        )
        await asyncio.wait_for(started_materialize.wait(), timeout=2.0)

        stage_obj._on_abort("req-1")
        assert "req-1:blob" not in relay.storage

        release_materialize.set()
        await asyncio.wait_for(second_payload, timeout=2.0)

    asyncio.run(_run())


def test_stage_relay_read_failure_completes_with_error() -> None:
    """Preserves failure reporting when a stage cannot read its relay payload."""

    async def _run() -> None:
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        stage_obj = make_stage(relay=relay, control_plane=control_plane)
        payload = make_stage_payload(request_id="req-1")
        metadata, _ = await relay_io.write_payload(relay, "req-1", payload)
        relay.fail_get = RuntimeError("read failed")

        await stage_obj._on_data_ready(
            DataReadyMessage("req-1", "upstream", "stage", metadata)
        )

        assert control_plane.completions[0].success is False
        assert "relay read failed" in control_plane.completions[0].error
        assert relay.cleaned[-1] == "req-1"

    asyncio.run(_run())


def test_stage_uses_dynamic_route_and_stream_done_targets() -> None:
    async def _run() -> None:
        control_plane = RecordingStageControlPlane()
        stage_obj = make_stage(
            control_plane=control_plane,
            endpoints={"decode": "inproc://decode", "talker": "inproc://talker"},
            get_next=lambda request_id, output: output.request.metadata["next"],
            stream_targets=["talker", "decode"],
            get_stream_done_targets=lambda request_id, output: output.request.metadata[
                "stream_targets"
            ],
        )
        payload = make_stage_payload(request_id="req-1")
        payload.request.metadata["next"] = "decode"
        payload.request.metadata["stream_targets"] = ["decode"]
        stage_obj._active_requests.add("req-1")

        await stage_obj._route_result("req-1", payload)

        stream_done_target, _, stream_done_msg = control_plane.sent_to_stage[0]
        routed_target, _, routed_msg = control_plane.sent_to_stage[1]
        assert stream_done_target == "decode"
        assert isinstance(stream_done_msg, DataReadyMessage)
        assert stream_done_msg.is_done
        assert routed_target == "decode"
        assert isinstance(routed_msg, DataReadyMessage)
        assert not routed_msg.is_done

    asyncio.run(_run())


def test_stage_sends_same_process_payload_as_local_object(monkeypatch) -> None:
    events: list[dict] = []
    monkeypatch.setattr(
        "sglang_omni.pipeline.stage.runtime._emit_event",
        lambda **kwargs: events.append(kwargs),
    )

    async def _run() -> None:
        dispatcher = LocalStageDispatcher()
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        receiver_scheduler = FakeScheduler()
        receiver = make_stage(name="decode", scheduler=receiver_scheduler)
        sender = make_stage(
            name="thinker",
            endpoints={"decode": "inproc://decode"},
            relay=relay,
            control_plane=control_plane,
            same_process_targets={"decode"},
            local_dispatcher=dispatcher,
        )
        dispatcher.register_many([sender, receiver])

        tensor = torch.arange(4)
        payload = make_stage_payload(request_id="req-local", data={"tensor": tensor})

        await sender._send_to_stage(
            "req-local",
            "decode",
            payload,
            allow_local_object=True,
        )

        assert relay.storage == {}
        assert control_plane.sent_to_stage == []
        queued = receiver_scheduler.inbox.get_nowait()
        assert queued.type == "new_request"
        assert queued.data is payload
        assert queued.data.data["tensor"] is tensor

    asyncio.run(_run())

    hop_events = [event for event in events if event["event_name"] == "stage_hop_sent"]
    assert hop_events == [
        {
            "request_id": "req-local",
            "stage": "thinker",
            "event_name": "stage_hop_sent",
            "metadata": {"to_stage": "decode", "transport": "local_object"},
        }
    ]


def test_stage_applies_projector_before_local_object_send() -> None:
    async def _run() -> None:
        dispatcher = LocalStageDispatcher()
        receiver_scheduler = FakeScheduler()
        receiver = make_stage(name="decode", scheduler=receiver_scheduler)
        sender = make_stage(
            name="thinker",
            endpoints={"decode": "inproc://decode"},
            project_payload={"decode": make_noop_projector("decode-only")},
            same_process_targets={"decode"},
            local_dispatcher=dispatcher,
        )
        dispatcher.register_many([sender, receiver])

        await sender._send_to_stage(
            "req-local",
            "decode",
            make_stage_payload(request_id="req-local", data={"answer": 7}),
            allow_local_object=True,
        )

        queued = receiver_scheduler.inbox.get_nowait()
        assert queued.data.data == {
            "marker": "decode-only",
            "data": {"answer": 7},
        }

    asyncio.run(_run())


def test_stage_local_object_preserves_fan_in_semantics() -> None:
    async def _run() -> None:
        dispatcher = LocalStageDispatcher()
        receiver_scheduler = FakeScheduler()
        receiver = make_stage(
            name="aggregate",
            scheduler=receiver_scheduler,
            input_handler=AggregatedInput(
                {"preprocess", "thinker"},
                lambda payloads: make_stage_payload(
                    request_id="req-local",
                    data={
                        "sources": sorted(payloads),
                        "values": {
                            name: payload.data for name, payload in payloads.items()
                        },
                    },
                ),
            ),
        )
        preprocess = make_stage(
            name="preprocess",
            endpoints={"aggregate": "inproc://aggregate"},
            same_process_targets={"aggregate"},
            local_dispatcher=dispatcher,
        )
        thinker = make_stage(
            name="thinker",
            endpoints={"aggregate": "inproc://aggregate"},
            same_process_targets={"aggregate"},
            local_dispatcher=dispatcher,
        )
        dispatcher.register(receiver)

        await preprocess._send_to_stage(
            "req-local",
            "aggregate",
            make_stage_payload(request_id="req-local", data={"p": 1}),
            allow_local_object=True,
        )
        assert receiver_scheduler.inbox.empty()

        await thinker._send_to_stage(
            "req-local",
            "aggregate",
            make_stage_payload(request_id="req-local", data={"t": 2}),
            allow_local_object=True,
        )

        queued = receiver_scheduler.inbox.get_nowait()
        assert queued.type == "new_request"
        assert queued.data.data["sources"] == ["preprocess", "thinker"]
        assert queued.data.data["values"] == {
            "preprocess": {"p": 1},
            "thinker": {"t": 2},
        }

    asyncio.run(_run())


def test_stage_fan_out_payloads_fall_back_to_relay() -> None:
    async def _run() -> None:
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        sender = make_stage(
            name="thinker",
            get_next=lambda request_id, output: ["decode", "archive"],
            endpoints={
                "decode": "inproc://decode",
                "archive": "inproc://archive",
            },
            relay=relay,
            control_plane=control_plane,
            same_process_targets={"decode", "archive"},
        )

        await sender._route_result(
            "req-fanout",
            make_stage_payload(request_id="req-fanout", data={"answer": 7}),
        )

        assert [target for target, _, _ in control_plane.sent_to_stage] == [
            "decode",
            "archive",
        ]
        assert control_plane.sent_to_stage[0][2].chunk_id is None
        assert control_plane.sent_to_stage[1][2].chunk_id is None

    asyncio.run(_run())


def test_stage_projected_fan_out_payloads_use_local_object_when_isolated() -> None:
    def _isolated_projector(marker):
        def _project(payload):
            return make_stage_payload(
                request_id=payload.request_id,
                inputs=payload.request.inputs,
                params=payload.request.params,
                data={"marker": marker, "data": dict(payload.data)},
            )

        return _project

    async def _run() -> None:
        dispatcher = LocalStageDispatcher()
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        decode_scheduler = FakeScheduler()
        archive_scheduler = FakeScheduler()
        decode = make_stage(name="decode", scheduler=decode_scheduler)
        archive = make_stage(name="archive", scheduler=archive_scheduler)
        sender = make_stage(
            name="thinker",
            get_next=lambda request_id, output: ["decode", "archive"],
            endpoints={
                "decode": "inproc://decode",
                "archive": "inproc://archive",
            },
            relay=relay,
            control_plane=control_plane,
            project_payload={
                "decode": _isolated_projector("decode-only"),
                "archive": _isolated_projector("archive-only"),
            },
            same_process_targets={"decode", "archive"},
            local_dispatcher=dispatcher,
        )
        dispatcher.register_many([sender, decode, archive])

        await sender._route_result(
            "req-fanout",
            make_stage_payload(request_id="req-fanout", data={"answer": 7}),
        )

        assert relay.storage == {}
        assert control_plane.sent_to_stage == []
        decode_msg = decode_scheduler.inbox.get_nowait()
        archive_msg = archive_scheduler.inbox.get_nowait()
        assert decode_msg.data.data == {
            "marker": "decode-only",
            "data": {"answer": 7},
        }
        assert archive_msg.data.data == {
            "marker": "archive-only",
            "data": {"answer": 7},
        }

    asyncio.run(_run())


def test_stage_projected_fan_out_requires_isolated_data_container() -> None:
    def _shared_data_projector(payload):
        return make_stage_payload(
            request_id=payload.request_id,
            inputs=payload.request.inputs,
            params=payload.request.params,
            data=payload.data,
        )

    async def _run() -> None:
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        sender = make_stage(
            name="thinker",
            get_next=lambda request_id, output: ["decode", "archive"],
            endpoints={
                "decode": "inproc://decode",
                "archive": "inproc://archive",
            },
            relay=relay,
            control_plane=control_plane,
            project_payload={
                "decode": _shared_data_projector,
                "archive": _shared_data_projector,
            },
            same_process_targets={"decode", "archive"},
            local_dispatcher=LocalStageDispatcher(),
        )

        await sender._route_result(
            "req-fanout",
            make_stage_payload(request_id="req-fanout", data={"answer": 7}),
        )

        assert [target for target, _, _ in control_plane.sent_to_stage] == [
            "decode",
            "archive",
        ]
        assert relay.storage

    asyncio.run(_run())


def test_stage_projected_fan_out_rejects_nested_mutable_aliases() -> None:
    def _shallow_copy_projector(payload):
        return make_stage_payload(
            request_id=payload.request_id,
            inputs=payload.request.inputs,
            params=payload.request.params,
            data={"projected": dict(payload.data)},
        )

    async def _run() -> None:
        dispatcher = LocalStageDispatcher()
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        decode = make_stage(name="decode", scheduler=FakeScheduler())
        archive = make_stage(name="archive", scheduler=FakeScheduler())
        sender = make_stage(
            name="thinker",
            get_next=lambda request_id, output: ["decode", "archive"],
            endpoints={
                "decode": "inproc://decode",
                "archive": "inproc://archive",
            },
            relay=relay,
            control_plane=control_plane,
            project_payload={
                "decode": _shallow_copy_projector,
                "archive": _shallow_copy_projector,
            },
            same_process_targets={"decode", "archive"},
            local_dispatcher=dispatcher,
        )
        dispatcher.register_many([sender, decode, archive])

        await sender._route_result(
            "req-fanout",
            make_stage_payload(
                request_id="req-fanout",
                data={"nested": {"tokens": [1, 2, 3]}, "answer": 7},
            ),
        )

        assert [target for target, _, _ in control_plane.sent_to_stage] == [
            "decode",
            "archive",
        ]
        assert relay.storage

    asyncio.run(_run())


def test_stage_projected_fan_out_rejects_wrapped_original_data() -> None:
    def _wrapped_data_projector(payload):
        return make_stage_payload(
            request_id=payload.request_id,
            inputs=payload.request.inputs,
            params=payload.request.params,
            data={"projected": payload.data},
        )

    async def _run() -> None:
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        sender = make_stage(
            name="thinker",
            get_next=lambda request_id, output: ["decode", "archive"],
            endpoints={
                "decode": "inproc://decode",
                "archive": "inproc://archive",
            },
            relay=relay,
            control_plane=control_plane,
            project_payload={
                "decode": _wrapped_data_projector,
                "archive": _wrapped_data_projector,
            },
            same_process_targets={"decode", "archive"},
            local_dispatcher=LocalStageDispatcher(),
        )

        await sender._route_result(
            "req-fanout",
            make_stage_payload(request_id="req-fanout", data={"answer": 7}),
        )

        assert [target for target, _, _ in control_plane.sent_to_stage] == [
            "decode",
            "archive",
        ]
        assert relay.storage

    asyncio.run(_run())


def test_stage_projected_fan_out_allows_tensor_leaf_sharing() -> None:
    def _tensor_leaf_projector(payload):
        return make_stage_payload(
            request_id=payload.request_id,
            inputs=payload.request.inputs,
            params=payload.request.params,
            data={"tensor": payload.data["tensor"], "target_only": []},
        )

    async def _run() -> None:
        dispatcher = LocalStageDispatcher()
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        decode_scheduler = FakeScheduler()
        decode = make_stage(name="decode", scheduler=decode_scheduler)
        sender = make_stage(
            name="thinker",
            get_next=lambda request_id, output: "decode",
            endpoints={"decode": "inproc://decode"},
            relay=relay,
            control_plane=control_plane,
            project_payload={"decode": _tensor_leaf_projector},
            same_process_targets={"decode"},
            local_dispatcher=dispatcher,
        )
        dispatcher.register_many([sender, decode])
        tensor = torch.arange(4)

        await sender._route_result(
            "req-tensor-leaf",
            make_stage_payload(
                request_id="req-tensor-leaf",
                data={"tensor": tensor, "scratch": []},
            ),
        )

        assert relay.storage == {}
        assert control_plane.sent_to_stage == []
        queued = decode_scheduler.inbox.get_nowait()
        assert queued.data.data["tensor"] is tensor

    asyncio.run(_run())


def test_stage_projected_fan_out_requires_stage_payload_projection() -> None:
    def _invalid_projector(payload):
        del payload
        return {"not": "a-stage-payload"}

    async def _run() -> None:
        sender = make_stage(
            name="thinker",
            get_next=lambda request_id, output: ["decode", "archive"],
            endpoints={
                "decode": "inproc://decode",
                "archive": "inproc://archive",
            },
            project_payload={
                "decode": _invalid_projector,
                "archive": _invalid_projector,
            },
            same_process_targets={"decode", "archive"},
            local_dispatcher=LocalStageDispatcher(),
        )

        with pytest.raises(
            TypeError,
            match="projectors to return StagePayload",
        ):
            await sender._route_result(
                "req-fanout",
                make_stage_payload(request_id="req-fanout", data={"answer": 7}),
            )

    asyncio.run(_run())


def test_stage_sends_same_process_stream_chunk_as_local_object(monkeypatch) -> None:
    events: list[dict] = []
    monkeypatch.setattr(
        "sglang_omni.pipeline.stage.runtime._emit_event",
        lambda **kwargs: events.append(kwargs),
    )

    async def _run() -> None:
        dispatcher = LocalStageDispatcher()
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        receiver_scheduler = FakeScheduler()
        receiver = make_stage(
            name="talker",
            scheduler=receiver_scheduler,
            can_accept_stream_before_payload=True,
        )
        receiver._stream_queue = StreamQueue()
        sender = make_stage(
            name="thinker",
            endpoints={"talker": "inproc://talker"},
            relay=relay,
            control_plane=control_plane,
            same_process_targets={"talker"},
            local_dispatcher=dispatcher,
        )
        dispatcher.register_many([sender, receiver])

        chunk = torch.arange(4)
        metadata = {"modality": "audio"}

        await sender._send_stream_to_target(
            "req-stream-local",
            chunk,
            "talker",
            metadata,
        )

        assert relay.storage == {}
        assert control_plane.sent_to_stage == []
        queued = receiver_scheduler.inbox.get_nowait()
        assert queued.type == "stream_chunk"
        assert queued.data.chunk_id == 0
        assert queued.data.data is chunk
        assert queued.data.metadata is metadata

    asyncio.run(_run())

    receive_events = [
        event
        for event in events
        if event["event_name"] == "stage_stream_chunk_received"
    ]
    assert receive_events == [
        {
            "request_id": "req-stream-local",
            "stage": "talker",
            "event_name": "stage_stream_chunk_received",
            "metadata": {"from_stage": "thinker", "chunk_id": 0},
        }
    ]


def test_stage_sends_same_process_stream_done_and_final_payload_locally() -> None:
    async def _run() -> None:
        dispatcher = LocalStageDispatcher()
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        receiver_scheduler = FakeScheduler()
        receiver = make_stage(
            name="decode",
            scheduler=receiver_scheduler,
            can_accept_stream_before_payload=True,
        )
        receiver._stream_queue = StreamQueue()
        sender = make_stage(
            name="thinker",
            get_next=lambda request_id, output: "decode",
            endpoints={"decode": "inproc://decode"},
            relay=relay,
            control_plane=control_plane,
            stream_targets=["decode"],
            same_process_targets={"decode"},
            local_dispatcher=dispatcher,
        )
        dispatcher.register_many([sender, receiver])

        payload = make_stage_payload(request_id="req-stream-local", data={"answer": 7})
        await sender._route_result("req-stream-local", payload)

        assert relay.storage == {}
        assert control_plane.sent_to_stage == []
        stream_done = receiver_scheduler.inbox.get_nowait()
        full_payload = receiver_scheduler.inbox.get_nowait()
        assert stream_done.type == "stream_done"
        assert full_payload.type == "new_request"
        assert full_payload.data is payload

    asyncio.run(_run())


def test_stage_allows_local_payload_when_static_stream_target_is_inactive() -> None:
    async def _run() -> None:
        dispatcher = LocalStageDispatcher()
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        receiver_scheduler = FakeScheduler()
        receiver = make_stage(name="decode", scheduler=receiver_scheduler)
        sender = make_stage(
            name="thinker",
            get_next=lambda request_id, output: "decode",
            get_stream_done_targets=lambda request_id, output: None,
            endpoints={"decode": "inproc://decode"},
            relay=relay,
            control_plane=control_plane,
            stream_targets=["decode"],
            same_process_targets={"decode"},
            local_dispatcher=dispatcher,
        )
        dispatcher.register_many([sender, receiver])

        payload = make_stage_payload(request_id="req-no-stream", data={"answer": 7})
        await sender._route_result("req-no-stream", payload)

        assert relay.storage == {}
        assert control_plane.sent_to_stage == []
        queued = receiver_scheduler.inbox.get_nowait()
        assert queued.type == "new_request"
        assert queued.data is payload

    asyncio.run(_run())


def test_stage_preserves_relay_order_when_target_also_receives_stream() -> None:
    async def _run() -> None:
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        sender = make_stage(
            name="thinker",
            get_next=lambda request_id, output: "decode",
            endpoints={"decode": "inproc://decode"},
            relay=relay,
            control_plane=control_plane,
            stream_targets=["decode"],
        )

        await sender._route_result(
            "req-streamed",
            make_stage_payload(request_id="req-streamed", data={"answer": 7}),
        )

        assert [msg.is_done for _, _, msg in control_plane.sent_to_stage] == [
            True,
            False,
        ]
        assert control_plane.sent_to_stage[1][2].chunk_id is None
        assert relay.storage

    asyncio.run(_run())


def test_stage_local_object_requires_registered_target() -> None:
    async def _run() -> None:
        sender = make_stage(
            name="thinker",
            endpoints={"decode": "inproc://decode"},
            same_process_targets={"decode"},
            local_dispatcher=LocalStageDispatcher(),
        )

        with pytest.raises(RuntimeError, match="not registered"):
            await sender._send_to_stage(
                "req-local",
                "decode",
                make_stage_payload(request_id="req-local"),
                allow_local_object=True,
            )

    asyncio.run(_run())
