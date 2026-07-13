# SPDX-License-Identifier: Apache-2.0
"""TensorRef protocol, policy, lifecycle, and observability contracts."""

from __future__ import annotations

import asyncio
import pickle

import pytest
import torch

from sglang_omni.comm import stage_io
from sglang_omni.comm.data_ref import (
    BackendRef,
    DataKind,
    DataLayout,
    DataRef,
    TransportKind,
)
from sglang_omni.comm.engine import CommEngine
from sglang_omni.comm.router import CommRouter
from sglang_omni.comm.tensor_ref import TensorRef
from sglang_omni.config.schema import StageConfig, TensorRefEdgeConfig
from sglang_omni.pipeline.mp_runner import _build_tensor_ref_policies
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.pipeline.stage.tensor_ref import TensorRefPolicy
from sglang_omni.proto import DataAckMessage, DataReadyMessage, StagePayload
from tests.unit_test.fixtures.pipeline_fakes import (
    FakeRelay,
    FakeScheduler,
    RecordingStageControlPlane,
    make_stage_payload,
    wait_until,
)

# ---------------------------------------------------------------------------
# Test environment
# ---------------------------------------------------------------------------


def _make_stage(
    name: str,
    *,
    relay: FakeRelay,
    control_plane: RecordingStageControlPlane | None = None,
    endpoints: dict[str, str] | None = None,
    scheduler: FakeScheduler | None = None,
    tensor_ref_policies: dict[str, TensorRefPolicy] | None = None,
) -> Stage:
    """Build only the Stage collaborators relevant to TensorRef tests."""
    return Stage(
        name=name,
        role="single",
        get_next=lambda *_: None,
        gpu_id=None,
        endpoints={} if endpoints is None else endpoints,
        control_plane=(
            RecordingStageControlPlane() if control_plane is None else control_plane
        ),
        relay=relay,
        scheduler=scheduler,
        tensor_ref_policies=tensor_ref_policies,
    )


def _policy(*paths: str, threshold_bytes: int = 1) -> TensorRefPolicy:
    return TensorRefPolicy(
        threshold_bytes=threshold_bytes,
        consumer_stage="thinker",
        paths=paths,
    )


def _make_tensor_ref(
    *,
    object_id: str = "tensor-ref-1",
    path: str = "video_embeds",
) -> TensorRef:
    return TensorRef(
        request_id="req-1",
        producer_stage="encoder",
        consumer_stage="thinker",
        path=path,
        nbytes=32,
        data_ref=DataRef(
            version=1,
            kind=DataKind.TENSOR_REF,
            object_id=object_id,
            transport=TransportKind.SHM,
            layout=DataLayout.RAW_TENSOR,
            buffer=BackendRef(
                transport=TransportKind.SHM,
                info={"transfer_info": {"size": 32}},
                length=32,
            ),
            shape=(8,),
            dtype="torch.float32",
            offset=0,
        ),
    )


def _direct_payload_ref(payload: StagePayload) -> dict[str, object]:
    return {
        "_type": "TorchCudaIpcPayload",
        "version": 1,
        "header": pickle.dumps(payload),
        "tensors": [],
    }


# ---------------------------------------------------------------------------
# TensorRef wire protocol
# ---------------------------------------------------------------------------


def test_tensor_ref_uses_versioned_round_trip_wire_format() -> None:
    """A valid descriptor keeps its identity across the public wire format."""
    ref = _make_tensor_ref()

    value = ref.to_dict()

    assert value["_type"] == "TensorRef"
    assert value["version"] == 1
    assert TensorRef.from_dict(value) == ref


def test_tensor_ref_rejects_wrong_protocol_version() -> None:
    """The decoder rejects descriptors from an unsupported protocol version."""
    value = _make_tensor_ref().to_dict()
    value["version"] = 2

    with pytest.raises(ValueError, match="unsupported TensorRef version"):
        TensorRef.from_dict(value)


def test_tensor_ref_rejects_non_tensor_data_ref() -> None:
    """A TensorRef cannot wrap a DataRef for an ordinary Stage payload."""
    value = _make_tensor_ref().to_dict()
    value["data_ref"]["kind"] = DataKind.STAGE_PAYLOAD.value

    with pytest.raises(ValueError, match="data_ref kind"):
        TensorRef.from_dict(value)


# ---------------------------------------------------------------------------
# TensorRef policy and configuration
# ---------------------------------------------------------------------------


def test_tensor_ref_policy_is_compiled_from_stage_config() -> None:
    """Compilation maps stage names and preserves the exact size/path boundary."""
    stage_cfg = StageConfig(
        name="encoder",
        process="pipeline",
        factory="tests.fake_factory",
        next="aggregate",
        tensor_ref_edges={
            "aggregate": TensorRefEdgeConfig(
                consumer_stage="thinker",
                threshold_mb=1.0,
                paths=("encoder_outs.image_encoder.video_embeds",),
            )
        },
    )
    name_map = {
        "aggregate": "aggregate_fused",
        "thinker": "thinker_fused",
    }

    policies = _build_tensor_ref_policies(stage_cfg, name_map)
    policy = policies["aggregate_fused"]

    assert set(policies) == {"aggregate_fused"}
    assert policy.consumer_stage == "thinker_fused"
    assert policy.threshold_bytes == 1024 * 1024
    assert policy.should_externalize(
        "encoder_outs.image_encoder.video_embeds", 1024 * 1024
    )
    assert not policy.should_externalize(
        "encoder_outs.image_encoder.video_embeds", 1024 * 1024 - 1
    )
    assert not policy.should_externalize(
        "encoder_outs.other_encoder.video_embeds", 1024 * 1024
    )


# ---------------------------------------------------------------------------
# TensorRef lifecycle
# ---------------------------------------------------------------------------

# Publication: decide which tensors become references and retain their resources.


def test_externalization_skips_nonmatching_tensor_without_side_effects() -> None:
    """A non-allowlisted tensor stays inline and performs no Relay write."""

    async def _run() -> None:
        # Arrange: only video_embeds is eligible for externalization.
        relay = FakeRelay()
        producer = _make_stage(
            "encoder",
            relay=relay,
            endpoints={"aggregate": "inproc://aggregate"},
            tensor_ref_policies={"aggregate": _policy("video_embeds")},
        )
        tensor = torch.arange(8)
        payload = make_stage_payload(request_id="req-1", data={"unrelated": tensor})

        # Act: visit a payload without a matching path.
        externalized, stats = await producer._externalize_tensor_refs(
            payload, "aggregate"
        )

        # Assert: the original object and empty data-plane state are preserved.
        assert externalized.data["unrelated"] is tensor
        assert relay.ops == []
        assert stats is None

    asyncio.run(_run())


def test_externalization_publishes_configured_nested_tensor_list() -> None:
    """Runtime path construction joins list indices to the configured base path."""

    async def _run() -> None:
        # Arrange: the second tensor is exactly on the 16-byte threshold.
        relay = FakeRelay()
        configured_path = "encoder_outs.image_encoder.deepstack_visual_embeds_image"
        producer = _make_stage(
            "encoder",
            relay=relay,
            endpoints={"aggregate": "inproc://aggregate"},
            tensor_ref_policies={
                "aggregate": _policy(configured_path, threshold_bytes=16)
            },
        )
        payload = make_stage_payload(
            request_id="req-1",
            data={
                "encoder_outs": {
                    "image_encoder": {
                        "deepstack_visual_embeds_image": [
                            torch.arange(8, dtype=torch.float32),
                            torch.arange(4, dtype=torch.float32),
                        ]
                    }
                }
            },
        )

        # Act: externalize both configured list elements.
        externalized, stats = await producer._externalize_tensor_refs(
            payload, "aggregate"
        )
        values = externalized.data["encoder_outs"]["image_encoder"][
            "deepstack_visual_embeds_image"
        ]
        refs = [TensorRef.from_dict(value) for value in values]

        # Assert: paths and byte statistics describe the two physical tensors.
        assert [ref.path for ref in refs] == [
            f"{configured_path}[0]",
            f"{configured_path}[1]",
        ]
        assert [ref.nbytes for ref in refs] == [32, 16]
        assert stats == {"tensor_ref_count": 2, "tensor_ref_bytes": 48}

        # Cleanup: complete the two publications created by this test.
        for ref in refs:
            producer._comm.ack_transfer(
                DataAckMessage(
                    request_id="req-1",
                    from_stage="thinker",
                    to_stage="encoder",
                    object_id=ref.data_ref.object_id,
                )
            )
        await wait_until(lambda: all(op.waited for op in relay.ops))

    asyncio.run(_run())


def test_published_tensor_ref_keeps_sender_operation_until_consumer_ack() -> None:
    """Publication resources remain pending until the final consumer ACK arrives."""

    async def _run() -> None:
        # Arrange: publish one 8 x float32 tensor through the communication engine.
        relay = FakeRelay()
        engine = CommEngine(
            CommRouter(
                stage_name="encoder",
                gpu_id=None,
                same_process_targets=set(),
                gpu_stage_names=set(),
                comm_config={"ack_timeout_s": 1.0},
                injected_relay=relay,
            )
        )
        ref = await engine.publish_tensor_ref(
            relay=relay,
            request_id="req-1",
            tensor=torch.arange(8, dtype=torch.float32),
            transport=TransportKind.SHM,
            producer_stage="encoder",
            consumer_stage="thinker",
            path="video_embeds",
        )
        op = relay.ops[0]

        # Assert before ACK: nbytes is physical payload size and the op is retained.
        assert ref.nbytes == 8 * 4
        assert ref.data_ref.kind is DataKind.TENSOR_REF
        assert op.acked is False
        assert op.waited is False

        # Act: deliver the final consumer's successful read acknowledgement.
        engine.ack_transfer(
            DataAckMessage(
                request_id="req-1",
                from_stage="thinker",
                to_stage="encoder",
                object_id=ref.data_ref.object_id,
            )
        )

        # Assert after ACK: the Relay operation reaches its terminal state.
        await wait_until(lambda: op.waited)
        assert op.acked is True

    asyncio.run(_run())


# Materialization: only the declared consumer reads and acknowledges a reference.


def test_consumer_materializes_tensor_ref_and_acks_original_producer() -> None:
    """The final consumer restores tensor contents and ACKs the original producer."""

    async def _run() -> None:
        # Arrange: encoder publishes a ref that may pass through aggregate.
        relay = FakeRelay()
        producer = _make_stage(
            "encoder",
            relay=relay,
            endpoints={"aggregate": "inproc://aggregate"},
            tensor_ref_policies={"aggregate": _policy("video_embeds")},
        )
        expected = torch.arange(256)
        forwarded, _ = await producer._externalize_tensor_refs(
            make_stage_payload(request_id="req-1", data={"video_embeds": expected}),
            "aggregate",
        )
        ref = TensorRef.from_dict(forwarded.data["video_embeds"])
        consumer_control = RecordingStageControlPlane()
        consumer = _make_stage(
            "thinker",
            relay=relay,
            control_plane=consumer_control,
            endpoints={"encoder": "inproc://encoder"},
        )

        # Act: thinker resolves the descriptor.
        materialized = await consumer._materialize_tensor_refs("req-1", forwarded)

        # Assert: data is correct and exactly one success ACK names the ref object.
        assert torch.equal(materialized.data["video_embeds"], expected)
        assert len(consumer_control.sent_to_stage) == 1
        target, _, ack = consumer_control.sent_to_stage[0]
        assert target == "encoder"
        assert ack.object_id == ref.data_ref.object_id
        assert ack.success is True

        producer._comm.ack_transfer(ack)
        await wait_until(lambda: relay.ops[0].waited)

    asyncio.run(_run())


def test_consumer_materializes_duplicate_tensor_ref_once() -> None:
    """Repeated descriptors for one object share one read, tensor, and success ACK."""

    async def _run() -> None:
        # Arrange: two payload positions contain copies of the same descriptor.
        relay = FakeRelay()
        producer = _make_stage(
            "encoder",
            relay=relay,
            endpoints={"aggregate": "inproc://aggregate"},
            tensor_ref_policies={"aggregate": _policy("video_embeds")},
        )
        forwarded, _ = await producer._externalize_tensor_refs(
            make_stage_payload(
                request_id="req-1", data={"video_embeds": torch.arange(8)}
            ),
            "aggregate",
        )
        ref = forwarded.data["video_embeds"]
        duplicated = make_stage_payload(
            request_id="req-1", data={"left": ref, "right": dict(ref)}
        )
        consumer_control = RecordingStageControlPlane()
        consumer = _make_stage(
            "thinker",
            relay=relay,
            control_plane=consumer_control,
            endpoints={"encoder": "inproc://encoder"},
        )

        # Act: materialize the payload containing duplicate descriptors.
        materialized = await consumer._materialize_tensor_refs("req-1", duplicated)

        # Assert: object_id is the deduplication key for the whole materialization.
        assert relay.get_calls == 1
        assert materialized.data["left"] is materialized.data["right"]
        assert len(consumer_control.sent_to_stage) == 1

        _, _, ack = consumer_control.sent_to_stage[0]
        producer._comm.ack_transfer(ack)
        await wait_until(lambda: relay.ops[0].waited)

    asyncio.run(_run())


def test_consumer_read_failure_sends_failed_ack_and_ends_sender_operation() -> None:
    """A failed read becomes a failed ACK and terminates the producer resource."""

    async def _run() -> None:
        # Arrange: publish successfully, then fail the consumer's Relay read.
        relay = FakeRelay()
        producer = _make_stage(
            "encoder",
            relay=relay,
            endpoints={"aggregate": "inproc://aggregate"},
            tensor_ref_policies={"aggregate": _policy("video_embeds")},
        )
        forwarded, _ = await producer._externalize_tensor_refs(
            make_stage_payload(
                request_id="req-1", data={"video_embeds": torch.arange(8)}
            ),
            "aggregate",
        )
        ref = TensorRef.from_dict(forwarded.data["video_embeds"])
        relay.fail_get = RuntimeError("read failed")
        consumer_control = RecordingStageControlPlane()
        consumer = _make_stage(
            "thinker",
            relay=relay,
            control_plane=consumer_control,
            endpoints={"encoder": "inproc://encoder"},
        )

        # Act: materialization fails at the data-plane read boundary.
        with pytest.raises(RuntimeError, match="read failed"):
            await consumer._materialize_tensor_refs("req-1", forwarded)

        # Assert: the failed ACK identifies the same publication and releases it.
        assert len(consumer_control.sent_to_stage) == 1
        target, _, ack = consumer_control.sent_to_stage[0]
        assert target == "encoder"
        assert ack.object_id == ref.data_ref.object_id
        assert ack.success is False
        assert ack.error == "read failed"

        producer._comm.ack_transfer(ack)
        await wait_until(lambda: relay.ops[0].waited)
        assert str(relay.ops[0].failed) == "read failed"

    asyncio.run(_run())


# Abort: only unresolved refs owned by the final consumer receive failed ACKs.


def test_consumer_abort_fails_all_unresolved_tensor_refs_for_request() -> None:
    """Request abort emits one failed ACK for every unresolved consumer ref."""

    async def _run() -> None:
        # Arrange: thinker has discovered two refs belonging to one request.
        control = RecordingStageControlPlane()
        scheduler = FakeScheduler()
        consumer = _make_stage(
            "thinker",
            relay=FakeRelay(),
            control_plane=control,
            endpoints={"encoder": "inproc://encoder"},
            scheduler=scheduler,
        )
        refs = [
            _make_tensor_ref(object_id="tensor-ref-1", path="video_embeds[0]"),
            _make_tensor_ref(object_id="tensor-ref-2", path="video_embeds[1]"),
        ]
        payload = make_stage_payload(
            request_id="req-1", data={"video_embeds": [ref.to_dict() for ref in refs]}
        )
        consumer._remember_tensor_refs("req-1", payload)

        # Act: abort the request before either ref is materialized.
        consumer._on_abort("req-1")
        await asyncio.gather(*list(consumer._receive_tasks))

        # Assert: request work and both unresolved publications reach failure.
        assert scheduler.aborted == ["req-1"]
        tensor_acks = [message for _, _, message in control.sent_to_stage]
        assert {ack.object_id for ack in tensor_acks} == {
            ref.data_ref.object_id for ref in refs
        }
        assert all(ack.success is False for ack in tensor_acks)
        assert all(ack.error == "request aborted" for ack in tensor_acks)

    asyncio.run(_run())


def test_aborted_non_consumer_does_not_ack_refs_in_relay_payload() -> None:
    """A relay Stage drains and ACKs the payload but never ACKs forwarded refs."""

    async def _run() -> None:
        # Arrange: aggregate receives an already-aborted relay payload for thinker.
        relay = FakeRelay()
        ref = _make_tensor_ref()
        payload = make_stage_payload(
            request_id="req-1", data={"video_embeds": ref.to_dict()}
        )
        data_ref, _ = await stage_io.write_payload(
            relay,
            "req-1",
            payload,
            transport=TransportKind.SHM,
            from_stage="encoder",
            to_stage="aggregate",
        )
        control = RecordingStageControlPlane()
        receiver = _make_stage(
            "aggregate",
            relay=relay,
            control_plane=control,
            endpoints={"encoder": "inproc://encoder"},
        )
        receiver._aborted.add("req-1")

        # Act: discard the late payload rather than scheduling it.
        await receiver._on_data_ready(
            DataReadyMessage(
                request_id="req-1",
                from_stage="encoder",
                to_stage="aggregate",
                data_ref=data_ref.to_dict(),
            )
        )

        # Assert: data is drained and only the outer payload object is ACKed.
        acked_ids = {
            message.object_id
            for _, _, message in control.sent_to_stage
            if isinstance(message, DataAckMessage)
        }
        assert relay.get_calls == 1
        assert data_ref.object_id in acked_ids
        assert ref.data_ref.object_id not in acked_ids

    asyncio.run(_run())


def test_aborted_non_consumer_does_not_ack_refs_in_direct_ipc_payload(
    monkeypatch,
) -> None:
    """Direct-IPC discard consumes the payload without claiming consumer status."""
    deserialize_calls = 0
    deserialize = stage_io.deserialize_direct_cuda_ipc_payload

    def _record_deserialize(data_ref):
        nonlocal deserialize_calls
        deserialize_calls += 1
        return deserialize(data_ref)

    monkeypatch.setattr(
        stage_io, "deserialize_direct_cuda_ipc_payload", _record_deserialize
    )

    async def _run() -> None:
        # Arrange: aggregate receives an inline descriptor intended for thinker.
        ref = _make_tensor_ref()
        payload = make_stage_payload(
            request_id="req-1", data={"video_embeds": ref.to_dict()}
        )
        control = RecordingStageControlPlane()
        receiver = _make_stage(
            "aggregate",
            relay=FakeRelay(),
            control_plane=control,
            endpoints={"encoder": "inproc://encoder"},
        )
        receiver._aborted.add("req-1")

        # Act: discard the direct-IPC payload.
        await receiver._on_data_ready(
            DataReadyMessage(
                request_id="req-1",
                from_stage="encoder",
                to_stage="aggregate",
                data_ref=_direct_payload_ref(payload),
            )
        )

        # Assert: the direct descriptor was consumed and no TensorRef was ACKed.
        assert deserialize_calls == 1
        assert not any(
            isinstance(message, DataAckMessage)
            and message.object_id == ref.data_ref.object_id
            for _, _, message in control.sent_to_stage
        )

    asyncio.run(_run())


def test_aborted_consumer_fails_refs_in_discarded_relay_payload() -> None:
    """A late relay payload is drained before its final consumer fails its refs."""

    async def _run() -> None:
        # Arrange: thinker has already aborted when aggregate's payload arrives.
        relay = FakeRelay()
        ref = _make_tensor_ref()
        payload = make_stage_payload(
            request_id="req-1", data={"video_embeds": ref.to_dict()}
        )
        data_ref, _ = await stage_io.write_payload(
            relay,
            "req-1",
            payload,
            transport=TransportKind.SHM,
            from_stage="aggregate",
            to_stage="thinker",
        )
        control = RecordingStageControlPlane()
        receiver = _make_stage(
            "thinker",
            relay=relay,
            control_plane=control,
            endpoints={
                "aggregate": "inproc://aggregate",
                "encoder": "inproc://encoder",
            },
        )
        receiver._aborted.add("req-1")

        # Act: discard the late relay payload.
        await receiver._on_data_ready(
            DataReadyMessage(
                request_id="req-1",
                from_stage="aggregate",
                to_stage="thinker",
                data_ref=data_ref.to_dict(),
            )
        )

        # Assert: the outer payload succeeds, while the unresolved ref fails.
        acks = {
            message.object_id: message
            for _, _, message in control.sent_to_stage
            if isinstance(message, DataAckMessage)
        }
        assert acks[data_ref.object_id].success is True
        assert acks[ref.data_ref.object_id].success is False
        assert acks[ref.data_ref.object_id].error == "request aborted"

    asyncio.run(_run())


def test_aborted_consumer_fails_refs_in_discarded_direct_ipc_payload() -> None:
    """A late direct-IPC payload lets the final consumer fail embedded refs."""

    async def _run() -> None:
        # Arrange: thinker has already aborted when an inline payload arrives.
        ref = _make_tensor_ref()
        payload = make_stage_payload(
            request_id="req-1", data={"video_embeds": ref.to_dict()}
        )
        control = RecordingStageControlPlane()
        receiver = _make_stage(
            "thinker",
            relay=FakeRelay(),
            control_plane=control,
            endpoints={"encoder": "inproc://encoder"},
        )
        receiver._aborted.add("req-1")

        # Act: discard and inspect the direct-IPC payload.
        await receiver._on_data_ready(
            DataReadyMessage(
                request_id="req-1",
                from_stage="aggregate",
                to_stage="thinker",
                data_ref=_direct_payload_ref(payload),
            )
        )

        # Assert: thinker sends one failed terminal ACK to encoder.
        assert len(control.sent_to_stage) == 1
        target, _, ack = control.sent_to_stage[0]
        assert target == "encoder"
        assert ack.object_id == ref.data_ref.object_id
        assert ack.success is False
        assert ack.error == "request aborted"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# TensorRef observability
# ---------------------------------------------------------------------------


def test_tensor_ref_stats_are_attached_to_payload_hop(monkeypatch) -> None:
    """A hop event reports the count and physical bytes externalized on that edge."""
    events: list[dict] = []
    monkeypatch.setattr(
        "sglang_omni.pipeline.stage.runtime._emit_event",
        lambda **kwargs: events.append(kwargs),
    )

    async def _run() -> None:
        # Arrange: one int64 tensor (8 x 8 bytes) is eligible on the edge.
        relay = FakeRelay()
        control = RecordingStageControlPlane()
        producer = _make_stage(
            "encoder",
            relay=relay,
            control_plane=control,
            endpoints={"aggregate": "inproc://aggregate"},
            tensor_ref_policies={"aggregate": _policy("video_embeds")},
        )

        # Act: send a payload through the normal Stage hop.
        await producer._send_to_stage(
            "req-1",
            "aggregate",
            make_stage_payload(
                request_id="req-1", data={"video_embeds": torch.arange(8)}
            ),
            allow_local_object=False,
        )

        # Assert: observability reports bytes, not element count.
        hop = next(event for event in events if event["event_name"] == "stage_hop_sent")
        assert hop["metadata"]["tensor_ref_count"] == 1
        assert hop["metadata"]["tensor_ref_bytes"] == 64

        # Cleanup: ACK both the outer payload and its embedded TensorRef.
        _, _, payload_message = control.sent_to_stage[0]
        payload_ref = DataRef.from_dict(payload_message.data_ref)
        forwarded = stage_io.deserialize_payload_header(payload_ref)
        tensor_ref = TensorRef.from_dict(forwarded.data["video_embeds"])
        producer._comm.ack_transfer(
            DataAckMessage(
                request_id="req-1",
                from_stage="aggregate",
                to_stage="encoder",
                object_id=payload_ref.object_id,
            )
        )
        producer._comm.ack_transfer(
            DataAckMessage(
                request_id="req-1",
                from_stage="thinker",
                to_stage="encoder",
                object_id=tensor_ref.data_ref.object_id,
            )
        )
        await wait_until(lambda: all(op.waited for op in relay.ops))

    asyncio.run(_run())
