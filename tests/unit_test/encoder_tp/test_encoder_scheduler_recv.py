# SPDX-License-Identifier: Apache-2.0
"""EncoderScheduler ``_recv_messages`` contract tests.

These tests exercise the contracts that don't need a live
SGLang TP group: ``tp_size=1`` lift+strip semantics, the unified
``(messages, error)`` return type, the ``BatchCollectError`` capture,
the per-iteration error boundary, and the request-level error
emission against drained messages.

The TP-multi-rank lanes (broadcast_pyobj, dist.broadcast,
allocation-ready gather) are mocked: we feed a fake ``runner.tp_group``
and a stand-in ``broadcast_pyobj`` / ``dist.all_gather_object`` /
``dist.broadcast`` so we can drive both the leader and follower paths
deterministically without a multi-process distributed context.
"""
from __future__ import annotations

import queue as _queue_mod
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.encoder_scheduler import (
    _RECV_ERROR_KIND,
    EncoderScheduler,
    _TensorSpec,
    _find_tensor_paths,
)
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _mk_payload(request_id: str, data: dict | None = None) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=None,
        data=data if data is not None else {"foo": "bar"},
    )


def _mk_msg(request_id: str, data: dict | None = None) -> IncomingMessage:
    return IncomingMessage(
        request_id=request_id,
        type="new_request",
        data=_mk_payload(request_id, data),
    )


class _FakeAdapter:
    stage_name = "image_encoder"

    def __init__(self):
        self.build_calls = 0
        self.run_calls = 0
        self.slice_calls = 0

    def build_batch(self, messages):
        self.build_calls += 1
        return SimpleNamespace(
            adapter=self,
            image_items=[],
            video_items=[],
            audio_items=[],
            spans=[],
            is_empty=True,
        )

    def run_feature(self, model, plan):
        self.run_calls += 1
        return {"image": None, "video": None, "audio": None}

    def slice_results(self, raw, plan, messages):
        self.slice_calls += 1
        # mirror what real adapters do: build a StagePayload per message
        return [_mk_payload(m.request_id, data={"ok": True}) for m in messages]


class _FakeWorker:
    def __init__(self, *, tp_size: int = 1, tp_rank: int = 0):
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.is_entry_rank = tp_rank == 0
        self.device = torch.device("cpu")  # tests run CPU-only
        self.tp_group = SimpleNamespace(
            rank=tp_rank,
            ranks=[0, 1] if tp_size > 1 else [0],
            cpu_group=object(),
            device_group=object(),
        )

    def encode_batch(self, plan):
        return plan.adapter.run_feature(self, plan)


# ---------------------------------------------------------------------------
# tp_size == 1 lane
# ---------------------------------------------------------------------------


def test_recv_returns_empty_when_inbox_empty():
    sch = EncoderScheduler(_FakeWorker(tp_size=1), _FakeAdapter())
    msgs, err = sch._recv_messages()
    assert msgs == [] and err is None


def test_recv_drains_inbox_and_lifts_tensors_at_tp_size_1():
    sch = EncoderScheduler(_FakeWorker(tp_size=1), _FakeAdapter())
    t = torch.randn(2, 3)
    sch.inbox.put(_mk_msg("r0", data={"pixel_values": t}))
    msgs, err = sch._recv_messages()
    assert err is None
    assert len(msgs) == 1
    out_payload = msgs[0].data
    # tensor preserved (no reattachment surgery on tp_size=1; lifted in-place)
    out_t = out_payload.data["pixel_values"]
    assert isinstance(out_t, torch.Tensor)
    assert torch.equal(out_t, t)
    assert sch._last_recv_timing.inbox_admission_ms >= 0.0
    assert sch._last_recv_timing.strip_h2d_ms >= 0.0
    assert sch._last_recv_timing.metadata_broadcast_ms == 0.0
    assert sch._last_recv_timing.tensor_broadcast_ms == 0.0


def test_recv_reports_collect_error_with_drained_messages():
    """``request_cost_fn`` is adapter/model code; raising must surface as
    ``(messages, error)``, not propagate as an exception."""

    def _bad_cost(payload):
        raise RuntimeError("boom in cost fn")

    sch = EncoderScheduler(
        _FakeWorker(tp_size=1),
        _FakeAdapter(),
        max_batch_size=4,
        max_batch_wait_ms=0,
        request_cost_fn=_bad_cost,
        max_batch_cost=10**9,
    )
    sch.inbox.put(_mk_msg("r0"))
    sch.inbox.put(_mk_msg("r1"))
    msgs, err = sch._recv_messages()
    assert isinstance(err, RuntimeError)
    assert str(err) == "boom in cost fn"
    # First message drained before raising; admission attempted.
    assert msgs and msgs[0].request_id == "r0"


def test_recv_returns_strip_and_lift_failure_at_tp_size_1():
    """If lifting tensors raises (e.g. dtype coercion), the error rides
    the tuple return path."""
    sch = EncoderScheduler(_FakeWorker(tp_size=1), _FakeAdapter())
    sch.inbox.put(_mk_msg("r0", data={"pixel_values": torch.randn(2, 2)}))

    with patch.object(sch, "_strip_and_lift", side_effect=RuntimeError("lift boom")):
        msgs, err = sch._recv_messages()
    assert isinstance(err, RuntimeError)
    assert "lift boom" in str(err)
    assert msgs and msgs[0].request_id == "r0"


def test_emit_error_emits_one_outgoing_per_drained_request():
    sch = EncoderScheduler(_FakeWorker(tp_size=1), _FakeAdapter())
    msgs = [_mk_msg("r0"), _mk_msg("r1"), _mk_msg("r2")]
    sch._emit_error(msgs, RuntimeError("nope"))
    out: list[OutgoingMessage] = []
    while True:
        try:
            out.append(sch.outbox.get_nowait())
        except _queue_mod.Empty:
            break
    assert [o.request_id for o in out] == ["r0", "r1", "r2"]
    assert all(o.type == "error" for o in out)


def test_emit_error_skips_aborted_requests():
    sch = EncoderScheduler(_FakeWorker(tp_size=1), _FakeAdapter())
    sch.abort("r1")
    msgs = [_mk_msg("r0"), _mk_msg("r1"), _mk_msg("r2")]
    sch._emit_error(msgs, RuntimeError("x"))
    rids = []
    while True:
        try:
            rids.append(sch.outbox.get_nowait().request_id)
        except _queue_mod.Empty:
            break
    assert rids == ["r0", "r2"]


# ---------------------------------------------------------------------------
# tp_size > 1 lane: leader / follower with mocked broadcasts
# ---------------------------------------------------------------------------


class _BroadcastBus:
    """Coordinates fake broadcast_pyobj + dist.{broadcast, all_gather_object}.

    Used to drive the two-channel ``_recv_messages`` from inside a
    single test process.
    """

    def __init__(self):
        self.metadata: list = None  # leader writes; follower reads
        self.tensor_calls: list[torch.Tensor] = []
        self.alloc_flags: list[list[bool]] = []
        self.broadcast_calls = 0


@pytest.fixture
def bus():
    return _BroadcastBus()


def _patched_broadcast_pyobj(bus: _BroadcastBus):
    def _f(data, rank, dist_group, src=0, force_cpu_device=True):
        # Leader (rank == src) publishes; follower (rank != src) reads.
        if rank == src:
            bus.metadata = data
            return data
        return bus.metadata

    return _f


def _patched_dist_broadcast(bus: _BroadcastBus):
    def _f(tensor, src=0, group=None):
        bus.tensor_calls.append(tensor)
        bus.broadcast_calls += 1

    return _f


def _patched_all_gather_object(bus: _BroadcastBus):
    def _f(out_list, val, group=None):
        # Single test process — simulate "every rank reports the same"
        for i in range(len(out_list)):
            out_list[i] = val
        bus.alloc_flags.append(list(out_list))

    return _f


def test_pre_broadcast_collect_error_publishes_sentinel(bus):
    """Locks the [pre-broadcast error sentinel] contract: a recv-time
    failure on the entry rank must broadcast the tagged-dict sentinel
    over the same CPU-group slot the success path uses."""

    def _bad_cost(payload):
        raise RuntimeError("collect boom")

    sch = EncoderScheduler(
        _FakeWorker(tp_size=2, tp_rank=0),
        _FakeAdapter(),
        max_batch_size=4,
        max_batch_wait_ms=0,
        request_cost_fn=_bad_cost,
        max_batch_cost=10**9,
    )
    sch.inbox.put(_mk_msg("r0"))

    with (
        patch(
            "sglang_omni.scheduling.encoder_scheduler.broadcast_pyobj",
            _patched_broadcast_pyobj(bus),
        ),
        patch(
            "sglang_omni.scheduling.encoder_scheduler.dist.all_gather_object",
            _patched_all_gather_object(bus),
        ),
        patch(
            "sglang_omni.scheduling.encoder_scheduler.dist.broadcast",
            _patched_dist_broadcast(bus),
        ),
    ):
        msgs, err = sch._recv_messages()

    assert isinstance(err, RuntimeError)
    assert msgs and msgs[0].request_id == "r0"
    # Sentinel published. The metadata bus carries the tagged dict.
    assert isinstance(bus.metadata, list)
    assert isinstance(bus.metadata[0], dict)
    assert bus.metadata[0].get("kind") == _RECV_ERROR_KIND
    assert "collect boom" in bus.metadata[0]["error"]
    # No device broadcast issued before the metadata error.
    assert bus.broadcast_calls == 0
    assert sch._last_recv_timing.inbox_admission_ms >= 0.0
    assert sch._last_recv_timing.metadata_broadcast_ms >= 0.0


def test_follower_decodes_pre_broadcast_sentinel(bus):
    """Follower side: receive the tagged-dict sentinel, return ``([],
    RuntimeError(...))`` instead of blocking forever."""
    bus.metadata = [{"kind": _RECV_ERROR_KIND, "error": "RuntimeError('boom')"}]

    sch = EncoderScheduler(
        _FakeWorker(tp_size=2, tp_rank=1),
        _FakeAdapter(),
    )

    with (
        patch(
            "sglang_omni.scheduling.encoder_scheduler.broadcast_pyobj",
            _patched_broadcast_pyobj(bus),
        ),
        patch(
            "sglang_omni.scheduling.encoder_scheduler.dist.all_gather_object",
            _patched_all_gather_object(bus),
        ),
        patch(
            "sglang_omni.scheduling.encoder_scheduler.dist.broadcast",
            _patched_dist_broadcast(bus),
        ),
    ):
        msgs, err = sch._recv_messages()

    assert msgs == []
    assert isinstance(err, RuntimeError)
    assert "entry rank failed" in str(err) and "boom" in str(err)
    # Follower never issued a device broadcast.
    assert bus.broadcast_calls == 0


def test_follower_allocation_failure_returns_before_device_broadcast(bus):
    """Locks the allocation-ready handshake.

    If a non-entry rank cannot allocate receive placeholders, it must publish
    ``local_ok=False`` on the CPU group and return before any device broadcast
    is attempted. That avoids leaving the entry rank unmatched in NCCL.
    """
    bus.metadata = [
        [_mk_msg("r0", data={"pixel_values": {"__tensor__": "placeholder"}})],
        [[_TensorSpec(path="pixel_values", shape=(2, 3), dtype=torch.float32)]],
    ]
    sch = EncoderScheduler(
        _FakeWorker(tp_size=2, tp_rank=1),
        _FakeAdapter(),
    )

    with (
        patch(
            "sglang_omni.scheduling.encoder_scheduler.broadcast_pyobj",
            _patched_broadcast_pyobj(bus),
        ),
        patch(
            "sglang_omni.scheduling.encoder_scheduler.dist.all_gather_object",
            _patched_all_gather_object(bus),
        ),
        patch(
            "sglang_omni.scheduling.encoder_scheduler.dist.broadcast",
            _patched_dist_broadcast(bus),
        ),
        patch(
            "sglang_omni.scheduling.encoder_scheduler.torch.empty",
            side_effect=RuntimeError("placeholder oom"),
        ),
    ):
        msgs, err = sch._recv_messages()

    assert msgs == []
    assert isinstance(err, RuntimeError)
    assert "placeholder oom" in str(err)
    assert bus.alloc_flags == [[False, False]]
    assert bus.broadcast_calls == 0


def test_entry_broadcasts_stripped_metadata_and_returns_restored_messages(bus):
    """Entry rank should not send tensor bytes through broadcast_pyobj.

    The metadata channel carries only placeholder skeletons; the entry rank
    still receives GPU/CPU-resident tensors reattached for its own forward.
    """
    sch = EncoderScheduler(
        _FakeWorker(tp_size=2, tp_rank=0),
        _FakeAdapter(),
        max_batch_size=1,
        max_batch_wait_ms=0,
    )
    tensor = torch.randn(2, 3)
    sch.inbox.put(_mk_msg("r0", data={"pixel_values": tensor}))

    with (
        patch(
            "sglang_omni.scheduling.encoder_scheduler.broadcast_pyobj",
            _patched_broadcast_pyobj(bus),
        ),
        patch(
            "sglang_omni.scheduling.encoder_scheduler.dist.all_gather_object",
            _patched_all_gather_object(bus),
        ),
        patch(
            "sglang_omni.scheduling.encoder_scheduler.dist.broadcast",
            _patched_dist_broadcast(bus),
        ),
    ):
        msgs, err = sch._recv_messages()

    assert err is None
    assert len(msgs) == 1
    assert torch.equal(msgs[0].data.data["pixel_values"], tensor)
    assert len(bus.metadata) == 3
    published_msgs = bus.metadata[0]
    published_value = published_msgs[0].data.data["pixel_values"]
    assert isinstance(published_value, dict)
    assert published_value["_tensor_placeholder"] == "pixel_values"
    assert not isinstance(published_value, torch.Tensor)
    assert _find_tensor_paths(bus.metadata) == []
    assert bus.broadcast_calls == 1


def test_strip_and_lift_rejects_tensor_left_in_request_metadata():
    """The CPU-group metadata channel must never pickle tensor-bearing objects.

    ``payload.data`` tensors are extracted into the device-broadcast lane; a
    tensor left elsewhere in the message, such as request metadata, would be
    pickled by ``broadcast_pyobj`` and can allocate on follower ranks.
    """

    sch = EncoderScheduler(_FakeWorker(tp_size=2, tp_rank=0), _FakeAdapter())
    msg = _mk_msg("r0", data={"pixel_values": torch.randn(2, 3)})
    msg.data.request = SimpleNamespace(inputs=torch.randn(1), params={}, metadata={})

    with pytest.raises(RuntimeError, match="metadata broadcast payload"):
        sch._strip_and_lift([msg])


def test_entry_metadata_tensor_leak_publishes_error_sentinel(bus):
    """If tensor-free metadata validation fails on entry rank, the follower
    receives the normal recv-error sentinel instead of waiting for device
    broadcasts that will never happen."""

    sch = EncoderScheduler(
        _FakeWorker(tp_size=2, tp_rank=0),
        _FakeAdapter(),
        max_batch_size=1,
        max_batch_wait_ms=0,
    )
    msg = _mk_msg("r0", data={"pixel_values": torch.randn(2, 3)})
    msg.data.request = SimpleNamespace(inputs=torch.randn(1), params={}, metadata={})
    sch.inbox.put(msg)

    with (
        patch(
            "sglang_omni.scheduling.encoder_scheduler.broadcast_pyobj",
            _patched_broadcast_pyobj(bus),
        ),
        patch(
            "sglang_omni.scheduling.encoder_scheduler.dist.all_gather_object",
            _patched_all_gather_object(bus),
        ),
        patch(
            "sglang_omni.scheduling.encoder_scheduler.dist.broadcast",
            _patched_dist_broadcast(bus),
        ),
    ):
        msgs, err = sch._recv_messages()

    assert msgs and msgs[0].request_id == "r0"
    assert isinstance(err, RuntimeError)
    assert "metadata broadcast payload" in str(err)
    assert bus.metadata[0].get("kind") == _RECV_ERROR_KIND
    assert "metadata broadcast payload" in bus.metadata[0]["error"]
    assert bus.broadcast_calls == 0


def test_strip_and_lift_returns_typed_dtype_specs():
    """Specs carry torch.dtype objects, not stringified dtypes — locks
    the [_TensorSpec typed dtype] note in the RFC."""
    sch = EncoderScheduler(_FakeWorker(tp_size=2, tp_rank=0), _FakeAdapter())
    msg = _mk_msg(
        "r0",
        data={
            "image_grid_thw": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "pixel_values": torch.randn(2, 3, dtype=torch.float16),
        },
    )
    meta_msgs, tensor_lists, specs_lists = sch._strip_and_lift([msg])
    assert len(specs_lists) == 1
    specs = specs_lists[0]
    assert len(specs) == 2
    assert isinstance(meta_msgs[0].data.data["pixel_values"], dict)
    assert meta_msgs[0].data.data["pixel_values"]["_tensor_placeholder"] == (
        "pixel_values"
    )
    dtypes = {spec.dtype for spec in specs}
    assert torch.float16 in dtypes
    assert torch.long in dtypes
    # All dtypes are real torch.dtype objects, not strings.
    for spec in specs:
        assert isinstance(spec.dtype, torch.dtype)
