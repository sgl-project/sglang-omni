# SPDX-License-Identifier: Apache-2.0
"""Session value validation and stage state ownership without worker processes."""
import queue
import threading
from dataclasses import asdict

import msgpack
import pytest

from sglang_omni.admission import QueueFullError
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    SESSION_METADATA_KEY,
    OutputChunk,
    ResourceUsage,
    SessionRef,
    TimedChunk,
    wire_size,
)
from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.scheduling.session import SessionHooks, SessionScheduler
from tests.unit_test.fixtures.session_pipeline import compute_registered


class Hooks(SessionHooks):
    def __init__(self, name, events):
        self.name, self.events = name, events

    def open(self, ref, request):
        self.events.put(("open", self.name, ref.session_id))
        return {"id": ref.session_id}

    def close(self, state):
        self.events.put(("close", self.name, state["id"]))


def test_open_usage_failure_releases_state():

    class BrokenUsage(Hooks):
        def usage(self, state):
            raise RuntimeError("usage failed")

    events = queue.Queue()
    scheduler = SessionScheduler(BrokenUsage("source", events))
    request = OmniRequest(
        None,
        metadata={
            SESSION_METADATA_KEY: {"op": "open", "ref": asdict(SessionRef("one"))}
        },
    )
    with pytest.raises(RuntimeError, match="usage failed"):
        compute_registered(scheduler, StagePayload("one-open", request, {}))
    assert events.get_nowait()[0] == "open"
    assert events.get_nowait()[0] == "close"
    with pytest.raises(RuntimeError, match="usage failed"):
        compute_registered(scheduler, StagePayload("one-open-again", request, {}))


def test_stage_capacity_is_aggregate():

    class SizedHooks(Hooks):
        def open(self, ref, request):
            return {"id": ref.session_id, "bytes": 2}

        def abort(self, state, ref):
            state["bytes"] = 0

        def usage(self, state):
            return ResourceUsage(bytes=state["bytes"])

    events = queue.Queue()
    scheduler = SessionScheduler(SizedHooks("source", events), max_state_bytes=3)

    def invoke(sid, op, epoch=0):
        ref = asdict(SessionRef(sid, epoch=epoch))
        request = OmniRequest(
            None, metadata={SESSION_METADATA_KEY: {"op": op, "ref": ref}}
        )
        return compute_registered(scheduler, StagePayload(sid + op, request, {}))

    invoke("one", "open")
    with pytest.raises(QueueFullError):
        invoke("two", "open")
    assert events.get_nowait() == ("close", "source", "two")
    invoke("one", "abort", epoch=1)
    invoke("two", "open")
    scheduler.stop()
    closed = sorted(events.get_nowait()[2] for _ in range(events.qsize()))
    assert closed == ["one", "two"]


def test_malformed_command_fails_inside_the_request_boundary():

    scheduler = SessionScheduler(Hooks("source", queue.Queue()))
    worker = threading.Thread(target=scheduler.start)
    worker.start()
    try:
        request = OmniRequest(None, metadata={"omni_session": {"op": "append"}})
        scheduler.inbox.put(
            IncomingMessage("bad", "new_request", StagePayload("bad", request, {}))
        )
        output = scheduler.outbox.get(timeout=5)
        assert output.request_id == "bad"
        assert output.type == "error"
        assert isinstance(output.data, KeyError)
        request = OmniRequest(
            None,
            metadata={"omni_session": {"op": "open", "ref": asdict(SessionRef("ok"))}},
        )
        scheduler.inbox.put(
            IncomingMessage("open", "new_request", StagePayload("open", request, {}))
        )
        assert scheduler.outbox.get(timeout=5).type == "result"
    finally:
        scheduler.stop()
        worker.join(timeout=5)


@pytest.mark.parametrize("configured", [False, True])
def test_ordinary_request_uses_handler_or_reports_scoped_error(configured):

    def compute(payload):
        payload.data = {"ordinary": True}
        return payload

    kwargs = {"compute_fn": compute} if configured else {}
    scheduler = SessionScheduler(Hooks("source", queue.Queue()), **kwargs)
    worker = threading.Thread(target=scheduler.start)
    worker.start()
    try:
        payload = StagePayload("ordinary", OmniRequest(None), {})
        scheduler.inbox.put(IncomingMessage("ordinary", "new_request", payload))
        output = scheduler.outbox.get(timeout=5)
        assert output.request_id == "ordinary"
        if configured:
            assert output.type == "result"
            assert output.data.data == {"ordinary": True}
        else:
            assert output.type == "error"
            assert isinstance(output.data, ValueError)
            assert "ordinary requests" in str(output.data)
        request = OmniRequest(
            None,
            metadata={
                "omni_session": {
                    "op": "open",
                    "ref": asdict(SessionRef("after-ordinary")),
                }
            },
        )
        scheduler.inbox.put(
            IncomingMessage("open", "new_request", StagePayload("open", request, {}))
        )
        assert scheduler.outbox.get(timeout=5).type == "result"
    finally:
        scheduler.stop()
        worker.join(timeout=5)
    assert not worker.is_alive()


@pytest.mark.parametrize("size", [0, 255, 256, 65535, 65536])
def test_binary_chunk_wire_size_matches_msgpack(size, monkeypatch):

    chunk = TimedChunk("audio", 0, 80, 0, b"x" * size, format="pcm16")
    output = OutputChunk(
        SessionRef("session"),
        0,
        0,
        **{key: value for key, value in asdict(chunk).items() if key != "seq"},
    )
    pack = msgpack.packb
    values = [asdict(chunk), asdict(output)]
    expected = [len(pack(value, use_bin_type=True)) for value in values]
    encoded_payloads = []

    def record(value, **kwargs):
        encoded_payloads.append(len(value["payload"]))
        return pack(value, **kwargs)

    monkeypatch.setattr(msgpack, "packb", record)
    assert [wire_size(value) for value in values] == expected
    assert all(size == 0 for size in encoded_payloads)


def test_session_commands_run_in_arrival_order_even_when_one_is_aborted():

    class BlockingHooks(Hooks):
        def __init__(self):
            super().__init__("source", queue.Queue())
            self.entered, self.release = threading.Event(), threading.Event()

        def append(self, state, chunk, payload, context):
            self.events.put(("append", payload.request_id))
            if payload.request_id == "first":
                self.entered.set()
                self.release.wait(5)
            return payload

    hooks = BlockingHooks()
    scheduler = SessionScheduler(hooks, max_concurrency=3)

    def command(rid, op):
        return StagePayload(
            rid,
            OmniRequest(
                None,
                metadata={
                    SESSION_METADATA_KEY: {
                        "op": op,
                        "ref": asdict(SessionRef("s")),
                        "chunk": asdict(TimedChunk("audio", 0, 20, 0, b"x")),
                    }
                },
            ),
            {},
        )

    compute_registered(scheduler, command("open", "open"))
    payloads = [command(rid, "append") for rid in ("first", "second", "third")]
    for payload in payloads:
        scheduler.inbox.put(IncomingMessage(payload.request_id, "new_request", payload))
    messages = [scheduler.inbox.get_nowait() for _ in payloads]
    threads = [threading.Thread(target=scheduler._compute, args=(messages[0].data,))]
    threads[0].start()
    assert hooks.entered.wait(5)
    scheduler._aborted.add("second")
    assert scheduler._consume_if_aborted("second")
    threads.append(
        threading.Thread(target=scheduler._compute, args=(messages[2].data,))
    )
    threads[1].start()
    threads[1].join(0.2)
    assert threads[1].is_alive(), "third command ran before the first finished"
    hooks.release.set()
    for thread in threads:
        thread.join(5)
    assert not any(thread.is_alive() for thread in threads)
    order = []
    while not hooks.events.empty():
        event = hooks.events.get_nowait()
        if event[0] == "append":
            order.append(event[1])
    assert order == ["first", "third"]
    assert not scheduler._orders and not scheduler._tickets


def test_command_finished_by_abort_before_running_does_not_wait():

    events = queue.Queue()
    scheduler = SessionScheduler(Hooks("source", events), max_concurrency=2)

    def command(rid, op):
        return StagePayload(
            rid,
            OmniRequest(
                None,
                metadata={
                    SESSION_METADATA_KEY: {
                        "op": op,
                        "ref": asdict(SessionRef("s")),
                        "chunk": asdict(TimedChunk("audio", 0, 20, 0, b"x")),
                    }
                },
            ),
            {},
        )

    compute_registered(scheduler, command("open", "open"))
    payload = command("late", "close")
    scheduler.inbox.put(IncomingMessage("late", "new_request", payload))
    message = scheduler.inbox.get_nowait()
    # A request-level abort consumed the number first; the command still runs.
    scheduler._aborted.add("late")
    assert scheduler._consume_if_aborted("late")
    errors = []

    def run():
        try:
            scheduler._compute(message.data)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    worker.join(5)
    assert not worker.is_alive(), "command waited for a number that was already served"
    assert not errors, errors
    seen = []
    while not events.empty():
        seen.append(events.get_nowait()[0])
    assert seen == ["open", "close"]
    assert not scheduler._orders and not scheduler._tickets
