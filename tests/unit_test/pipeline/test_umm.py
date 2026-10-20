# SPDX-License-Identifier: Apache-2.0
"""UMM controller contracts with three fake adapters and real stages on CPU."""

from __future__ import annotations

import asyncio
import base64
import json
import pickle
import threading
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from sglang_omni.admission import QueueFullError
from sglang_omni.comm.data_ref import DataRef
from sglang_omni.config.schema import FactoryArgs
from sglang_omni.pipeline.local_dispatch import LocalStageDispatcher
from sglang_omni.pipeline.stage.stream_queue import StreamQueue
from sglang_omni.pipeline.umm import (
    CONTEXT_KEY,
    CONTINUATION_METADATA_KEY,
    CONTROLLER_FACTORY_PATH,
    OUTPUT_FACTORY_PATH,
    ROUTE_FN_PATH,
    STREAM_DONE_FN_PATH,
    ContextRef,
    Continuation,
    UMMController,
    UMMDispatch,
    UMMLimits,
    UMMOutput,
    UMMPipelineConfig,
    UMMStep,
    create_umm_controller,
    create_umm_output_stage,
    route_umm,
    umm_stream_done_targets,
)
from sglang_omni.proto import DataAckMessage, DataReadyMessage
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SESSION_METADATA_KEY
from sglang_omni.relay.shm import ShmRelay
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.serve.openai_errors import generation_error_status_code
from tests.unit_test.fixtures.pipeline_fakes import (
    RecordingStageControlPlane,
    block_on_inbox,
)
from tests.unit_test.pipeline.helpers import (
    build_compiled_process_topology,
    make_stage,
    stage,
)

SPLIT_PHASES = {"reason": "reasoner", "generate": "generator"}
BOUNDS = {
    "max_dispatches": 17,
    "max_outputs": 17,
    "max_active": 32,
    "max_retained_bytes": 64 * 1024 * 1024,
    "timeout_s": 3600.0,
}


def limits(**overrides: int | float) -> UMMLimits:
    return UMMLimits(**{**BOUNDS, **overrides})


def make_controller(adapter, phases=SPLIT_PHASES, **kwargs) -> UMMController:
    kwargs.setdefault("limits", limits())
    return UMMController(adapter, phases=phases, output_stage="output", **kwargs)


class Recorder:
    def __init__(self) -> None:
        self.closed: list[str] = []

    def usage(self, context) -> int:
        return 0

    def close(self, context, reason) -> None:
        self.closed.append(reason)


class SplitAdapter(Recorder):
    """Cosmos 3 Nano: a reasoner decides, a generator renders, media re-enters."""

    def __init__(self) -> None:
        super().__init__()
        self.histories: dict[str, dict] = {}

    def start(self, context):
        self.histories[context.execution_id] = {"messages": [], "bytes": 0}
        self.remember(context, {"role": "user", "content": context.request.inputs})
        return self.reason(context, ())

    def remember(self, context, message) -> None:
        # Incremental accounting: each appended message is measured once.
        history = self.histories[context.execution_id]
        history["messages"].append(message)
        history["bytes"] += len(json.dumps(message))

    def reason(self, context, outputs):
        final_only = context.remaining_dispatches < 3
        messages = list(self.histories[context.execution_id]["messages"])
        request = OmniRequest({"messages": messages}, {"final_only": final_only})
        return UMMStep("reason", request, outputs)

    def advance(self, context, phase, result):
        if phase == "reason":
            text = (UMMOutput("text", result["text"]),) if result["text"] else ()
            if result["kind"] == "final":
                return UMMStep(outputs=text)
            else:
                pass
            self.remember(context, {"role": "assistant", "content": result["prompt"]})
            return UMMStep("generate", OmniRequest({"prompt": result["prompt"]}), text)
        else:
            pass
        media = result.get("media")
        if media is None:
            # A zero-media turn, such as a filtered generation, is not an error.
            self.remember(context, {"role": "user", "content": "no media returned"})
            return self.reason(context, ())
        else:
            pass
        self.remember(context, {"role": "user", "content": {"image": media}})
        return self.reason(context, (UMMOutput("image", media),))

    def usage(self, context) -> int:
        return self.histories[context.execution_id]["bytes"]

    def close(self, context, reason) -> None:
        self.histories.pop(context.execution_id)
        super().close(context, reason)


def split_adapter(model_path: str) -> SplitAdapter:
    return SplitAdapter()


class OneCallAdapter(Recorder):
    """SenseNova-U1 think mode: one native call returns thinking and an image."""

    def start(self, context):
        request = OmniRequest({"prompt": context.request.inputs}, {"think_mode": True})
        return UMMStep("generate", request)

    def advance(self, context, phase, result):
        outputs = (
            [UMMOutput("text", result["think_text"])] if result["think_text"] else []
        )
        outputs += [UMMOutput("image", image) for image in result["images"]]
        return UMMStep(outputs=tuple(outputs))


class SharedContextAdapter(Recorder):
    """BAGEL: generation conditions on the understanding context it receives."""

    def __init__(self, *, consume: bool = True) -> None:
        super().__init__()
        self.consume = consume

    def start(self, context):
        return UMMStep(
            "understand", OmniRequest(context.request.inputs, {"think": True})
        )

    def advance(self, context, phase, result):
        if phase == "generate":
            # Early completion: no further understanding turn after the image.
            return UMMStep(outputs=(UMMOutput("image", result["image"]),))
        else:
            pass
        think = (UMMOutput("text", result["think"]),)
        if not self.consume:
            return UMMStep(outputs=think)
        else:
            pass
        return UMMStep(
            "generate", OmniRequest({"cfg": 4.0}), think, context=result[CONTEXT_KEY]
        )


class ParallelAdapter(Recorder):
    """LLaDA2-Uni: an image decodes while the model keeps writing."""

    def start(self, context):
        return UMMStep(
            dispatches=(
                UMMDispatch("reason", OmniRequest({"think": True})),
                UMMDispatch("generate", OmniRequest({"prompt": "a cat"})),
            )
        )

    def advance(self, context, phase, result):
        if phase == "reason":
            return UMMStep(outputs=(UMMOutput("text", result["text"]),))
        else:
            return UMMStep(outputs=(UMMOutput("image", result["media"]),))


def reasoner(decisions):
    script = iter(decisions)

    def run(payload):
        return next(script)

    return run


def generator(media):
    script = iter(media)

    def run(payload):
        item = next(script)
        return {} if item is None else {"media": item}

    return run


def unified(payload):
    return {"think_text": "plan", "images": [{"url": "inline-image-0"}]}


def bagel(payload):
    ref = payload.request.metadata.get(CONTEXT_KEY)
    if ref is None:
        return {"think": "plan", CONTEXT_KEY: ContextRef("bagel", "kv-1")}
    else:
        assert ref == ContextRef("bagel", "kv-1")
        return {"image": {"url": "inline-image-1"}}


def generate(prompt, text=""):
    return {"kind": "generate", "text": text, "prompt": prompt}


def final(text):
    return {"kind": "final", "text": text, "prompt": None}


def initial(request_id="req", *, stream=False):
    request = OmniRequest("draw, inspect, and revise", {"stream": stream})
    return StagePayload(request_id, request, request.inputs)


def reply(dispatched, data):
    return StagePayload(dispatched.request_id, dispatched.request, data)


def drive(controller, stages, payload):
    trace = []
    current = controller.advance(payload)
    while current is not None and (
        (target := route_umm(payload.request_id, current)) != "output"
    ):
        trace.append(
            (current.request.metadata[CONTINUATION_METADATA_KEY].phase, target)
        )
        current = controller.advance(reply(current, stages[target](current)))
    return current, trace


def drain(controller):
    messages = []
    while not controller.outbox.empty():
        messages.append(controller.outbox.get_nowait())
    return messages


SPLIT = (
    SplitAdapter,
    SPLIT_PHASES,
    lambda: {
        "reasoner": reasoner(
            [generate("a cat", "drawing"), generate("a dog"), final("done")]
        ),
        "generator": generator([{"url": "cat"}, None]),
    },
    [
        ("reason", "reasoner"),
        ("generate", "generator"),
        ("reason", "reasoner"),
        ("generate", "generator"),
        ("reason", "reasoner"),
    ],
    ["text", "image", "text"],
)
ONE_CALL = (
    OneCallAdapter,
    {"generate": "unified"},
    lambda: {"unified": unified},
    [("generate", "unified")],
    ["text", "image"],
)
SHARED = (
    SharedContextAdapter,
    {"understand": "bagel", "generate": "bagel"},
    lambda: {"bagel": bagel},
    [("understand", "bagel"), ("generate", "bagel")],
    ["text", "image"],
)


@pytest.mark.parametrize(
    "adapter_cls, phases, stages, expected_trace, expected_kinds",
    [SPLIT, ONE_CALL, SHARED],
    ids=["split-engines", "one-engine", "shared-context"],
)
def test_each_execution_shape_completes_with_ordered_segments(
    adapter_cls, phases, stages, expected_trace, expected_kinds
):
    adapter = adapter_cls()
    controller = make_controller(adapter, phases)
    phase_streams = []

    def recording(run):
        def record(payload):
            phase_streams.append(payload.request.params["stream"])
            return run(payload)

        return record

    phase_stages = {name: recording(run) for name, run in stages().items()}
    result, trace = drive(controller, phase_stages, initial(stream=True))
    assert trace == expected_trace
    # Only the controller streams. A phase stage that streamed would have no
    # stream target.
    assert phase_streams == [False] * len(trace)
    streamed = [message.data for message in drain(controller)]
    assert [segment["kind"] for segment in streamed] == expected_kinds
    assert [segment["segment_index"] for segment in streamed] == list(
        range(len(streamed))
    )
    assert len({segment["execution_id"] for segment in streamed}) == 1
    # The final result does not repeat media the client already streamed.
    assert result.data["segments"] == [
        {name: value for name, value in segment.items() if name != "data"}
        for segment in streamed
    ]
    assert adapter.closed == ["completed"]
    assert not controller.contexts and not getattr(adapter, "histories", None)


def test_stale_duplicate_forged_and_lost_continuations_never_advance():
    adapter = SplitAdapter()
    controller = make_controller(adapter)
    first = controller.advance(initial())
    token = first.request.metadata[CONTINUATION_METADATA_KEY]
    forged = reply(first, generate("x"))
    forged.request = replace(
        first.request,
        metadata={CONTINUATION_METADATA_KEY: replace(token, nonce="forged")},
    )
    assert controller.advance(forged) is None
    second = controller.advance(reply(first, generate("a cat")))
    assert controller.advance(reply(first, generate("again"))) is None
    expected = second.request.metadata[CONTINUATION_METADATA_KEY]
    assert controller.contexts["req"].expected == {expected.stage: expected}
    rebuilt = StagePayload("req", OmniRequest(second.request.inputs), {"media": None})
    with pytest.raises(RuntimeError, match="lost its continuation"):
        controller.advance(rebuilt)
    assert adapter.closed == ["failed"] and not controller.contexts
    with pytest.raises(RuntimeError, match="inactive request"):
        controller.advance(reply(second, {"media": {"url": "late"}}))


def test_abort_while_waiting_or_mid_transition_closes_once_and_publishes_nothing():
    adapter = SplitAdapter()
    controller = make_controller(adapter)
    controller.advance(initial())
    controller.abort("req")
    assert adapter.closed == ["aborted"] and not controller.contexts

    entered, release = threading.Event(), threading.Event()

    class Blocking(SplitAdapter):
        def advance(self, context, phase, result):
            entered.set()
            release.wait(timeout=5)
            return super().advance(context, phase, result)

    adapter = Blocking()
    controller = make_controller(adapter)
    dispatched = controller.advance(initial(stream=True))
    outcome = []
    worker = threading.Thread(
        target=lambda: outcome.append(
            controller.advance(reply(dispatched, generate("a cat", "drawing")))
        )
    )
    worker.start()
    assert entered.wait(timeout=5)
    controller.abort("req")
    assert adapter.closed == []
    release.set()
    worker.join(timeout=5)
    assert outcome == [None]
    assert adapter.closed == ["aborted"]
    assert controller.outbox.empty() and not controller.contexts
    controller.advance(initial("other"))
    controller.stop()
    controller.stop()
    assert adapter.closed == ["aborted", "shutdown"]


def test_deadline_fails_a_waiting_request_once_and_bounds_adapter_work():
    clock = [0.0]
    adapter = SplitAdapter()
    controller = make_controller(
        adapter, limits=limits(timeout_s=1), clock=lambda: clock[0]
    )
    dispatched = controller.advance(initial())
    clock[0] = 1.0
    controller.expire()
    controller.expire()
    admitted, error = drain(controller)
    assert (admitted.type, error.type) == ("admitted", "error")
    assert "timeout_s=1" in str(error.data) and adapter.closed == ["expired"]
    with pytest.raises(RuntimeError, match="inactive request"):
        controller.advance(reply(dispatched, generate("late")))

    class Slow(SplitAdapter):
        def advance(self, context, phase, result):
            clock[0] = 5.0
            return super().advance(context, phase, result)

    adapter = Slow()
    controller = make_controller(
        adapter, limits=limits(timeout_s=1), clock=lambda: clock[0]
    )
    clock[0] = 0.0
    dispatched = controller.advance(initial(stream=True))
    with pytest.raises(TimeoutError, match="deadline"):
        controller.advance(reply(dispatched, generate("a cat", "late text")))
    assert adapter.closed == ["expired"]
    assert controller.outbox.empty() and not controller.contexts


@pytest.mark.parametrize("wait", [None, block_on_inbox], ids=["shipped", "blocking"])
def test_scheduler_loop_sweeps_the_deadline_of_a_waiting_request(monkeypatch, wait):
    if wait is not None:
        # A loop that waits for work, so only the controller can wake it.
        wait(monkeypatch)
    else:
        pass
    controller = make_controller(SplitAdapter(), limits=limits(timeout_s=0.1))
    worker = threading.Thread(target=controller.start)
    worker.start()
    try:
        controller.inbox.put(IncomingMessage("req", "new_request", initial()))
        assert controller.outbox.get(timeout=2).type == "result"
        assert controller.outbox.get(timeout=2).type == "admitted"
        error = controller.outbox.get(timeout=2)
        assert error.type == "error" and "deadline" in str(error.data)
    finally:
        controller.stop()
        worker.join(timeout=2)
    assert not worker.is_alive() and not controller.contexts


def test_scheduler_loop_serves_requests_whose_deadline_outlasts_any_wait():
    # The deadline lies past the longest timeout a wait accepts.
    controller = make_controller(SplitAdapter(), limits=limits(timeout_s=1e12))
    worker = threading.Thread(target=controller.start)
    worker.start()
    try:
        for request_id in ("req", "other"):
            message = IncomingMessage(request_id, "new_request", initial(request_id))
            controller.inbox.put(message)
            assert controller.outbox.get(timeout=2).type == "result"
    finally:
        controller.stop()
        worker.join(timeout=2)
    assert not worker.is_alive() and controller.outbox.empty()


@pytest.mark.parametrize(
    "bounds, script, media, name, published, status",
    [
        (
            {"max_retained_bytes": 160},
            [generate("a cat")],
            [{"url": "x" * 200}],
            "max_retained_bytes",
            0,
            500,
        ),
        ({"max_retained_bytes": 10}, [], [], "max_retained_bytes", 0, 400),
        (
            {"max_outputs": 1},
            [generate("a cat", "drawing")],
            [{}],
            "max_outputs",
            1,
            500,
        ),
        (
            {"max_dispatches": 3},
            [generate("a cat"), generate("over")],
            [{"url": "cat"}],
            "max_dispatches",
            1,
            500,
        ),
    ],
    ids=["generated-bytes", "client-bytes", "outputs", "dispatches"],
)
def test_bounds_reject_the_whole_transition(
    bounds, script, media, name, published, status
):
    adapter = SplitAdapter()
    controller = make_controller(adapter, limits=limits(**bounds))
    stages = {"reasoner": reasoner(script), "generator": generator(media)}
    try:
        result, _ = drive(controller, stages, initial(stream=True))
    except RuntimeError as exc:
        error = exc
    else:
        # A client-caused rejection is reported through the outbox.
        assert result is None
        error = drain(controller).pop().data
    assert f"factory.{name}" in str(error)
    assert generation_error_status_code(RuntimeError(str(error))) == status
    assert adapter.closed == ["failed"] and not controller.contexts
    # Earlier transitions streamed their segments, the rejected one none.
    assert [m.type for m in drain(controller)] == ["stream"] * published


def test_admission_rejects_excess_and_session_requests_and_keeps_live_ones():
    controller = make_controller(SplitAdapter(), limits=limits(max_active=1))
    first = controller.advance(initial())
    assert controller.advance(initial("excess")) is None
    session = initial("session")
    session.request.metadata[SESSION_METADATA_KEY] = {"operation": "open"}
    assert controller.advance(session) is None
    errors = drain(controller)
    assert [error.request_id for error in errors] == ["excess", "session"]
    assert [
        generation_error_status_code(RuntimeError(str(error.data))) for error in errors
    ] == [503, 400]
    assert list(controller.contexts) == ["req"]
    done = controller.advance(reply(first, final("done")))
    assert route_umm("req", done) == "output"


def test_unconsumed_engine_context_fails_the_request():
    adapter = SharedContextAdapter(consume=False)
    controller = make_controller(adapter, {"understand": "bagel"})
    with pytest.raises(RuntimeError, match="unconsumed"):
        drive(controller, {"bagel": bagel}, initial())
    assert adapter.closed == ["failed"] and not controller.contexts


def umm_stages(**changes):
    controller = {
        "factory_path": CONTROLLER_FACTORY_PATH,
        "next": ["reasoner", "generator", "output"],
        "route_fn": ROUTE_FN_PATH,
        "stream_to": ["output"],
        "stream_done_to_fn": STREAM_DONE_FN_PATH,
        "factory": FactoryArgs(phases=SPLIT_PHASES, **{**BOUNDS, "max_active": 4}),
        **changes.get("orchestrator", {}),
    }
    output = {
        "factory_path": OUTPUT_FACTORY_PATH,
        "terminal": True,
        "can_accept_stream_before_payload": True,
        **changes.get("output", {}),
    }
    return [
        stage("orchestrator", **controller),
        stage("output", **output),
        stage("reasoner", next="orchestrator", process="reasoner"),
        stage(
            "generator",
            **{
                "next": "orchestrator",
                "process": "generator",
                **changes.get("generator", {}),
            },
        ),
    ]


@pytest.mark.parametrize(
    "changes, message",
    [
        ({}, None),
        ({"orchestrator": {"next": ["reasoner", "output"]}}, "next must name"),
        (
            {"orchestrator": {"factory": FactoryArgs(phases={"reason": "output"})}},
            "factory.phases",
        ),
        ({"orchestrator": {"stream_to": ["reasoner"]}}, "stream_to must be"),
        ({"generator": {"next": "output"}}, "whose next is"),
        ({"generator": {"stream_to": ["output"]}}, "generator.stream_to must be"),
        ({"output": {"process": "elsewhere"}}, "share a process"),
        (
            {
                "orchestrator": {
                    "factory": FactoryArgs(
                        phases=SPLIT_PHASES, **{**BOUNDS, "max_active": True}
                    )
                }
            },
            "factory.max_active must be",
        ),
    ],
    ids=[
        "valid",
        "undeclared-phase-stage",
        "phase-to-output",
        "no-relay",
        "phase-skips-controller",
        "phase-streams",
        "split",
        "boolean-bound",
    ],
)
def test_pipeline_config_accepts_only_the_umm_wiring(changes, message):
    def build():
        config = UMMPipelineConfig(
            model_path="model", entry_stage="orchestrator", stages=umm_stages(**changes)
        )
        build_compiled_process_topology(config)
        return config

    if message is None:
        config = build()
        assert config.entry_stage_admission_limit() == 4
        assert config.stage_factory_kwargs("orchestrator") == {
            "output_stage": "output",
            "stage_name": "orchestrator",
        }
    else:
        with pytest.raises(ValueError, match=message):
            build()


async def settle(stages):
    pending = [
        item.task
        for obj in stages.values()
        for item in obj.comm.pending.values()
        if item.task is not None
    ]
    if pending:
        await asyncio.wait_for(asyncio.gather(*pending), timeout=2)
    else:
        pass


def real_stages(controller, relay, *, reasoner_script, generator_script):
    stages = {}
    crossed = []

    class Bridge(RecordingStageControlPlane):
        async def send_to_stage(self, target, endpoint, msg):
            await super().send_to_stage(target, endpoint, msg)
            raw = json.loads(json.dumps(msg.to_dict()))
            if isinstance(msg, DataAckMessage):
                stages[target].comm.ack_transfer(DataAckMessage.from_dict(raw))
                return
            else:
                pass
            ready = DataReadyMessage.from_dict(raw)
            ref = DataRef.from_dict(ready.data_ref)
            header = pickle.loads(base64.b64decode(ref.header))
            token = header.request.metadata.get(CONTINUATION_METADATA_KEY)
            crossed.append((msg.from_stage, target, isinstance(token, Continuation)))
            await stages[target].on_data_ready(ready)

    names = ("orchestrator", "output", "reasoner", "generator")
    endpoints = {name: "inproc://" + name for name in names}
    dispatcher = LocalStageDispatcher()
    local = {"local_dispatcher": dispatcher}
    stages["orchestrator"] = make_stage(
        name="orchestrator",
        get_next=route_umm,
        endpoints=endpoints,
        control_plane=Bridge(),
        relay=ShmRelay("orchestrator", device="cpu", credits=1),
        scheduler=controller,
        stream_targets=["output"],
        get_stream_done_targets=umm_stream_done_targets,
        same_process_targets={"output"},
        **local,
    )
    stages["output"] = make_stage(
        name="output",
        endpoints=endpoints,
        scheduler=relay,
        is_terminal=True,
        can_accept_stream_before_payload=True,
        same_process_targets={"orchestrator"},
        **local,
    )
    stages["output"].stream_queue = StreamQueue()
    stages["output"].control_plane.send_stream = AsyncMock()
    dispatcher.register_many([stages["orchestrator"], stages["output"]])
    for name, script in (
        ("reasoner", reasoner(reasoner_script)),
        ("generator", generator(generator_script)),
    ):
        stages[name] = make_stage(
            name=name,
            get_next=lambda *_: "orchestrator",
            endpoints=endpoints,
            control_plane=Bridge(),
            relay=ShmRelay(name, device="cpu", credits=1),
        )
        stages[name].script = script
    return stages, crossed


def start_relay():
    relay = create_umm_output_stage()

    def serve():
        # The stage warms its scheduler on the serving thread, then starts it.
        relay.warm_up_serving_thread()
        relay.start()

    thread = threading.Thread(target=serve)
    thread.start()
    return relay, thread


async def stop_real_stages(stages, controller, relay, relay_thread):
    controller.stop()
    relay.stop()
    relay_thread.join(timeout=2)
    tasks = [task for obj in stages.values() for task in obj.comm.send_workers.values()]
    for obj in stages.values():
        await obj.comm.close()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    else:
        pass


async def run_real_stages(stages, controller, rid="req"):
    """Route through real Stage objects, the orchestrator through its outbox."""

    async def orchestrate():
        item = controller.inbox.get_nowait().data
        controller.emit_result(rid, controller.advance(item), controller.outbox)
        stages["orchestrator"].end_outbox_drain()
        await asyncio.wait_for(stages["orchestrator"].drain_outbox_external(), 2)
        await settle(stages)

    await orchestrate()
    while not stages["reasoner"].scheduler.inbox.empty() or not (
        stages["generator"].scheduler.inbox.empty()
    ):
        for name in ("reasoner", "generator"):
            inbox = stages[name].scheduler.inbox
            if inbox.empty():
                continue
            else:
                pass
            item = inbox.get_nowait().data
            result = reply(item, stages[name].script(item))
            await asyncio.wait_for(stages[name].route_result(rid, result), 2)
            await settle(stages)
            await orchestrate()
    output = stages["output"]
    for _ in range(200):
        output.end_outbox_drain()
        await asyncio.wait_for(output.drain_outbox_external(), 2)
        if output.control_plane.completions:
            break
        else:
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_real_stages_loop_over_relays_and_stream_through_the_output_stage():
    controller = create_umm_controller(
        "model",
        adapter=f"{__name__}.split_adapter",
        phases=SPLIT_PHASES,
        output_stage="output",
        **{
            "max_dispatches": 17,
            "max_outputs": 17,
            "max_active": 4,
            "max_retained_bytes": 1 << 20,
            "timeout_s": 60.0,
        },
    )
    relay, relay_thread = start_relay()
    stages, crossed = real_stages(
        controller,
        relay,
        reasoner_script=[
            generate("a cat", "drawing"),
            generate("a dog"),
            final("done"),
        ],
        generator_script=[{"url": "cat"}, {"url": "dog"}],
    )
    try:
        stages["orchestrator"].active_requests.add("req")
        await stages["orchestrator"].execute(initial(stream=True))
        await run_real_stages(stages, controller)
        output = stages["output"]
        assert [c.success for c in output.control_plane.completions] == [True]
        streams = [call.args[0] for call in output.control_plane.send_stream.mock_calls]
        assert [message.chunk_id for message in streams] == [0, 1, 2, 3]
        assert [message.chunk["kind"] for message in streams] == [
            "text",
            "image",
            "image",
            "text",
        ]
        assert [message.chunk["data"] for message in streams][1:3] == [
            {"url": "cat"},
            {"url": "dog"},
        ]
        result = output.control_plane.completions[0].result
        assert [s["segment_index"] for s in result["segments"]] == [0, 1, 2, 3]
        # Five dispatches cross the relay to a phase and back, each carrying
        # its continuation. The completed request reaches the output locally.
        assert len(crossed) == 10 and all(token for *_, token in crossed)
        assert stages["orchestrator"].control_plane.completions == []
        for obj in stages.values():
            assert not obj.active_requests and not obj.comm.pending
        assert not controller.contexts
    finally:
        await stop_real_stages(stages, controller, relay, relay_thread)


@pytest.mark.asyncio
async def test_parallel_phases_leave_in_one_routing_pass_and_join_once():
    adapter = ParallelAdapter()
    controller = make_controller(adapter)
    relay, relay_thread = start_relay()
    stages, crossed = real_stages(
        controller, relay, reasoner_script=[final("plan")], generator_script=[{}]
    )
    try:
        stages["orchestrator"].active_requests.add("req")
        await stages["orchestrator"].execute(initial(stream=True))
        await run_real_stages(stages, controller)
        output = stages["output"]
        assert [c.success for c in output.control_plane.completions] == [True]
        chunks = [
            call.args[0].chunk for call in output.control_plane.send_stream.mock_calls
        ]
        assert [(c["segment_index"], c["kind"]) for c in chunks] == [
            (0, "text"),
            (1, "image"),
        ]
        assert crossed[:2] == [
            ("orchestrator", "reasoner", True),
            ("orchestrator", "generator", True),
        ]
        for obj in stages.values():
            assert not obj.active_requests and not obj.comm.pending
        assert not controller.contexts and adapter.closed == ["completed"]
    finally:
        await stop_real_stages(stages, controller, relay, relay_thread)
    adapter = ParallelAdapter()
    controller = make_controller(adapter, {"reason": "solo", "generate": "solo"})
    with pytest.raises(ValueError, match="already runs a phase"):
        controller.advance(initial())
    assert adapter.closed == ["failed"] and not controller.contexts


@pytest.mark.asyncio
async def test_phase_results_queued_together_both_leave_the_stage():
    class Rewriter(ParallelAdapter):
        """The first text starts another turn while the image still decodes."""

        def advance(self, context, phase, result):
            step = super().advance(context, phase, result)
            if phase == "reason" and result["text"] == "plan":
                turn = UMMDispatch("reason", OmniRequest({"think": True}))
                return replace(step, dispatches=(turn,))
            else:
                return step

    controller = make_controller(Rewriter())
    orchestrator = make_stage(
        name="orchestrator",
        get_next=route_umm,
        scheduler=controller,
        stream_targets=["output"],
        get_stream_done_targets=umm_stream_done_targets,
    )
    orchestrator.send_to_stage = AsyncMock()
    orchestrator.send_stream_to_target = AsyncMock()
    started = controller.advance(initial(stream=True))
    # Both phase results reached the stage before either one is routed.
    orchestrator.active_requests.add("req")
    for phase, data in (("reasoner", {"text": "plan"}), ("generator", {"media": "a"})):
        result = controller.advance(reply(started.payloads[phase], data))
        controller.emit_result("req", result, controller.outbox)
    orchestrator.end_outbox_drain()
    await orchestrator.drain_outbox_external()
    sent = orchestrator.send_stream_to_target.await_args_list
    assert [call.args[1]["kind"] for call in sent] == ["text", "image"]
    assert orchestrator.send_to_stage.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["deadline", "shed"])
async def test_waiting_controller_reports_its_deadline_through_its_stage(ending):
    clock = [0.0]
    waiting = []

    class Planner(SplitAdapter):
        def start(self, context):
            if ending == "shed":
                # The plan for a new request leaves no time for a waiting one.
                for other in waiting:
                    other.shed()
            else:
                pass
            waiting.append(context)
            return super().start(context)

    adapter = Planner()
    # A controller stage named other than the default, which errors must name.
    controller = make_controller(
        adapter,
        limits=limits(timeout_s=1),
        stage_name="planner",
        clock=lambda: clock[0],
    )
    orchestrator = make_stage(name="planner", get_next=route_umm, scheduler=controller)
    orchestrator.send_to_stage = AsyncMock()
    orchestrator.active_requests.add("req")
    await orchestrator.execute(initial())
    dispatched = controller.advance(controller.inbox.get_nowait().data)
    controller.emit_result("req", dispatched, controller.outbox)
    orchestrator.end_outbox_drain()
    await orchestrator.drain_outbox_external()
    # The dispatch left the stage, which no longer tracks the request.
    assert orchestrator.send_to_stage.await_count == 1
    assert not orchestrator.active_requests
    if ending == "shed":
        controller.advance(initial("other"))
    else:
        clock[0] = 2.0
    controller.expire()
    controller.expire()
    orchestrator.end_outbox_drain()
    await orchestrator.drain_outbox_external()
    # The failure makes the coordinator abort the request's queued phase work.
    (failure,) = orchestrator.control_plane.completions
    assert not failure.success and "timeout_s=1" in failure.error
    assert not orchestrator.active_requests and "req" in orchestrator.aborted
    # The coordinator rebuilds the failure from its text for the client.
    error = QueueFullError.from_message(failure.error)
    status, wording, reason = {
        "deadline": (500, "deadline exceeded", "expired"),
        "shed": (503, "Retry it later", "shed"),
    }[ending]
    assert generation_error_status_code(error) == status and wording in str(error)
    assert "stages.planner.factory.timeout_s" in str(error)
    assert adapter.closed == [reason]
    assert list(controller.contexts) == ([] if ending == "deadline" else ["other"])
    with pytest.raises(RuntimeError, match="inactive request"):
        controller.advance(reply(dispatched, generate("late")))
