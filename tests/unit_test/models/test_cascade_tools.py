# SPDX-License-Identifier: Apache-2.0
"""Tool calls in the streaming cascade: the generator's tool rounds, the dialog's history of them, background tasks and their follow-up turns."""

import asyncio
import base64
import io

import pytest
from PIL import Image

from examples.streaming_cascade.tools.image_generation import generate_image
from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.streaming_cascade.base import (
    messages_chunk,
    messages_of,
    text_chunk,
    transcript_chunk,
)
from sglang_omni.models.streaming_cascade.config import ComponentGraphConfig
from sglang_omni.models.streaming_cascade.dialog import prompt_messages
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import (
    INTERRUPT,
    MESSAGES,
    REPLY_DELIVERED,
    REPLY_PROGRESS,
    SPEECH_STARTED,
    SPEECH_STOPPED,
    TASK_RESULT,
    TASK_UPDATE,
    TOOL_CALL,
    TOOL_EXCHANGE,
    TOOL_RESPONSE,
    TURN_DONE,
    TURN_QUEUED,
    TURN_STARTED,
    control_type,
)
from sglang_omni.proto.tools import (
    TaskResult,
    ToolCall,
    ToolExchange,
    ToolResponse,
    task_result_chunk,
    tool_call_chunk,
    tool_exchange_chunk,
    tool_response_chunk,
)
from tests.unit_test.fixtures.cascade_fakes import build, drain
from tests.unit_test.fixtures.cascade_tools import (  # noqa: F401  # pytest fixture
    TOOL_GATES,
    TOOL_STARTED,
    modelscope_service,
    scripted_chat,
    text_events,
    tool_call_events,
)
from tests.unit_test.fixtures.session_outputs import control_chunk

PROFILE = (
    "examples/streaming_cascade/profiles/qwen3-asr-qwen3.8-27b-image-tool/launch.yaml"
)
DRAW_TOOL = {
    "type": "function",
    "function": {
        "name": "draw",
        "parameters": {"type": "object", "properties": {"gate": {"type": "string"}}},
    },
}
FIXTURES = "tests.unit_test.fixtures.cascade_tools"


async def until(condition, timeout_s: float = 5.0) -> None:
    """Yield to the loop until condition() holds."""
    deadline = asyncio.get_running_loop().time() + timeout_s
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, "condition never held"
        await asyncio.sleep(0.001)


def generator_for(chat) -> object:
    return build(
        {
            "kind": "openai_chat",
            "base_url": chat.base_url,
            "model": "m",
            "tools": [DRAW_TOOL],
            "tool_response_timeout_s": 0.5,
        }
    )


@pytest.mark.asyncio
async def test_a_tool_round_waits_for_its_response_and_continues_the_reply_with_it(
    scripted_chat,
):
    scripted_chat.replies = [
        tool_call_events("draw", {"gate": "cat"}, text="\n\nOK. "),
        text_events("Started."),
    ]
    generator = generator_for(scripted_chat)
    response = ToolResponse(
        id="call_a", name="draw", response={"status": "accepted", "task_id": "task_1"}
    )
    outputs = await drain(
        generator,
        [
            messages_chunk([], 3),
            lambda: until(lambda: 3 in generator.pending_responses),
            lambda: generator.control(tool_response_chunk([response], 3)),
        ],
    )
    assert [o.format for o in outputs if o.format != "session"] == [
        "text_delta",
        TOOL_CALL,
        TOOL_EXCHANGE,
        "text_delta",
    ]
    assert control_type(outputs[-1]) == TURN_DONE and all(o.turn == 3 for o in outputs)
    call = next(o for o in outputs if o.format == TOOL_CALL).payload["calls"][0]
    assert (call["name"], call["arguments"]) == ("draw", {"gate": "cat"})
    exchange = next(o for o in outputs if o.format == TOOL_EXCHANGE).payload
    assert exchange["text_offset"] == len("OK. ")
    second = scripted_chat.requests[1]["messages"]
    assert second[-2]["role"] == "assistant" and second[-2]["content"] == "OK. "
    assert second[-2]["tool_calls"][0]["function"] == {
        "name": "draw",
        "arguments": '{"gate": "cat"}',
    }
    assert second[-1] == {
        "role": "tool",
        "content": '{"status": "accepted", "task_id": "task_1"}',
        "tool_call_id": "call_a",
    }
    assert scripted_chat.requests[0]["tools"] == [DRAW_TOOL]


@pytest.mark.asyncio
async def test_a_tool_that_does_not_answer_in_time_answers_with_an_error_and_the_reply_ends(
    scripted_chat,
):
    scripted_chat.replies = [tool_call_events("draw", {}), text_events("Sorry.")]
    outputs = await drain(generator_for(scripted_chat), [messages_chunk([], 1)])
    assert control_type(outputs[-1]) == TURN_DONE
    assert (
        "did not answer in time" in scripted_chat.requests[1]["messages"][-1]["content"]
    )


@pytest.mark.asyncio
async def test_an_interrupt_while_a_reply_waits_for_its_tools_ends_it_without_turn_done(
    scripted_chat,
):
    scripted_chat.replies = [tool_call_events("draw", {})]
    generator = generator_for(scripted_chat)
    outputs = await drain(
        generator,
        [
            messages_chunk([], 1),
            lambda: until(lambda: 1 in generator.pending_responses),
            lambda: generator.control(control_chunk(INTERRUPT, turn=1)),
        ],
    )
    assert TURN_DONE not in [control_type(o) for o in outputs]
    assert not generator.pending_responses and len(scripted_chat.requests) == 1


DRAW_CALL = ToolCall(id="call_a", name="draw", arguments={"gate": "cat"})
ACCEPTED = ToolResponse(
    id="call_a", name="draw", response={"status": "accepted", "task_id": "task_1"}
)


def roles(messages: list) -> list[tuple[str, str, bool]]:
    return [(m.role, m.content, bool(m.tool_calls)) for m in messages]


@pytest.mark.asyncio
async def test_the_prompt_expands_a_reply_tool_round_at_its_text_offset():
    dialog = build({"kind": "dialog", "system_prompt": "S"})
    await drain(
        dialog,
        [
            transcript_chunk("draw a cat", True),
            text_chunk("OK. ", 1),
            tool_exchange_chunk(
                ToolExchange(text_offset=4, calls=[DRAW_CALL], responses=[ACCEPTED]), 1
            ),
            text_chunk("Started.", 1),
            control_chunk(TURN_DONE, turn=1),
        ],
    )
    assert len(dialog.messages) == 3
    assert roles(prompt_messages(dialog.messages)) == [
        ("system", "S", False),
        ("user", "draw a cat", False),
        ("assistant", "OK. ", True),
        ("tool", '{"status": "accepted", "task_id": "task_1"}', False),
        ("assistant", "Started.", False),
    ]


@pytest.mark.asyncio
async def test_a_reply_cut_before_its_tool_round_keeps_the_call_but_not_the_text_after_the_cut():
    dialog = build({"kind": "dialog", "system_prompt": "S"})
    await drain(
        dialog,
        [
            transcript_chunk("draw a cat", True),
            text_chunk("Sure thing. I will draw. ", 1),
            lambda: dialog.control(
                control_chunk(REPLY_PROGRESS, turn=1, text_end=12, audio_ms=800)
            ),
            tool_exchange_chunk(
                ToolExchange(text_offset=25, calls=[DRAW_CALL], responses=[ACCEPTED]), 1
            ),
            control_chunk(SPEECH_STARTED),
            control_chunk(REPLY_DELIVERED, turn=1, audio_ms=900),
        ],
    )
    prompt = prompt_messages(dialog.messages)
    assert roles(prompt)[2:] == [
        ("assistant", "Sure thing.", True),
        ("tool", '{"status": "accepted", "task_id": "task_1"}', False),
    ]


def finished_task(status: str = "succeeded", follow_up: str = "speak") -> TimedChunk:
    return task_result_chunk(
        TaskResult(
            task_id="task_1",
            name="draw",
            status=status,
            summary="an image of a cat",
            error="boom" if status == "failed" else None,
            follow_up=follow_up,
        )
    )


async def answered_dialog(*after: object) -> tuple[object, list[TimedChunk]]:
    """A dialog whose first reply started task_1, with `after` fed next; a TaskResult among them reaches the dialog as a control."""
    dialog = build({"kind": "dialog", "system_prompt": "S"})
    outputs = await drain(
        dialog,
        [
            transcript_chunk("draw a cat", True),
            tool_exchange_chunk(
                ToolExchange(text_offset=0, calls=[DRAW_CALL], responses=[ACCEPTED]), 1
            ),
            text_chunk("Started.", 1),
            control_chunk(TURN_DONE, turn=1),
            *(
                (
                    (lambda event=entry: dialog.control(event))
                    if entry.format == TASK_RESULT
                    else entry
                )
                for entry in after
            ),
        ],
    )
    return dialog, outputs


@pytest.mark.asyncio
async def test_a_finished_task_opens_a_turn_whose_prompt_ends_with_a_notice_and_marks_the_tool_response():
    dialog, outputs = await answered_dialog(finished_task())
    assert [
        (control_type(o), o.turn)
        for o in outputs
        if control_type(o) in (TURN_QUEUED, TURN_STARTED)
    ][-2:] == [(TURN_QUEUED, 2), (TURN_STARTED, 2)]
    prompt = messages_of([o for o in outputs if o.format == MESSAGES][-1])
    assert (
        prompt[-1].role == "user"
        and "not said by the user" in prompt[-1].content
        and "an image of a cat" in prompt[-1].content
    )
    assert '"status": "succeeded"' in next(
        m.content for m in prompt if m.role == "tool"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hold, release",
    [
        (
            [transcript_chunk("another question", True)],
            [text_chunk("Answer.", 2), control_chunk(TURN_DONE, turn=2)],
        ),
        ([control_chunk(SPEECH_STARTED)], [control_chunk(SPEECH_STOPPED)]),
    ],
    ids=["a reply runs", "the user speaks"],
)
async def test_a_finished_task_waits_while_a_reply_runs_or_the_user_speaks(
    hold, release
):
    dialog = build({"kind": "dialog", "system_prompt": "S"})
    held = await drain(
        dialog,
        [
            transcript_chunk("draw a cat", True),
            text_chunk("Started.", 1),
            control_chunk(TURN_DONE, turn=1),
            *hold,
            lambda: dialog.control(finished_task()),
        ],
    )
    assert [o.turn for o in held if control_type(o) == TURN_STARTED] == [
        1,
        *([2] if hold[0].format != "session" else []),
    ]
    dialog.emit = lambda chunk: released.append(chunk)
    released: list[TimedChunk] = []
    for chunk in release:
        dialog.handle(chunk)
        dialog.respond_to_waiting()
    assert [o.turn for o in released if control_type(o) == TURN_STARTED] == [
        3 if hold[0].format != "session" else 2
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, follow_up", [("cancelled", "silent"), ("succeeded", "silent")]
)
async def test_a_silent_task_result_opens_no_turn_but_updates_the_tool_response(
    status, follow_up
):
    dialog, outputs = await answered_dialog(finished_task(status, follow_up))
    assert [o.turn for o in outputs if control_type(o) == TURN_STARTED] == [1]
    assert (
        dialog.messages[-1].tool_exchanges[0].responses[0].response["status"] == status
    )


@pytest.mark.asyncio
async def test_a_reply_cut_to_nothing_after_a_notice_is_dropped_without_joining_the_notice_into_the_next_question():
    dialog, _ = await answered_dialog(
        finished_task(),
        control_chunk(SPEECH_STARTED),
        control_chunk(REPLY_DELIVERED, turn=2, audio_ms=0),
        control_chunk(SPEECH_STOPPED),
    )
    dialog.emit = lambda chunk: None
    dialog.handle(transcript_chunk("what now", True))
    dialog.respond_to_waiting()
    assert [(m.role, m.is_notice) for m in dialog.messages][-2:] == [
        ("user", True),
        ("user", False),
    ]
    assert dialog.messages[-1].content == "what now"


def tool_node(**tools: dict) -> object:
    return build(
        {
            "kind": "async_tools",
            "max_jobs_per_session": 1,
            "max_artifact_bytes": 1000,
            "tools": tools,
        }
    )


GATED = {"implementation": f"{FIXTURES}.gated_tool", "timeout_s": 5}


async def release(gate: str) -> None:
    TOOL_GATES[gate].set()


def call_chunk(turn: int, name: str, **arguments: object) -> TimedChunk:
    return tool_call_chunk(
        [ToolCall(id=f"call_{turn}", name=name, arguments=arguments)], turn
    )


def task_events(outputs: list[TimedChunk]) -> list[tuple[str, str]]:
    return [
        (o.format, o.payload["status"])
        for o in outputs
        if o.format in (TASK_UPDATE, TASK_RESULT)
    ]


def responses(outputs: list[TimedChunk]) -> list[object]:
    return [
        r["response"]
        for o in outputs
        if o.format == TOOL_RESPONSE
        for r in o.payload["responses"]
    ]


@pytest.mark.asyncio
async def test_an_async_tool_answers_at_once_and_reports_its_task_without_a_turn():
    TOOL_GATES["a"], TOOL_STARTED["a"] = asyncio.Event(), asyncio.Event()
    tasks = tool_node(
        draw=GATED, echo={"implementation": f"{FIXTURES}.echo_tool", "mode": "sync"}
    )
    outputs = await drain(
        tasks,
        [
            call_chunk(1, "draw", gate="a"),
            lambda: TOOL_STARTED["a"].wait(),
            call_chunk(2, "echo", x=1),
            lambda: release("a"),
            lambda: until(lambda: not tasks.running_jobs()),
        ],
    )
    assert responses(outputs) == [
        {"status": "accepted", "task_id": "task_1"},
        {"echo": {"x": 1}},
    ]
    assert [o.turn for o in outputs if o.format == TOOL_RESPONSE] == [1, 2]
    assert task_events(outputs) == [
        (TASK_UPDATE, "created"),
        (TASK_UPDATE, "running"),
        (TASK_UPDATE, "succeeded"),
        (TASK_RESULT, "succeeded"),
    ]
    assert all(
        o.turn is None for o in outputs if o.format in (TASK_UPDATE, TASK_RESULT)
    )
    assert (
        next(o for o in outputs if o.format == TASK_RESULT).payload["follow_up"]
        == "speak"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupted_turn, ends", [(1, "cancelled"), (2, "succeeded")])
async def test_an_interrupt_cancels_only_the_tasks_of_its_turn(interrupted_turn, ends):
    TOOL_GATES["b"], TOOL_STARTED["b"] = asyncio.Event(), asyncio.Event()
    tasks = tool_node(draw=GATED)
    outputs = await drain(
        tasks,
        [
            call_chunk(1, "draw", gate="b"),
            lambda: TOOL_STARTED["b"].wait(),
            lambda: tasks.control(control_chunk(INTERRUPT, turn=interrupted_turn)),
            lambda: release("b"),
            lambda: until(lambda: not tasks.running_jobs()),
        ],
    )
    result = next(o for o in outputs if o.format == TASK_RESULT).payload
    assert (result["status"], result["follow_up"]) == (
        ends,
        "silent" if ends == "cancelled" else "speak",
    )


@pytest.mark.asyncio
async def test_a_call_the_node_took_after_its_turn_was_interrupted_starts_no_task():
    """A call already handed to the node when its reply's interrupt arrived started a task the interrupt was meant to stop."""
    tasks = tool_node(draw=GATED)
    outputs = await drain(
        tasks,
        [
            lambda: tasks.control(control_chunk(INTERRUPT, turn=1)),
            call_chunk(1, "draw", gate="never"),
        ],
    )
    assert responses(outputs) == [
        {"status": "cancelled", "error": "the reply was interrupted"}
    ]
    assert not tasks.jobs


@pytest.mark.asyncio
async def test_a_closed_session_cancels_its_tasks_and_sends_nothing_more():
    TOOL_GATES["c"], TOOL_STARTED["c"] = asyncio.Event(), asyncio.Event()
    tasks = tool_node(draw=GATED)
    outputs: list[TimedChunk] = []
    tasks.emit = outputs.append
    assert responses(
        [
            tool_response_chunk(
                [
                    await tasks.respond(
                        ToolCall(id="x", name="draw", arguments={"gate": "c"}), 1
                    )
                ],
                1,
            )
        ]
    ) == [{"status": "accepted", "task_id": "task_1"}]
    await TOOL_STARTED["c"].wait()
    before = len(outputs)
    await tasks.close()
    assert len(outputs) == before and tasks.jobs["task_1"].status == "cancelled"


@pytest.mark.asyncio
async def test_a_session_beyond_its_task_limit_is_refused_and_cancel_task_frees_it():
    TOOL_GATES["d"], TOOL_STARTED["d"] = asyncio.Event(), asyncio.Event()
    tasks = tool_node(draw=GATED)
    outputs = await drain(
        tasks,
        [
            call_chunk(1, "draw", gate="d"),
            call_chunk(2, "draw", gate="d"),
            call_chunk(3, "cancel_task", task_id="task_1"),
            call_chunk(4, "cancel_task", task_id="task_9"),
        ],
    )
    assert responses(outputs) == [
        {"status": "accepted", "task_id": "task_1"},
        {"status": "rejected", "error": "busy: too many tasks are running"},
        {"task_id": "task_1", "status": "cancelled"},
        {"error": "no task task_9"},
    ]
    result = next(o for o in outputs if o.format == TASK_RESULT).payload
    assert (result["status"], result["follow_up"]) == ("cancelled", "silent")


@pytest.mark.asyncio
async def test_the_end_of_the_input_cancels_running_tasks():
    TOOL_GATES["e"], TOOL_STARTED["e"] = asyncio.Event(), asyncio.Event()
    outputs = await drain(
        tool_node(draw=GATED),
        [call_chunk(1, "draw", gate="e"), lambda: TOOL_STARTED["e"].wait()],
    )
    assert task_events(outputs)[-2:] == [
        (TASK_UPDATE, "cancelled"),
        (TASK_RESULT, "cancelled"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool, error",
    [("failing_tool", "the service is down"), ("oversized_tool", "too large")],
)
async def test_a_failed_or_oversized_task_fails_with_a_reason(tool, error):
    tasks = tool_node(draw={"implementation": f"{FIXTURES}.{tool}"})
    outputs = await drain(
        tasks, [call_chunk(1, "draw"), lambda: until(lambda: not tasks.running_jobs())]
    )
    result = next(o for o in outputs if o.format == TASK_RESULT).payload
    assert result["status"] == "failed" and error in result["error"]
    assert (
        "artifact"
        not in next(
            o
            for o in outputs
            if o.format == TASK_UPDATE and o.payload["status"] == "failed"
        ).payload
    )


def tool_graph(**changes: object) -> dict:
    nodes = {
        "asr": {
            "kind": "openai_transcription",
            "base_url": "http://127.0.0.1:1/v1",
            "model": "a",
            "inputs": ["input"],
        },
        "dialog": {
            "kind": "dialog",
            "inputs": [
                "asr",
                "input:session",
                "llm:text_delta",
                "tts:session",
                {"kind": "control", "source": "tool", "event": "task_result"},
            ],
        },
        "llm": {
            "kind": "openai_chat",
            "base_url": "http://127.0.0.1:1/v1",
            "model": "m",
            "tools": [DRAW_TOOL],
            "inputs": [
                "dialog",
                {"kind": "control", "source": "tool", "event": "tool_response"},
                {"kind": "control", "source": "dialog", "event": "interrupt"},
            ],
        },
        "tool": {
            "kind": "async_tools",
            "cancel_tool": False,
            "tools": {"draw": GATED},
            "inputs": [
                "llm:tool_call",
                {"kind": "control", "source": "dialog", "event": "interrupt"},
            ],
        },
        "tts": {
            "kind": "openai_speech",
            "base_url": "http://127.0.0.1:1/v1",
            "model": "t",
            "inputs": [
                "llm",
                {"kind": "control", "source": "dialog", "event": "interrupt"},
            ],
        },
    }
    for name, node in changes.items():
        nodes[name] = node
    return {
        "nodes": [{"name": name, **node} for name, node in nodes.items()],
        "publish": {
            "dialog": ["session"],
            "tts": ["pcm_f32le", "session"],
            "tool": ["task_update"],
        },
    }


def test_a_tool_graph_with_one_tools_node_answering_on_a_control_edge_loads():
    ports = ComponentGraphConfig.model_validate(tool_graph()).ports(200)
    assert (ports.tasks, ports.is_proactive) == (("draw",), True)


@pytest.mark.parametrize(
    "change, error",
    [
        (
            {
                "tool": {
                    "kind": "async_tools",
                    "cancel_tool": False,
                    "tools": {"paint": GATED},
                    "inputs": [
                        "llm:tool_call",
                        {"kind": "control", "source": "dialog", "event": "interrupt"},
                    ],
                }
            },
            "differ from tools",
        ),
        (
            {
                "tool": {
                    "kind": "async_tools",
                    "tools": {"draw": GATED},
                    "inputs": [
                        "llm:tool_call",
                        {"kind": "control", "source": "dialog", "event": "interrupt"},
                    ],
                }
            },
            "differ from tools",
        ),
        (
            {
                "tool": {
                    "kind": "async_tools",
                    "cancel_tool": False,
                    "tools": {"draw": GATED},
                    "inputs": ["llm:tool_call"],
                }
            },
            "cancels tasks on interrupt",
        ),
        (
            {
                "dialog": {
                    "kind": "dialog",
                    "inputs": ["asr", "input:session", "llm:text_delta", "tts:session"],
                }
            },
            "speaks of task results",
        ),
        (
            {
                "llm": {
                    "kind": "openai_chat",
                    "base_url": "http://127.0.0.1:1/v1",
                    "model": "m",
                    "tools": [DRAW_TOOL],
                    "inputs": [
                        "dialog",
                        {"kind": "control", "source": "dialog", "event": "interrupt"},
                    ],
                }
            },
            "answer on a control edge",
        ),
    ],
    ids=[
        "other tool names",
        "cancel_task undeclared",
        "no interrupt edge",
        "no task_result edge",
        "no tool_response edge",
    ],
)
def test_a_tool_graph_missing_a_wire_fails_to_load(change, error):
    with pytest.raises(ValueError, match=error):
        ComponentGraphConfig.model_validate(tool_graph(**change))


def test_the_image_tool_profile_loads_with_its_tools_wired():
    config = ConfigManager.from_file(PROFILE).config
    ports = config.component_graph().ports(config.unit_ms)
    assert (ports.tasks, ports.is_proactive) == (("generate_image",), True)


@pytest.mark.asyncio
async def test_the_image_tool_polls_a_modelscope_task_and_returns_a_downscaled_jpeg(
    modelscope_service, monkeypatch
):
    monkeypatch.setenv("IMAGE_KEY", "secret")
    options = {
        "provider": "modelscope",
        "base_url": modelscope_service.base_url,
        "model": "m",
        "api_key_env": "IMAGE_KEY",
        "poll_interval_s": 0.01,
        "max_side_px": 512,
    }
    outcome = await generate_image({"prompt": "a cat"}, options)
    assert outcome.artifact is not None and outcome.artifact.media_type == "image/jpeg"
    assert Image.open(io.BytesIO(base64.b64decode(outcome.artifact.data))).size == (
        512,
        256,
    )
    assert modelscope_service.polls == 2
    assert modelscope_service.authorizations == ["Bearer secret", ""]


@pytest.mark.asyncio
async def test_a_failed_modelscope_task_raises_and_a_missing_key_fails_before_any_request(
    modelscope_service, monkeypatch
):
    modelscope_service.status = "FAILED"
    options = {
        "provider": "modelscope",
        "base_url": modelscope_service.base_url,
        "model": "m",
        "api_key_env": "IMAGE_KEY",
        "poll_interval_s": 0.01,
    }
    monkeypatch.delenv("IMAGE_KEY", raising=False)
    with pytest.raises(RuntimeError, match="IMAGE_KEY is not set"):
        await generate_image({"prompt": "a cat"}, options)
    monkeypatch.setenv("IMAGE_KEY", "secret")
    with pytest.raises(RuntimeError, match="failed the task"):
        await generate_image({"prompt": "a cat"}, options)
