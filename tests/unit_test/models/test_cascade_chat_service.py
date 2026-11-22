# SPDX-License-Identifier: Apache-2.0
"""The chat service's qwen3_coder parser: the calls of SGLang's detector, with a string argument released as it is written."""

import json
import random
import runpy
from types import SimpleNamespace

import pytest
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, Function, Tool
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector

from sglang_omni.models.streaming_cascade.chat_service import (
    StreamingQwen3CoderDetector,
    main,
)

TOOLS = [
    Tool(
        type="function",
        function=Function(
            name="tts_start",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["speak", "silent"]},
                    "text": {"type": "string"},
                    "speed": {"type": "number"},
                    "count": {"type": "integer"},
                    "loud": {"type": "boolean"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
            },
        ),
    ),
    Tool(
        type="function",
        function=Function(
            name="silent", parameters={"type": "object", "properties": {}}
        ),
    ),
]


def call(name: str, parameters: list[tuple[str, str]]) -> str:
    body = "".join(
        f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in parameters
    )
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


SPEECH = "Practice out loud. Record yourself. Slow down and pause."
COMPLETIONS = [
    call("tts_start", [("text", SPEECH)]),
    call("tts_start", [("action", "speak"), ("text", 'Two lines.\nA "quote", a \\.')]),
    call("tts_start", [("text", "a < b, </ c, <parameter and </param are no tags.")]),
    call("tts_start", [("text", "null")]),
    call("tts_start", [("text", "Null pointers are bad.")]),
    call("tts_start", [("text", "nul")]),
    call("tts_start", [("text", "ends with a newline\n")]),
    call("tts_start", [("text", "\nstarts with a newline")]),
    call("tts_start", [("text", "")]),
    call("tts_start", [("text", "你好，今天天气不错。😀 Tab\there.")]),
    call(
        "tts_start",
        [
            ("speed", "1.5"),
            ("text", "After a number."),
            ("count", "3"),
            ("loud", "true"),
            ("tags", '["a", "b"]'),
        ],
    ),
    call("tts_start", [("text", "first"), ("unknown", "not in the schema")]),
    call("get_weather", [("city", "a tool that was not offered")]),
    call("silent", []),
    "Some reasoning first.\n\n" + call("tts_start", [("text", "Then the call.")]),
    call("tts_start", [("text", "one")]) + "\n" + call("tts_start", [("text", "two")]),
    # Values the model ended without their closing tag.
    "<tool_call>\n<function=tts_start>\n<parameter=text>\nno closing tag\n<parameter=count>\n2\n</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=tts_start>\n<parameter=text>\nended by the function\n</function>\n</tool_call>",
    "No call at all, just text with a < sign.",
]


def parse(detector_cls: type, text: str, cuts: list[int]):
    """Feed the text in the pieces the cuts give; the reply text, each call as [name, arguments], and the number of argument fragments."""
    detector = detector_cls()
    reply = ""
    calls: dict[int, list] = {}
    fragment_count = 0
    for start, end in zip([0, *cuts], [*cuts, len(text)]):
        result = detector.parse_streaming_increment(text[start:end], TOOLS)
        reply += result.normal_text or ""
        for item in result.calls or []:
            entry = calls.setdefault(item.tool_index, [None, ""])
            if item.name:
                entry[0] = item.name
            else:
                pass
            if item.parameters:
                entry[1] += item.parameters
                fragment_count += 1
            else:
                pass
    return reply, calls, fragment_count


@pytest.mark.parametrize("completion", COMPLETIONS)
def test_any_pieces_give_the_calls_of_the_sglang_detector(completion: str):
    generator = random.Random(0)
    expected_reply, expected_calls, _ = parse(Qwen3CoderDetector, completion, [])
    positions = range(1, len(completion))
    chunkings = [[], list(positions)] + [
        sorted(
            generator.sample(positions, min(len(positions), generator.randint(1, 12)))
        )
        for _ in range(100)
    ]
    for cuts in chunkings:
        reply, calls, _ = parse(StreamingQwen3CoderDetector, completion, cuts)
        assert reply == expected_reply
        assert set(calls) == set(expected_calls)
        for index, (name, arguments) in calls.items():
            assert name == expected_calls[index][0]
            assert json.loads(arguments) == json.loads(expected_calls[index][1])


def test_a_string_argument_is_released_before_its_closing_tag():
    completion = COMPLETIONS[0]
    detector = StreamingQwen3CoderDetector()
    arguments = ""
    for size, char in enumerate(completion, start=1):
        for item in detector.parse_streaming_increment(char, TOOLS).calls or []:
            arguments += item.parameters or ""
        if "Practice out loud." in arguments:
            break
        else:
            pass
    assert size < completion.index("Record yourself.") + len("Record yourself.")
    # SGLang's detector sends the brace, the whole value and the closing brace.
    cuts = list(range(1, len(completion)))
    assert parse(Qwen3CoderDetector, completion, cuts)[2] == 3
    assert parse(StreamingQwen3CoderDetector, completion, cuts)[2] > 10


def test_a_call_the_token_budget_cut_keeps_the_text_written_so_far():
    cut = "<tool_call>\n<function=tts_start>\n<parameter=text>\nThis one never en"
    _, calls, _ = parse(StreamingQwen3CoderDetector, cut, list(range(1, len(cut))))
    assert calls == {0: ["tts_start", '{"text": "This one never en']}


def test_the_parser_name_stays_qwen3_coder():
    assert "qwen3_coder" in FunctionCallParser.ToolCallParserEnum
    assert issubclass(StreamingQwen3CoderDetector, Qwen3CoderDetector)


OPENING = "<tool_call>\n<function=tts_start>\n<parameter=text>\n"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("final_message", "is_continued", "completion", "reply", "calls"),
    [
        (
            OPENING,
            True,
            ["Hello", " world.", "\n</parameter>\n</function>\n</tool_call>"],
            "",
            [("tts_start", {"text": "Hello world."})],
        ),
        (call("tts_start", [("text", "Hi.")]), True, [" More."], " More.", []),
        (OPENING, False, ["Hi."], "Hi.", []),
    ],
)
async def test_a_completion_continues_the_call_its_assistant_message_left_open(
    monkeypatch: pytest.MonkeyPatch,
    final_message: str,
    is_continued: bool,
    completion: list[str],
    reply: str,
    calls: list[tuple[str, dict[str, str]]],
):
    """SGLang streamed a completion that continued an assistant message ending in a call's opening as text, the call's closing tags included, so the dialog would have spoken them."""
    # main patches SGLang and then serves; the patches are undone after the test, and nothing is served.
    monkeypatch.setitem(
        FunctionCallParser.ToolCallParserEnum, "qwen3_coder", Qwen3CoderDetector
    )
    monkeypatch.setattr(
        OpenAIServingChat,
        "_process_tool_call_stream",
        OpenAIServingChat._process_tool_call_stream,  # noqa: leading-underscore
    )
    monkeypatch.setattr(
        runpy, "run_module", lambda module_name, run_name, alter_sys: None
    )
    main()
    serving_chat = OpenAIServingChat.__new__(OpenAIServingChat)
    serving_chat.tool_call_parser = "qwen3_coder"
    serving_chat.tokenizer_manager = SimpleNamespace(tokenizer=None)
    request = ChatCompletionRequest(
        model="m",
        messages=[
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": final_message},
        ],
        tools=TOOLS,
        tool_choice="auto",
        stream=True,
        continue_final_message=is_continued,
    )
    parser_dict = {}
    streamed_reply = ""
    names: dict[int, str] = {}
    arguments: dict[int, str] = {}
    for position, piece in enumerate(completion, start=1):
        events = serving_chat._process_tool_call_stream(  # noqa: leading-underscore
            0,
            piece,
            parser_dict,
            {"meta_info": {"id": "chatcmpl-test"}},
            request,
            {},
            created=0,
            flush=position == len(completion),
        )
        async for event in events:
            delta = json.loads(event.removeprefix("data: "))["choices"][0]["delta"]
            streamed_reply += delta.get("content") or ""
            for tool_call in delta.get("tool_calls") or []:
                index = tool_call["index"]
                names[index] = tool_call["function"]["name"] or names.get(index, "")
                arguments[index] = arguments.get(index, "") + (
                    tool_call["function"]["arguments"] or ""
                )
    assert streamed_reply == reply
    assert [
        (names[index], json.loads(arguments[index])) for index in sorted(names)
    ] == calls
