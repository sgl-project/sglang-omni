# SPDX-License-Identifier: Apache-2.0
"""The model speaks through a tool: the dialog, the synthesizer and the generator around tts_start."""

import asyncio
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
import yaml

from sglang_omni.models.streaming_cascade.base import (
    ChatMessage,
    control_chunk,
    messages_of,
    text_chunk,
    transcript_chunk,
)
from sglang_omni.models.streaming_cascade.config import StreamingCascadePipelineConfig
from sglang_omni.models.streaming_cascade.dialog import SpokenTextReader
from sglang_omni.models.streaming_cascade.generators import (
    ChatDelta,
    FunctionDelta,
    ToolCallDelta,
)
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import (
    INTERRUPT,
    MESSAGES,
    PCM_F32,
    REPLY_DELIVERED,
    REPLY_PROGRESS,
    SPEECH_ERROR,
    SPEECH_STARTED,
    TEXT_DELTA,
    TOOL_CALL,
    TRUNCATE,
    TURN_DONE,
    TURN_EMPTY,
    TURN_INTERRUPTED,
    TURN_QUEUED,
    TURN_STARTED,
    control_type,
)
from sglang_omni.proto.tools import (
    ToolCall,
    is_partial_call,
    tool_call_chunk,
    tool_calls_of,
)
from tests.unit_test.fixtures.cascade_fakes import build, drain

SPEECH_PROFILE = Path(
    "examples/streaming_cascade/profiles/qwen3-asr-qwen3.8-27b-vivian-tts-tool/launch.yaml"
)
SPEECH = "One two.\nThree four. Five six."
# Sentence ends at characters 8 and 20, with their audio ending at 1000 and 2000 ms.
SPEECH_PROGRESS_POINTS = [(8, 1000.0), (20, 2000.0)]
COMPLETED = {"status": "completed"}


def speech_dialog(**options: object):
    return build({"kind": "dialog", "speech_tool": "tts_start", **options})


def speech_call(text: object = SPEECH, call_id: str = "call_1", turn: int = 1):
    return tool_call_chunk(
        [ToolCall(id=call_id, name="tts_start", arguments={"text": text})], turn
    )


def spoken(outputs: list[TimedChunk]) -> list[str]:
    return [
        tool_calls_of(output)[0].arguments["text"]
        for output in outputs
        if output.format == TOOL_CALL
    ]


def turn_events(outputs: list[TimedChunk]) -> list[tuple[str, int | None]]:
    return [
        (control_type(output), output.turn)
        for output in outputs
        if control_type(output) is not None
    ]


def prompts(outputs: list[TimedChunk]) -> list[list[ChatMessage]]:
    return [messages_of(output) for output in outputs if output.format == MESSAGES]


def answers(messages: list[ChatMessage]) -> dict[str, dict[str, str]]:
    """Each call's answer as the model reads it, by call id."""
    return {
        message.tool_call_id: json.loads(message.content)
        for message in messages
        if message.role == "tool"
    }


def progress_of(dialog) -> list[object]:
    return [
        lambda text_end=text_end, audio_ms=audio_ms: dialog.control(
            control_chunk(REPLY_PROGRESS, turn=1, text_end=text_end, audio_ms=audio_ms)
        )
        for text_end, audio_ms in SPEECH_PROGRESS_POINTS
    ]


@pytest.mark.asyncio
async def test_a_call_is_answered_when_its_speech_ends_and_the_model_is_not_prompted_again():
    dialog = speech_dialog()
    outputs = await drain(dialog, [transcript_chunk("question", True), speech_call()])
    assert spoken(outputs) == [SPEECH]
    assert len(prompts(outputs)) == 1

    # The model spoke for the utterance, so the end of its speech prompts nothing.
    assert await drain(dialog, [control_chunk(TURN_DONE, turn=1)]) == []
    assert dialog.reply is None
    assert answers(dialog.messages) == {"call_1": COMPLETED}

    # It reads the answer in the prompt for the next utterance.
    outputs = await drain(dialog, [transcript_chunk("next", True)])
    assert turn_events(outputs) == [(TURN_QUEUED, 2), (TURN_STARTED, 2)]
    [prompt] = prompts(outputs)
    assert [message.role for message in prompt] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert json.loads(prompt[2].tool_calls[0].function.arguments) == {"text": SPEECH}
    assert answers(prompt) == {"call_1": COMPLETED}


@pytest.mark.asyncio
async def test_every_prompt_requires_a_call():
    dialog = speech_dialog()
    outputs = await drain(
        dialog, [transcript_chunk("question", True), speech_call(text="")]
    )
    assert [
        output.payload.get("tool_choice")
        for output in outputs
        if output.format == MESSAGES
    ] == ["required", "required"]


@pytest.mark.asyncio
async def test_a_speech_that_failed_may_be_said_again():
    dialog = speech_dialog()
    outputs = await drain(
        dialog,
        [
            transcript_chunk("question", True),
            speech_call(),
            control_chunk(TURN_DONE, turn=1, **{SPEECH_ERROR: "503"}),
        ],
    )
    assert len(prompts(outputs)) == 2
    outputs = await drain(dialog, [speech_call(SPEECH, "call_2", 2)])
    assert spoken(outputs) == [SPEECH]
    assert await drain(dialog, [control_chunk(TURN_DONE, turn=2)]) == []
    assert answers(dialog.messages)["call_2"] == COMPLETED


@pytest.mark.asyncio
async def test_a_speech_after_a_refused_call_is_a_turn_announced_before_the_synthesizer_gets_it():
    dialog = speech_dialog()
    await drain(dialog, [transcript_chunk("question", True), speech_call(text="")])
    outputs = await drain(dialog, [speech_call("More.", "call_2", 2)])
    assert [control_type(output) or output.format for output in outputs] == [
        TURN_QUEUED,
        TURN_STARTED,
        TOOL_CALL,
    ]
    assert {output.turn for output in outputs} == {2}


@pytest.mark.asyncio
async def test_what_was_heard_keeps_rewriting_the_answer_later_prompts_read():
    dialog = speech_dialog()
    outputs = await drain(
        dialog,
        [
            transcript_chunk("question", True),
            speech_call(),
            *progress_of(dialog),
            control_chunk(SPEECH_STARTED),
        ],
    )
    assert turn_events(outputs)[-2:] == [(INTERRUPT, 1), (TURN_INTERRUPTED, 1)]
    assert answers(dialog.messages)["call_1"]["heard_prefix"] == ""

    await drain(dialog, [control_chunk(REPLY_DELIVERED, turn=1, audio_ms=2300)])
    assert answers(dialog.messages)["call_1"] == {
        "status": "interrupted",
        "heard_prefix": "One two.\nThree four.",
    }

    outputs = await drain(dialog, [transcript_chunk("next", True)])
    assert answers(prompts(outputs)[0])["call_1"]["heard_prefix"] == (
        "One two.\nThree four."
    )
    # A report that arrives after the next reply began still reaches the prompts after it.
    await drain(dialog, [control_chunk(TRUNCATE, turn=1, t_ms=1200)])
    assert answers(dialog.messages)["call_1"]["heard_prefix"] == "One two."
    assert json.loads(dialog.messages[2].tool_calls[0].function.arguments) == {
        "text": SPEECH
    }


@pytest.mark.asyncio
async def test_a_truncate_after_the_speech_ended_marks_it_interrupted():
    dialog = speech_dialog()
    await drain(
        dialog,
        [
            transcript_chunk("question", True),
            speech_call(),
            *progress_of(dialog),
            control_chunk(TURN_DONE, turn=1),
            control_chunk(TRUNCATE, turn=1, t_ms=1200),
        ],
    )
    assert answers(dialog.messages)["call_1"] == {
        "status": "interrupted",
        "heard_prefix": "One two.",
    }


@pytest.mark.asyncio
async def test_only_the_first_valid_call_is_spoken_and_every_call_gets_one_answer():
    dialog = speech_dialog()
    calls = [
        ToolCall(id="unknown", name="get_weather", arguments={"city": "Paris"}),
        ToolCall(id="empty", name="tts_start", arguments={"text": " "}),
        ToolCall(id="cut", name="tts_start", arguments='{"text": "One'),
        ToolCall(id="first", name="tts_start", arguments={"text": "First."}),
        ToolCall(id="second", name="tts_start", arguments={"text": "Second."}),
    ]
    outputs = await drain(
        dialog, [transcript_chunk("question", True), tool_call_chunk(calls, 1)]
    )
    assert spoken(outputs) == ["First."]
    outcomes = answers(dialog.messages)
    assert list(outcomes) == ["unknown", "empty", "cut", "first", "second"]
    assert {call: outcomes[call]["status"] for call in ("unknown", "empty", "cut")} == {
        "unknown": "failed",
        "empty": "failed",
        "cut": "failed",
    }
    assert outcomes["second"] == {"status": "failed", "error": "BUSY"}
    # The history can only render arguments that are a JSON object.
    assert dialog.messages[2].tool_calls[2].function.arguments == "{}"

    await drain(dialog, [control_chunk(TURN_DONE, turn=1)])
    assert answers(dialog.messages)["first"] == COMPLETED


@pytest.mark.asyncio
async def test_a_refused_call_ends_the_announced_turn_and_the_model_may_try_again():
    dialog = speech_dialog()
    outputs = await drain(
        dialog, [transcript_chunk("question", True), speech_call(text="")]
    )
    assert spoken(outputs) == []
    assert (TURN_EMPTY, 1) in turn_events(outputs)
    [_, retry] = prompts(outputs)
    assert answers(retry)["call_1"]["status"] == "failed"
    assert (TURN_STARTED, 2) not in turn_events(outputs)


@pytest.mark.asyncio
async def test_text_the_model_wrote_without_a_call_is_spoken_and_ends_its_replies():
    dialog = speech_dialog()
    outputs = await drain(
        dialog,
        [
            transcript_chunk("question", True),
            text_chunk("Plain answer.", 1),
            control_chunk(TURN_DONE, turn=1),
        ],
    )
    assert spoken(outputs) == ["Plain answer."]
    assert (TURN_EMPTY, 1) not in turn_events(outputs)
    assert await drain(dialog, [control_chunk(TURN_DONE, turn=1)]) == []
    assert dialog.reply is None
    assert (dialog.messages[-1].role, dialog.messages[-1].content) == (
        "assistant",
        "Plain answer.",
    )


@pytest.mark.asyncio
async def test_text_after_a_refused_call_is_spoken_since_nothing_was_said_yet():
    dialog = speech_dialog()
    outputs = await drain(
        dialog,
        [
            transcript_chunk("question", True),
            speech_call(text=""),
            text_chunk("Plain answer.", 2),
            control_chunk(TURN_DONE, turn=2),
        ],
    )
    assert spoken(outputs) == ["Plain answer."]
    assert (TURN_STARTED, 2) in turn_events(outputs)


@pytest.mark.asyncio
async def test_an_interrupted_text_without_a_call_is_cut_like_a_plain_reply():
    dialog = speech_dialog()
    await drain(
        dialog,
        [
            transcript_chunk("question", True),
            text_chunk(SPEECH, 1),
            control_chunk(TURN_DONE, turn=1),
            *progress_of(dialog),
            control_chunk(SPEECH_STARTED),
            control_chunk(REPLY_DELIVERED, turn=1, audio_ms=1100),
        ],
    )
    assert dialog.messages[-1].content == "One two."
    assert dialog.messages[-1].is_cut


@pytest.mark.asyncio
async def test_text_beside_a_call_is_recorded_but_not_spoken():
    dialog = speech_dialog()
    outputs = await drain(
        dialog,
        [
            transcript_chunk("question", True),
            text_chunk("Let me say this.", 1),
            speech_call("Said."),
        ],
    )
    assert spoken(outputs) == ["Said."]
    assert dialog.messages[2].content == "Let me say this."


@pytest.mark.asyncio
async def test_speech_over_an_unannounced_prompt_stops_the_model_and_tells_the_client_nothing():
    dialog = speech_dialog()
    # A refused call prompts the model again in a turn the client has not heard of.
    await drain(dialog, [transcript_chunk("question", True), speech_call(text="")])
    outputs = await drain(dialog, [control_chunk(SPEECH_STARTED)])
    assert turn_events(outputs) == [(INTERRUPT, 2)]

    outputs = await drain(dialog, [transcript_chunk("next", True)])
    assert [message.role for message in prompts(outputs)[0]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
    ]


@pytest.mark.asyncio
async def test_speech_over_the_first_prompt_joins_the_utterances_as_a_plain_reply_does():
    dialog = speech_dialog()
    await drain(
        dialog,
        [
            transcript_chunk("question", True),
            text_chunk("Some text.", 1),
            control_chunk(SPEECH_STARTED),
            control_chunk(REPLY_DELIVERED, turn=1, audio_ms=0),
        ],
    )
    outputs = await drain(dialog, [transcript_chunk("next", True)])
    assert [(message.role, message.content) for message in prompts(outputs)[0][1:]] == [
        ("user", "question next")
    ]


@pytest.mark.asyncio
async def test_a_waiting_utterance_is_answered_when_the_speech_ends():
    dialog = speech_dialog()
    await drain(
        dialog,
        [
            transcript_chunk("question", True),
            speech_call(),
            transcript_chunk("next", True),
        ],
    )
    outputs = await drain(dialog, [control_chunk(TURN_DONE, turn=1)])
    [prompt] = prompts(outputs)
    assert prompt[-1].content == "next"
    assert answers(prompt)["call_1"] == COMPLETED
    assert turn_events(outputs) == [(TURN_STARTED, 2)]


@pytest.mark.asyncio
async def test_a_model_whose_calls_keep_being_refused_stops_being_prompted_at_the_bound():
    dialog = speech_dialog(max_inferences_per_utterance=2)
    outputs = await drain(
        dialog,
        [
            transcript_chunk("question", True),
            speech_call(text=""),
            speech_call("", "call_2", 2),
        ],
    )
    assert len(prompts(outputs)) == 2
    assert dialog.reply is None

    outputs = await drain(
        dialog, [transcript_chunk("next", True), speech_call("Yes.", "call_3", 3)]
    )
    assert spoken(outputs) == ["Yes."]
    assert await drain(dialog, [control_chunk(TURN_DONE, turn=3)]) == []


@pytest.mark.asyncio
async def test_a_failed_speech_is_reported_to_the_model_and_no_truncate_rewrites_it():
    dialog = speech_dialog()
    outputs = await drain(
        dialog,
        [
            transcript_chunk("question", True),
            speech_call(),
            control_chunk(TURN_DONE, turn=1, **{SPEECH_ERROR: "503"}),
        ],
    )
    assert answers(prompts(outputs)[-1])["call_1"] == {
        "status": "failed",
        "error": "503",
    }
    await drain(dialog, [control_chunk(TRUNCATE, turn=1, t_ms=0)])
    assert answers(dialog.messages)["call_1"]["status"] == "failed"


@pytest.mark.asyncio
async def test_a_speech_that_failed_after_some_audio_tells_the_model_what_was_heard():
    dialog = speech_dialog()
    outputs = await drain(
        dialog,
        [
            transcript_chunk("question", True),
            speech_call(),
            *progress_of(dialog),
            control_chunk(TURN_DONE, turn=1, **{SPEECH_ERROR: "503"}),
        ],
    )
    assert answers(dialog.messages)["call_1"] == {
        "status": "failed",
        "error": "503",
        "heard_prefix": "One two.\nThree four.",
    }
    # The prompt that lets the model speak again says what heard_prefix means there.
    [_, retry] = prompts(outputs)
    assert "broke off" in retry[0].content
    assert "cut you off" not in retry[0].content


@pytest.mark.asyncio
async def test_a_speech_keeps_its_place_when_a_late_truncate_removes_the_reply_before_it():
    dialog = speech_dialog()
    await drain(
        dialog,
        [
            transcript_chunk("question", True),
            # A reply the model wrote without a call: the generator ends it, then the synthesizer.
            text_chunk("Plain answer.", 1),
            control_chunk(TURN_DONE, turn=1),
            control_chunk(TURN_DONE, turn=1),
            transcript_chunk("next", True),
            speech_call("Said.", "call_1", 2),
            # The client reports late that it heard nothing of the first reply, while the second is spoken.
            control_chunk(TRUNCATE, turn=1, t_ms=0),
            control_chunk(TURN_DONE, turn=2),
        ],
    )
    assert [(message.role, message.content) for message in dialog.messages[1:2]] == [
        ("user", "question next")
    ]
    assert [message.role for message in dialog.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert answers(dialog.messages) == {"call_1": COMPLETED}


@pytest.mark.asyncio
async def test_a_silent_first_prompt_ends_the_turn_and_the_next_utterance_joins_it():
    dialog = speech_dialog()
    outputs = await drain(
        dialog,
        [transcript_chunk("question", True), control_chunk(TURN_DONE, turn=1)],
    )
    assert (TURN_EMPTY, 1) in turn_events(outputs)
    outputs = await drain(dialog, [transcript_chunk("next", True)])
    assert prompts(outputs)[0][-1].content == "question next"


@pytest.mark.asyncio
async def test_the_arguments_of_past_speeches_count_toward_the_history_bound():
    dialog = speech_dialog(max_history_chars=2000, max_text_chars=100)
    long_speech = "word " * 19
    inputs: list[object] = []
    for index in range(1, 21):
        inputs += [
            transcript_chunk("question", True),
            speech_call(long_speech, f"call_{index}", index),
            control_chunk(TURN_DONE, turn=index),
        ]
    with pytest.raises(ValueError, match="max_history_chars"):
        await drain(dialog, inputs)


@pytest.mark.asyncio
async def test_only_a_cut_speech_adds_a_note_to_the_prompt_the_client_set():
    dialog = speech_dialog(system_prompt="Be terse.")
    outputs = await drain(
        dialog,
        [
            transcript_chunk("question", True),
            speech_call(),
            control_chunk(TURN_DONE, turn=1),
            transcript_chunk("next", True),
            speech_call(SPEECH + " More.", "call_2", 2),
            *[
                lambda text_end=text_end, audio_ms=audio_ms: dialog.control(
                    control_chunk(
                        REPLY_PROGRESS, turn=2, text_end=text_end, audio_ms=audio_ms
                    )
                )
                for text_end, audio_ms in SPEECH_PROGRESS_POINTS
            ],
            control_chunk(SPEECH_STARTED),
            control_chunk(REPLY_DELIVERED, turn=2, audio_ms=1100),
            transcript_chunk("last", True),
        ],
    )
    first, _, last = prompts(outputs)
    assert first[0].content == "Be terse."
    assert last[0].content.startswith("Be terse. ")
    assert "heard_prefix" in last[0].content
    assert answers(last)["call_2"]["heard_prefix"] == "One two."


@pytest.mark.asyncio
async def test_new_instructions_stand_beside_the_next_utterance_in_its_prompt_only():
    dialog = speech_dialog()
    await drain(
        dialog,
        [
            transcript_chunk("question", True),
            speech_call(),
            control_chunk(TURN_DONE, turn=1),
        ],
    )
    replacement = speech_dialog(system_prompt="Be terse.")
    replacement.adopt(dialog)
    outputs = await drain(
        replacement,
        [
            transcript_chunk("next", True),
            speech_call("Yes.", "call_2", 2),
            control_chunk(TURN_DONE, turn=2),
            transcript_chunk("last", True),
        ],
    )
    next_prompt, last_prompt = prompts(outputs)
    assert next_prompt[-1].content == "next\n\n[Be terse.]"
    assert last_prompt[-1].content == "last"
    assert [message.content for message in replacement.messages].count("next") == 1


@pytest.mark.asyncio
async def test_a_replacement_with_the_same_instructions_adds_nothing():
    dialog = speech_dialog()
    await drain(
        dialog,
        [
            transcript_chunk("question", True),
            speech_call(),
            control_chunk(TURN_DONE, turn=1),
        ],
    )
    replacement = speech_dialog()
    replacement.adopt(dialog)
    outputs = await drain(replacement, [transcript_chunk("next", True)])
    assert prompts(outputs)[0][-1].content == "next"


@pytest.mark.asyncio
async def test_a_replacement_dialog_keeps_the_speech_it_is_waiting_for():
    dialog = speech_dialog()
    await drain(dialog, [transcript_chunk("question", True), speech_call()])
    replacement = speech_dialog()
    replacement.adopt(dialog)
    await drain(replacement, [control_chunk(TURN_DONE, turn=1)])
    assert answers(replacement.messages)["call_1"] == COMPLETED


def speech_synthesizer():
    return build(
        {
            "kind": "openai_speech",
            "base_url": "http://127.0.0.1:1/v1",
            "model": "test-model",
        }
    )


@pytest.mark.asyncio
async def test_the_synthesizer_speaks_a_call_and_emits_its_text_before_the_audio():
    synthesizer = speech_synthesizer()
    sentences: list[str] = []

    async def speak(sentence: str, language: str | None):
        sentences.append(sentence)
        yield np.zeros(240, dtype="<f4")

    synthesizer.speak = speak
    call = ToolCall(id="call_1", name="tts_start", arguments={"text": "One. Two."})
    outputs = await drain(synthesizer, [tool_call_chunk([call], 1, "English")])
    assert sentences == ["One.", "Two."]
    formats = [control_type(output) or output.format for output in outputs]
    assert formats.index(TEXT_DELTA) < formats.index(PCM_F32)
    assert formats[-1] == TURN_DONE
    [text] = [output for output in outputs if output.format == TEXT_DELTA]
    assert (text.payload["text"], text.payload["language"], text.turn) == (
        "One. Two.",
        "English",
        1,
    )
    assert SPEECH_ERROR not in outputs[-1].payload


@pytest.mark.asyncio
async def test_a_failed_speech_service_ends_the_speech_with_the_error_instead_of_the_session():
    synthesizer = speech_synthesizer()

    async def speak(sentence: str, language: str | None):
        raise httpx.ConnectError("refused")
        yield

    synthesizer.speak = speak
    outputs = await drain(synthesizer, [speech_call("One.")])
    assert control_type(outputs[-1]) == TURN_DONE
    assert "refused" in outputs[-1].payload[SPEECH_ERROR]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["...", "👍"])
async def test_a_speech_with_nothing_to_speak_is_reported_instead_of_completed(text):
    synthesizer = speech_synthesizer()
    sentences: list[str] = []

    async def speak(sentence: str, language: str | None):
        sentences.append(sentence)
        yield np.zeros(240, dtype="<f4")

    synthesizer.speak = speak
    outputs = await drain(synthesizer, [speech_call(text)])
    assert sentences == []
    assert control_type(outputs[-1]) == TURN_DONE
    assert "no audio was produced" in outputs[-1].payload[SPEECH_ERROR]


def tool_generator():
    return build(
        {
            "kind": "openai_chat",
            "base_url": "http://127.0.0.1:1/v1",
            "model": "test-model",
            "tools": [{"name": "tts_start", "description": "Say text aloud."}],
        }
    )


@pytest.mark.asyncio
async def test_the_generator_joins_call_fragments_into_one_chunk_and_leaves_the_turn_open():
    generator = tool_generator()

    async def complete(messages: list[ChatMessage], tool_choice: str | None):
        yield ChatDelta(content="Note. ")
        yield ChatDelta(
            tool_calls=[
                ToolCallDelta(
                    index=0,
                    id="call_a",
                    function=FunctionDelta(name="tts_start", arguments='{"te'),
                )
            ]
        )
        yield ChatDelta(
            tool_calls=[
                ToolCallDelta(index=0, function=FunctionDelta(arguments='xt": "Hi."}'))
            ]
        )

    generator.complete = complete
    messages = [ChatMessage(role="user", content="question")]
    body = generator.request_body(messages)
    assert body["tools"] == [
        {
            "type": "function",
            "function": {"name": "tts_start", "description": "Say text aloud."},
        }
    ]
    assert generator.request_body(messages, "required")["tool_choice"] == "required"
    assert "tool_choice" not in body
    outputs = []
    await generator.reply(messages, 1, None, outputs.append)
    await generator.close()
    assert [output.format for output in outputs] == [
        TEXT_DELTA,
        TOOL_CALL,
        TOOL_CALL,
        TOOL_CALL,
    ]
    # The first call's arguments go out as they arrive, before the complete call.
    assert [is_partial_call(output) for output in outputs[1:]] == [True, True, False]
    assert [tool_calls_of(output)[0].arguments for output in outputs[1:3]] == [
        '{"te',
        'xt": "Hi."}',
    ]
    assert tool_calls_of(outputs[3]) == [
        ToolCall(id="call_a", name="tts_start", arguments={"text": "Hi."})
    ]


@pytest.mark.asyncio
async def test_a_call_the_token_budget_cut_keeps_its_raw_text_for_the_dialog_to_refuse():
    generator = tool_generator()

    async def complete(messages: list[ChatMessage], tool_choice: str | None):
        yield ChatDelta(
            tool_calls=[
                ToolCallDelta(
                    index=0,
                    function=FunctionDelta(name="tts_start", arguments='{"text": "Hi'),
                )
            ]
        )

    generator.complete = complete
    outputs = []
    await generator.reply([], 7, None, outputs.append)
    await generator.close()
    assert tool_calls_of(outputs[0]) == [
        ToolCall(id="call_7_0", name="tts_start", arguments='{"text": "Hi')
    ]


def speech_profile(**changes: dict[str, object]) -> StreamingCascadePipelineConfig:
    """The speech tool profile's declaration with the options of its nodes replaced by `changes`."""
    data = yaml.safe_load(SPEECH_PROFILE.read_text())
    return StreamingCascadePipelineConfig(
        model_path="m",
        session_nodes={
            name: {**node, **changes.get(name, {})}
            for name, node in data["session_nodes"].items()
        },
        session_publish=data["session_publish"],
    )


def test_the_profile_speaks_only_through_the_tool():
    settings = speech_profile().component_graph()
    assert settings.node("dialog").component.speech_tool == "tts_start"
    assert [tool.name for tool in settings.node("llm").component.tools] == ["tts_start"]
    assert settings.node("tts").edges()["dialog"] == {TOOL_CALL}
    assert TEXT_DELTA in settings.node("aligner").edges()["tts"]
    assert "llm" not in settings.publish


INTERRUPT_EDGE = {"kind": "control", "source": "dialog", "event": INTERRUPT}


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        (
            {
                "tts": {
                    "inputs": [
                        "llm",
                        "dialog:tool_call",
                        "input:session",
                        INTERRUPT_EDGE,
                    ]
                }
            },
            "spoken twice",
        ),
        (
            {"tts": {"inputs": ["llm", "input:session", INTERRUPT_EDGE]}},
            "no synthesizer",
        ),
        ({"llm": {"tools": []}}, "does not offer"),
        ({"llm": {"tools": [{"name": "other"}]}}, "does not speak through"),
        ({"dialog": {"speech_tool": None}}, "does not speak through"),
    ],
)
def test_a_graph_that_would_speak_twice_or_not_at_all_fails_at_load(changes, error):
    with pytest.raises(ValueError, match=error):
        speech_profile(**changes)


def speech_pieces(pieces: list[str], call_id: str = "call_1", turn: int = 1):
    """The arguments of a call as the generator sends them while the model writes."""
    return [
        tool_call_chunk(
            [ToolCall(id=call_id, name="tts_start", arguments=piece)],
            turn,
            is_partial=True,
        )
        for piece in pieces
    ]


def streamed(outputs: list[TimedChunk]) -> list[str]:
    return [
        tool_calls_of(output)[0].arguments["text"]
        for output in outputs
        if output.format == TOOL_CALL and is_partial_call(output)
    ]


def completed_calls(outputs: list[TimedChunk]) -> list[str]:
    return [
        tool_calls_of(output)[0].arguments["text"]
        for output in outputs
        if output.format == TOOL_CALL and not is_partial_call(output)
    ]


@pytest.mark.parametrize(
    ("arguments", "text"),
    [
        ('{"text": "One two. Three."}', "One two. Three."),
        ('{"action": "speak", "speed": 1.5, "text": "Hi."}', "Hi."),
        (
            json.dumps({"text": 'A "quote", a \\ and a\nline.'}),
            'A "quote", a \\ and a\nline.',
        ),
        (json.dumps({"text": "你好 😀"}), "你好 😀"),
        (json.dumps({"text": "你好 😀"}, ensure_ascii=False), "你好 😀"),
        (
            json.dumps({"text": "a \\ud83d is not an escape"}),
            "a \\ud83d is not an escape",
        ),
        ('{"note": "the \\"text\\": \\"trap", "text": "Real."}', "Real."),
        ('{"text": ""}', ""),
        ('{"other": "no text here"}', ""),
    ],
)
def test_the_reader_finds_the_text_whatever_the_pieces(arguments: str, text: str):
    for size in (1, 2, 3, 7, len(arguments)):
        reader = SpokenTextReader()
        seen = ""
        for start in range(0, len(arguments), size):
            reader.feed(arguments[start : start + size])
            # What was read never has to be taken back.
            assert reader.text.startswith(seen)
            seen = reader.text
        assert reader.text == text


@pytest.mark.asyncio
async def test_a_speech_starts_with_its_first_words_and_ends_with_the_call():
    dialog = speech_dialog()
    pieces = ['{"te', 'xt": "', "One two.", "\\nThree", " four. Five six."]
    outputs = await drain(
        dialog, [transcript_chunk("question", True), *speech_pieces(pieces)]
    )
    assert streamed(outputs) == ["One two.", "\nThree", " four. Five six."]
    assert completed_calls(outputs) == []
    # The history already holds the call, answered as cut with nothing heard, as a whole call's is.
    assert json.loads(dialog.messages[2].tool_calls[0].function.arguments) == {
        "text": SPEECH
    }
    assert answers(dialog.messages) == {
        "call_1": {"status": "interrupted", "heard_prefix": ""}
    }

    outputs = await drain(dialog, [*speech_pieces(['"}']), speech_call()])
    assert streamed(outputs) == []
    assert completed_calls(outputs) == [SPEECH]

    assert await drain(dialog, [control_chunk(TURN_DONE, turn=1)]) == []
    assert json.loads(dialog.messages[2].tool_calls[0].function.arguments) == {
        "text": SPEECH
    }
    assert answers(dialog.messages) == {"call_1": COMPLETED}


@pytest.mark.asyncio
async def test_a_speech_cut_while_the_model_writes_keeps_what_was_heard():
    dialog = speech_dialog()
    outputs = await drain(
        dialog,
        [
            transcript_chunk("question", True),
            *speech_pieces(['{"text": "One two.', "\\nThree four."]),
            *progress_of(dialog),
            control_chunk(SPEECH_STARTED),
        ],
    )
    assert turn_events(outputs)[-2:] == [(INTERRUPT, 1), (TURN_INTERRUPTED, 1)]
    await drain(dialog, [control_chunk(REPLY_DELIVERED, turn=1, audio_ms=1200)])
    assert answers(dialog.messages)["call_1"] == {
        "status": "interrupted",
        "heard_prefix": "One two.",
    }
    # The call keeps the text the model had written when it was cut; the rest never arrives.
    outputs = await drain(dialog, [*speech_pieces([" Five six."]), speech_call()])
    assert outputs == []
    assert json.loads(dialog.messages[2].tool_calls[0].function.arguments) == {
        "text": "One two.\nThree four."
    }


@pytest.mark.asyncio
async def test_a_streamed_call_the_token_budget_cut_keeps_the_text_that_was_spoken():
    dialog = speech_dialog()
    cut = '{"text": "One two. Three'
    outputs = await drain(
        dialog,
        [
            transcript_chunk("question", True),
            *speech_pieces([cut]),
            tool_call_chunk(
                [ToolCall(id="call_1", name="tts_start", arguments=cut)], 1
            ),
        ],
    )
    assert streamed(outputs) == ["One two. Three"]
    assert completed_calls(outputs) == ["One two. Three"]
    assert json.loads(dialog.messages[2].tool_calls[0].function.arguments) == {
        "text": "One two. Three"
    }


@pytest.mark.asyncio
async def test_a_second_call_beside_a_streamed_speech_is_refused():
    dialog = speech_dialog()
    calls = [
        ToolCall(id="call_1", name="tts_start", arguments={"text": "One."}),
        ToolCall(id="call_2", name="tts_start", arguments={"text": "Two."}),
    ]
    outputs = await drain(
        dialog,
        [
            transcript_chunk("question", True),
            *speech_pieces(['{"text": "One."}']),
            tool_call_chunk(calls, 1),
        ],
    )
    assert completed_calls(outputs) == ["One."]
    assert [message.role for message in dialog.messages[2:]] == [
        "assistant",
        "tool",
        "tool",
    ]
    assert answers(dialog.messages)["call_2"] == {"status": "failed", "error": "BUSY"}


@pytest.mark.asyncio
async def test_a_speech_after_a_refused_call_starts_with_its_first_words_too():
    dialog = speech_dialog()
    await drain(dialog, [transcript_chunk("question", True), speech_call(text="")])
    outputs = await drain(dialog, speech_pieces(['{"text": "Hi.'], "call_2", 2))
    assert turn_events(outputs) == [(TURN_QUEUED, 2), (TURN_STARTED, 2)]
    assert streamed(outputs) == ["Hi."]


@pytest.mark.asyncio
async def test_a_blank_streamed_text_is_refused_as_a_whole_call_is():
    dialog = speech_dialog()
    outputs = await drain(
        dialog,
        [
            transcript_chunk("question", True),
            *speech_pieces(['{"text": " "}']),
            speech_call(" "),
        ],
    )
    assert spoken(outputs) == []
    assert answers(dialog.messages)["call_1"]["status"] == "failed"


def spoken_pieces(texts: list[str], turn: int = 1) -> list[TimedChunk]:
    """A speech as the dialog streams it: its text in pieces, then the complete call."""
    pieces = [
        tool_call_chunk(
            [ToolCall(id="call_1", name="tts_start", arguments={"text": text})],
            turn,
            "English",
            is_partial=True,
        )
        for text in texts
    ]
    whole = ToolCall(id="call_1", name="tts_start", arguments={"text": "".join(texts)})
    return [*pieces, tool_call_chunk([whole], turn, "English")]


@pytest.mark.asyncio
async def test_the_synthesizer_speaks_a_sentence_before_the_call_is_complete():
    synthesizer = speech_synthesizer()
    events: list[str] = []

    async def speak(sentence: str, language: str | None):
        events.append(sentence)
        yield np.zeros(240, dtype="<f4")

    async def note_arrival():
        # The first sentence must already be spoken when the rest of the call arrives.
        for _ in range(20):
            await asyncio.sleep(0)
        events.append("rest arrives")

    synthesizer.speak = speak
    first, second, whole = spoken_pieces(["One. Tw", "o."])
    outputs = await drain(synthesizer, [first, note_arrival, second, whole])
    assert events == ["One.", "rest arrives", "Two."]
    texts = [
        output.payload["text"] for output in outputs if output.format == TEXT_DELTA
    ]
    assert texts == ["One. Tw", "o."]
    assert [control_type(output) for output in outputs].count(TURN_DONE) == 1
    assert control_type(outputs[-1]) == TURN_DONE


@pytest.mark.asyncio
async def test_pieces_of_a_streamed_speech_that_failed_start_no_other_speech():
    synthesizer = speech_synthesizer()
    attempts: list[str] = []

    async def speak(sentence: str, language: str | None):
        attempts.append(sentence)
        raise httpx.ConnectError("refused")
        yield

    async def let_it_fail():
        for _ in range(20):
            await asyncio.sleep(0)

    synthesizer.speak = speak
    first, second, whole = spoken_pieces(["One. ", "Two."])
    outputs = await drain(synthesizer, [first, let_it_fail, second, whole])
    assert attempts == ["One."]
    done = [output for output in outputs if control_type(output) == TURN_DONE]
    assert len(done) == 1 and "refused" in done[0].payload[SPEECH_ERROR]
