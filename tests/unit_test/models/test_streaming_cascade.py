# SPDX-License-Identifier: Apache-2.0
"""Cascade component protocols, configuration and launch profiles."""

import asyncio
import base64
import io
import json
import re
import wave
from dataclasses import replace
from pathlib import Path

import httpx
import numpy as np
import pytest
import yaml

from sglang_omni.admission import QueueFullError
from sglang_omni.audio.segmentation import UtteranceSegmenter
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.models.streaming_cascade.alignment import AlignmentConfig
from sglang_omni.models.streaming_cascade.base import (
    ChatMessage,
    Replies,
    audio_chunk,
    messages_chunk,
    messages_of,
    sentence_end,
    sentences,
    text_chunk,
    transcript_chunk,
)
from sglang_omni.models.streaming_cascade.components import COMPONENTS
from sglang_omni.models.streaming_cascade.config import (
    ComponentGraphConfig,
    StreamingCascadePipelineConfig,
)
from sglang_omni.models.streaming_cascade.dialog import CUT_REPLY_NOTE, heard_prefix
from sglang_omni.models.streaming_cascade.generators import ChatDelta
from sglang_omni.models.streaming_cascade.numerals import spoken_text
from sglang_omni.models.streaming_cascade.recognizers import probe_realtime_service
from sglang_omni.models.streaming_cascade.running_transcript import (
    RunningTranscript,
    RunningTranscriptConfig,
)
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.proto.streams import (
    AUDIO,
    COMMIT,
    CONFIGURE,
    INTERRUPT,
    MESSAGES,
    PCM16,
    PCM_F32,
    REPLY_DELIVERED,
    REPLY_HEARD,
    REPLY_PROGRESS,
    SESSION_CONTROL,
    SPEECH_STARTED,
    SPEECH_STOPPED,
    TEXT_DELTA,
    TRANSCRIPT_FORMAT,
    TRUNCATE,
    TURN_DONE,
    TURN_INTERRUPTED,
    TURN_QUEUED,
    TURN_STARTED,
    AudioPort,
    SessionPorts,
    control_type,
)
from sglang_omni.utils.imports import import_string
from tests.unit_test.fixtures.cascade_backends import CHAT_MODEL
from tests.unit_test.fixtures.cascade_fakes import QUIET, build, drain
from tests.unit_test.fixtures.session_outputs import control_chunk

PROFILES = Path("examples/streaming_cascade/profiles")
PROFILE_NAMES = (
    "nemotron-qwen3-tts",
    "nemotron-stage-qwen3-tts",
    "qwen3-asr-qwen3-tts-vivian",
    "qwen3-asr-qwen3.8-27b-vivian",
)
HTTP_PROFILE = PROFILES / "nemotron-qwen3-tts" / "launch.yaml"
CASCADE_PORTS = SessionPorts(
    input=(AudioPort(PCM16, 16000),),
    output=AudioPort(PCM_F32, 24000),
    unit_ms=200,
    controls=frozenset({COMMIT, CONFIGURE, INTERRUPT, TRUNCATE}),
    hot_settings=frozenset({"instructions", "sampling", "turn_detection"}),
    sampling_parameters=("temperature", "max_new_tokens_per_unit"),
    turn_detection_types=("server_vad",),
)


def profile_config(profile: str) -> StreamingCascadePipelineConfig:
    return ConfigManager.from_file(str(PROFILES / profile / "launch.yaml")).config


@pytest.mark.asyncio
async def test_speech_over_a_reply_makes_the_dialog_interrupt_before_turn_interrupted():
    dialog = build({"kind": "dialog"})
    outputs = await drain(
        dialog,
        [
            transcript_chunk("question", True),
            text_chunk("Hello world.", 1),
            control_chunk(SPEECH_STARTED),
        ],
    )
    assert [control_type(o) for o in outputs] == [
        TURN_QUEUED,
        TURN_STARTED,
        None,
        INTERRUPT,
        TURN_INTERRUPTED,
    ]


@pytest.mark.asyncio
async def test_a_truncate_after_turn_done_cuts_the_reply_that_was_still_playing():
    dialog = build({"kind": "dialog"})
    await drain(
        dialog,
        [
            transcript_chunk("question", True),
            text_chunk("Hello world. Second sentence.", 1),
            lambda: dialog.control(
                control_chunk(REPLY_PROGRESS, turn=1, text_end=12, audio_ms=1000)
            ),
            lambda: dialog.control(
                control_chunk(REPLY_PROGRESS, turn=1, text_end=29, audio_ms=2000)
            ),
            control_chunk(TURN_DONE, turn=1),
            control_chunk(TRUNCATE, t_ms=1500),
        ],
    )
    assert dialog.messages[-1].content == "Hello world."
    assert dialog.messages[-1].is_cut
    await drain(dialog, [transcript_chunk("next", True)])
    await drain(dialog, [control_chunk(TRUNCATE, t_ms=0)])
    assert dialog.messages[-2].content == "Hello world."


REPLY = "One two.\nThree four. 五六。七八。"
# Note (Junnan Li): Sentence ends at characters 8, 20 and 24 (and the reply's end, 28), with their audio ending at 1000, 2000 and 2500 ms.
REPLY_PROGRESS_POINTS = [(8, 1000.0), (20, 2000.0), (24, 2500.0)]


@pytest.mark.parametrize(
    "heard_ms, text_end, uncertain",
    [
        (0, 0, False),
        (400, 0, True),
        (1000, 8, False),
        (1500, 8, True),
        (2000, 20, False),
        (2400, 20, True),
        (2500, 24, False),
        (2900, 24, True),
    ],
)
def test_heard_prefix_cuts_at_the_last_sentence_heard_to_its_end(
    heard_ms, text_end, uncertain
):
    assert heard_prefix(REPLY_PROGRESS_POINTS, heard_ms, REPLY) == (
        text_end,
        uncertain,
    )


def test_heard_prefix_is_certain_once_the_whole_reply_was_heard():
    assert heard_prefix([*REPLY_PROGRESS_POINTS, (28, 3000.0)], 3600, REPLY) == (
        28,
        False,
    )
    assert heard_prefix([(8, 1000.0)], 1500, "One two.  ") == (8, False)


@pytest.mark.asyncio
async def test_sentences_keep_the_reply_spacing_and_split_cjk_without_spaces():
    deltas = [
        text_chunk(part, 1) for part in ("One two.\nThr", "ee four. 五六。", "七八。")
    ]

    async def stream():
        for delta in deltas:
            yield delta

    spans = [span async for span in sentences(stream(), max_words=24)]
    assert spans == ["One two.", "\nThree four.", " 五六。", "七八。"]
    assert "".join(spans) == REPLY


async def spans_of(*parts: str, max_words: int = 24) -> list[str]:
    async def stream():
        for part in parts:
            yield text_chunk(part, 1)

    return [span async for span in sentences(stream(), max_words)]


@pytest.mark.asyncio
async def test_sentences_stream_a_cjk_reply_sentence_by_sentence():
    spans = await spans_of("你好。今天天", "气很好！我们走吧；", "好…再见。")
    assert spans == ["你好。", "今天天气很好！", "我们走吧；", "好…", "再见。"]


@pytest.mark.asyncio
async def test_sentences_end_each_line_of_a_list():
    reply = "Steps:\n- open the door\n- walk in; sit down\n"
    spans = await spans_of(reply)
    assert spans == ["Steps:", "\n- open the door", "\n- walk in;", " sit down", "\n"]
    assert "".join(spans) == reply


@pytest.mark.asyncio
async def test_sentences_split_after_an_abbreviation():
    # Note (Junnan Li): Abbreviations are not recognised; "Dr." ends a sentence of its own, which costs one more speech request but keeps the text.
    assert await spans_of("Dr. Smith is here. ", "Welcome!") == [
        "Dr.",
        " Smith is here.",
        " Welcome!",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply, spans",
    [
        (
            "I apologize for the short answer earlier. ",
            ["I apologize for the", " short answer earlier.", " "],
        ),
        (
            "Dolphins recognize their friends, just like people recognize names. ",
            [
                "Dolphins recognize their friends,",
                " just like people recognize names.",
                " ",
            ],
        ),
        (
            "While a full history is too long to give here, the main points are clear. ",
            [
                "While a full history is too",
                " long to give here,",
                " the main points are clear.",
                " ",
            ],
        ),
    ],
    ids=["two-halves", "clause-end", "long-sentence"],
)
async def test_a_long_sentence_is_cut_into_spans_of_at_least_half_the_cap(reply, spans):
    """A sentence one word past the cap left its last word to be synthesized alone, which sometimes ran the speech model to its length limit."""
    assert await spans_of(*(word + " " for word in reply.split()), max_words=6) == spans


def test_a_long_sentence_is_cut_once_its_rest_cannot_be_shorter_than_half_the_cap():
    assert sentence_end("one two three four five six ", 6) is None
    assert sentence_end("one two three four five six seven ", 6) is None
    assert sentence_end("one two three four five six seven eight ", 6) == len(
        "one two three four five six"
    )


@pytest.mark.asyncio
async def test_sentences_keep_the_word_cap():
    assert await spans_of("one two three four five", max_words=2) == [
        "one two",
        " three four",
        " five",
    ]


@pytest.mark.asyncio
async def test_a_truncate_after_speech_over_a_reply_keeps_the_heard_sentences_and_marks_the_cut():
    dialog = build({"kind": "dialog"})
    progress = [
        lambda text_end=text_end, audio_ms=audio_ms: dialog.control(
            control_chunk(REPLY_PROGRESS, turn=1, text_end=text_end, audio_ms=audio_ms)
        )
        for text_end, audio_ms in REPLY_PROGRESS_POINTS[:2]
    ]
    outputs = await drain(
        dialog,
        [
            transcript_chunk("question", True),
            text_chunk(REPLY, 1),
            *progress,
            control_chunk(SPEECH_STARTED),
            control_chunk(TRUNCATE, t_ms=2300),
            transcript_chunk("what did you say", True),
        ],
    )
    assert [control_type(o) for o in outputs][3:5] == [INTERRUPT, TURN_INTERRUPTED]
    assert dialog.messages[2].content == "One two.\nThree four."
    assert dialog.messages[2].uncertain
    prompt = [o for o in outputs if o.format == MESSAGES][-1]
    assert prompt.payload["messages"][0]["content"].endswith(CUT_REPLY_NOTE)
    assert prompt.payload["messages"][2] == {
        "role": "assistant",
        "content": "One two.\nThree four.…",
    }
    assert messages_of(prompt)[-1].content == "what did you say"


@pytest.mark.asyncio
async def test_progress_of_the_interrupted_reply_still_counts_after_the_interrupt():
    dialog = build({"kind": "dialog"})
    await drain(
        dialog,
        [
            transcript_chunk("question", True),
            text_chunk(REPLY, 1),
            control_chunk(SPEECH_STARTED),
            lambda: dialog.control(
                control_chunk(REPLY_PROGRESS, turn=1, text_end=8, audio_ms=1000)
            ),
            control_chunk(TRUNCATE, t_ms=1000),
        ],
    )
    assert dialog.messages[2].content == "One two."
    assert not dialog.messages[2].uncertain


@pytest.mark.asyncio
async def test_progress_that_arrives_after_the_truncate_re_cuts_the_interrupted_reply():
    dialog = build({"kind": "dialog"})
    await drain(
        dialog,
        [
            transcript_chunk("A question?", True),
            text_chunk(REPLY, 1),
            control_chunk(SPEECH_STARTED, t_ms=900),
            control_chunk(TRUNCATE, t_ms=1000),
            lambda: dialog.control(
                control_chunk(REPLY_PROGRESS, turn=1, text_end=8, audio_ms=1000)
            ),
        ],
    )
    assert dialog.messages[2].content == "One two."
    assert not dialog.messages[2].uncertain


@pytest.mark.asyncio
async def test_each_cut_after_the_client_truncate_is_reported_as_reply_heard():
    dialog = build({"kind": "dialog"})
    outputs = await drain(
        dialog,
        [
            transcript_chunk("A question?", True),
            text_chunk(REPLY, 1),
            control_chunk(SPEECH_STARTED, t_ms=900),
            control_chunk(REPLY_DELIVERED, turn=1, audio_ms=2300),
            control_chunk(TRUNCATE, t_ms=1000),
            lambda: dialog.control(
                control_chunk(REPLY_PROGRESS, turn=1, text_end=8, audio_ms=1000)
            ),
        ],
    )
    assert [
        (o.turn, o.payload["text_end"])
        for o in outputs
        if control_type(o) == REPLY_HEARD
    ] == [(1, 0), (1, 8)]


def interrupted_dialog(*after: object) -> tuple[object, list[object]]:
    """A dialog whose reply REPLY was interrupted after its first two sentence ends arrived, with `after` fed next."""
    dialog = build({"kind": "dialog"})
    progress = [
        lambda text_end=text_end, audio_ms=audio_ms: dialog.control(
            control_chunk(REPLY_PROGRESS, turn=1, text_end=text_end, audio_ms=audio_ms)
        )
        for text_end, audio_ms in REPLY_PROGRESS_POINTS[:2]
    ]
    return dialog, [
        transcript_chunk("question", True),
        text_chunk(REPLY, 1),
        *progress,
        control_chunk(SPEECH_STARTED),
        *after,
    ]


@pytest.mark.asyncio
async def test_an_interrupted_reply_keeps_nothing_until_its_delivered_audio_is_known():
    dialog, inputs = interrupted_dialog()
    await drain(dialog, inputs)
    assert (dialog.messages[2].role, dialog.messages[2].content) == ("assistant", "")


@pytest.mark.asyncio
async def test_without_a_truncate_the_interrupted_reply_keeps_the_sentences_delivered_to_their_end():
    dialog, inputs = interrupted_dialog(
        control_chunk(REPLY_DELIVERED, turn=1, audio_ms=2300)
    )
    await drain(dialog, inputs)
    assert dialog.messages[2].content == "One two.\nThree four."
    assert dialog.messages[2].uncertain


@pytest.mark.asyncio
async def test_a_later_truncate_moves_the_cut_back_but_never_past_the_delivered_audio():
    dialog, inputs = interrupted_dialog(
        control_chunk(REPLY_DELIVERED, turn=1, audio_ms=2300),
        control_chunk(TRUNCATE, t_ms=1200),
    )
    await drain(dialog, inputs)
    assert dialog.messages[2].content == "One two."
    await drain(dialog, [control_chunk(TRUNCATE, t_ms=2900)])
    assert dialog.messages[2].content == "One two.\nThree four."


def history(dialog) -> list[tuple[str, str]]:
    return [(message.role, message.content) for message in dialog.messages]


def prompts(outputs: list[TimedChunk]) -> list[list[tuple[str, str]]]:
    """The history of each messages chunk the dialog sent to the language model."""
    return [
        [(message.role, message.content) for message in messages_of(o)]
        for o in outputs
        if o.format == MESSAGES
    ]


SECOND_REPLY = "Sure. Here it is."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "during, after, second",
    [
        ([control_chunk(TRUNCATE, turn=1, t_ms=1200)], [], SECOND_REPLY),
        ([], [control_chunk(TRUNCATE, turn=1, t_ms=200)], SECOND_REPLY),
        ([], [control_chunk(TRUNCATE, turn=2, t_ms=1000)] * 2, "Sure."),
        ([], [control_chunk(TRUNCATE, turn=2, t_ms=60000)], SECOND_REPLY),
        ([], [control_chunk(TRUNCATE, t_ms=1000)], "Sure."),
    ],
    ids=["late", "stale", "repeated", "past-the-end", "unnamed"],
)
async def test_a_truncate_cuts_only_the_reply_of_its_turn(during, after, second):
    """A truncate cut whichever ended reply the dialog still held, so one for an earlier reply cut the next one."""
    dialog, inputs = interrupted_dialog(
        control_chunk(REPLY_DELIVERED, turn=1, audio_ms=2300),
        control_chunk(SPEECH_STOPPED),
        transcript_chunk("next", True),
        text_chunk(SECOND_REPLY, 2),
        *during,
        lambda: dialog.control(
            control_chunk(REPLY_PROGRESS, turn=2, text_end=5, audio_ms=800)
        ),
        lambda: dialog.control(
            control_chunk(REPLY_PROGRESS, turn=2, text_end=17, audio_ms=1600)
        ),
        control_chunk(TURN_DONE, turn=2),
        *after,
    )
    await drain(dialog, inputs)
    assert [message.content for message in dialog.messages[2::2]] == [
        "One two.\nThree four.",
        second,
    ]


@pytest.mark.asyncio
async def test_a_reply_interrupted_in_its_first_sentence_leaves_no_answer_and_joins_the_questions():
    dialog, inputs = interrupted_dialog(
        control_chunk(REPLY_DELIVERED, turn=1, audio_ms=600),
        control_chunk(SPEECH_STOPPED),
        transcript_chunk("wait, what?", True),
    )
    outputs = await drain(dialog, inputs)
    assert prompts(outputs)[-1] == [
        ("system", dialog.config.system_prompt),
        ("user", "question wait, what?"),
    ]


@pytest.mark.asyncio
async def test_a_reply_interrupted_before_any_audio_leaves_no_answer():
    dialog = build({"kind": "dialog"})
    outputs = await drain(
        dialog,
        [
            transcript_chunk("I tried several times,", True),
            control_chunk(SPEECH_STARTED),
            control_chunk(REPLY_DELIVERED, turn=1, audio_ms=0),
            control_chunk(SPEECH_STOPPED),
            transcript_chunk("and this is my fourth attempt.", True),
            text_chunk("Sorry to hear that.", 2),
            control_chunk(TURN_DONE, turn=2),
        ],
    )
    assert prompts(outputs)[-1] == [
        ("system", dialog.config.system_prompt),
        ("user", "I tried several times, and this is my fourth attempt."),
    ]
    assert history(dialog)[1:] == [
        ("user", "I tried several times, and this is my fourth attempt."),
        ("assistant", "Sorry to hear that."),
    ]


@pytest.mark.asyncio
async def test_a_reply_truncated_to_nothing_leaves_no_answer():
    dialog = build({"kind": "dialog"})
    outputs = await drain(
        dialog,
        [
            transcript_chunk("question", True),
            text_chunk(REPLY, 1),
            control_chunk(TURN_DONE, turn=1),
            control_chunk(TRUNCATE, t_ms=0),
            transcript_chunk("hello?", True),
        ],
    )
    assert prompts(outputs)[-1] == [
        ("system", dialog.config.system_prompt),
        ("user", "question hello?"),
    ]


@pytest.mark.asyncio
async def test_a_reply_that_finished_without_text_leaves_no_answer():
    dialog = build({"kind": "dialog"})
    outputs = await drain(
        dialog,
        [
            transcript_chunk("question", True),
            control_chunk(TURN_DONE, turn=1),
            transcript_chunk("are you there?", True),
        ],
    )
    assert prompts(outputs)[-1] == [
        ("system", dialog.config.system_prompt),
        ("user", "question are you there?"),
    ]


@pytest.mark.asyncio
async def test_a_partly_heard_reply_keeps_its_heard_prefix_between_the_questions():
    dialog, inputs = interrupted_dialog(
        control_chunk(REPLY_DELIVERED, turn=1, audio_ms=2300),
        control_chunk(SPEECH_STOPPED),
        transcript_chunk("wait, what?", True),
    )
    outputs = await drain(dialog, inputs)
    assert prompts(outputs)[-1] == [
        ("system", f"{dialog.config.system_prompt} {CUT_REPLY_NOTE}"),
        ("user", "question"),
        ("assistant", "One two.\nThree four.…"),
        ("user", "wait, what?"),
    ]


@pytest.mark.asyncio
async def test_utterances_over_unheard_replies_in_a_row_join_one_user_message():
    dialog = build({"kind": "dialog"})
    outputs = await drain(
        dialog,
        [
            transcript_chunk("first", True),
            text_chunk("Reply one.", 1),
            lambda: dialog.control(
                control_chunk(REPLY_PROGRESS, turn=1, text_end=10, audio_ms=1000)
            ),
            control_chunk(TURN_DONE, turn=1),
            transcript_chunk("second", True),
            control_chunk(SPEECH_STARTED),
            control_chunk(REPLY_DELIVERED, turn=2, audio_ms=0),
            control_chunk(SPEECH_STOPPED),
            transcript_chunk("third", True),
            control_chunk(SPEECH_STARTED),
            control_chunk(REPLY_DELIVERED, turn=3, audio_ms=0),
            control_chunk(SPEECH_STOPPED),
            transcript_chunk("fourth", True),
            text_chunk("Reply four.", 4),
            lambda: dialog.control(
                control_chunk(REPLY_PROGRESS, turn=4, text_end=11, audio_ms=1000)
            ),
            control_chunk(SPEECH_STARTED),
            control_chunk(REPLY_DELIVERED, turn=4, audio_ms=1000),
            control_chunk(SPEECH_STOPPED),
            transcript_chunk("fifth", True),
        ],
    )
    system = ("system", dialog.config.system_prompt)
    assert prompts(outputs) == [
        [system, ("user", "first")],
        [system, ("user", "first"), ("assistant", "Reply one."), ("user", "second")],
        [
            system,
            ("user", "first"),
            ("assistant", "Reply one."),
            ("user", "second third"),
        ],
        [
            system,
            ("user", "first"),
            ("assistant", "Reply one."),
            ("user", "second third fourth"),
        ],
        [
            ("system", f"{dialog.config.system_prompt} {CUT_REPLY_NOTE}"),
            ("user", "first"),
            ("assistant", "Reply one."),
            ("user", "second third fourth"),
            ("assistant", "Reply four.…"),
            ("user", "fifth"),
        ],
    ]


@pytest.mark.asyncio
async def test_the_next_request_to_the_language_model_has_no_empty_answer():
    dialog, inputs = interrupted_dialog(
        control_chunk(TRUNCATE, t_ms=0),
        control_chunk(SPEECH_STOPPED),
        transcript_chunk("wait, what?", True),
    )
    prompt = [o for o in await drain(dialog, inputs) if o.format == MESSAGES][-1]
    bodies = []

    def answer(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        events = [
            {"choices": [{"delta": {"content": "Sure."}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        stream = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        return httpx.Response(200, text=f"{stream}data: [DONE]\n\n")

    generator = build({"kind": "openai_chat", **HTTP_SERVICE})
    generator.client = httpx.AsyncClient(transport=httpx.MockTransport(answer))
    await drain(generator, [prompt])
    assert bodies[0]["messages"] == [
        {"role": "system", "content": dialog.config.system_prompt},
        {"role": "user", "content": "question wait, what?"},
    ]


@pytest.mark.asyncio
async def test_delivered_audio_of_another_turn_does_not_cut_the_reply():
    dialog, inputs = interrupted_dialog(
        control_chunk(REPLY_DELIVERED, turn=7, audio_ms=2300)
    )
    await drain(dialog, inputs)
    assert dialog.messages[2].content == ""


@pytest.mark.asyncio
async def test_a_truncate_past_the_reply_end_keeps_it_whole_and_unmarked():
    dialog = build({"kind": "dialog"})
    progress = [
        lambda text_end=text_end, audio_ms=audio_ms: dialog.control(
            control_chunk(REPLY_PROGRESS, turn=1, text_end=text_end, audio_ms=audio_ms)
        )
        for text_end, audio_ms in [*REPLY_PROGRESS_POINTS, (len(REPLY), 3000.0)]
    ]
    outputs = await drain(
        dialog,
        [
            transcript_chunk("question", True),
            text_chunk(REPLY, 1),
            *progress,
            control_chunk(TURN_DONE, turn=1),
            control_chunk(TRUNCATE, t_ms=3600),
            transcript_chunk("next", True),
        ],
    )
    assert not dialog.messages[2].is_cut
    assert prompts(outputs)[-1] == [
        ("system", dialog.config.system_prompt),
        ("user", "question"),
        ("assistant", REPLY),
        ("user", "next"),
    ]


@pytest.mark.asyncio
async def test_a_reply_interrupted_after_all_its_text_was_heard_is_still_marked_cut():
    dialog = build({"kind": "dialog"})
    outputs = await drain(
        dialog,
        [
            transcript_chunk("question", True),
            text_chunk("One two.", 1),
            lambda: dialog.control(
                control_chunk(REPLY_PROGRESS, turn=1, text_end=8, audio_ms=1000)
            ),
            control_chunk(SPEECH_STARTED),
            control_chunk(TRUNCATE, t_ms=1500),
            control_chunk(SPEECH_STOPPED),
            transcript_chunk("wait, what?", True),
        ],
    )
    assert dialog.messages[2].is_cut
    assert prompts(outputs)[-1] == [
        ("system", f"{dialog.config.system_prompt} {CUT_REPLY_NOTE}"),
        ("user", "question"),
        ("assistant", "One two.…"),
        ("user", "wait, what?"),
    ]


@pytest.mark.asyncio
async def test_a_finished_reply_the_model_ends_with_an_ellipsis_does_not_read_as_cut():
    dialog = build({"kind": "dialog"})
    outputs = await drain(
        dialog,
        [
            transcript_chunk("question", True),
            text_chunk("Rome was an empire,…", 1),
            control_chunk(TURN_DONE, turn=1),
            transcript_chunk("and then?", True),
        ],
    )
    assert dialog.messages[2].content == "Rome was an empire,…"
    assert prompts(outputs)[-1] == [
        ("system", dialog.config.system_prompt),
        ("user", "question"),
        ("assistant", "Rome was an empire,"),
        ("user", "and then?"),
    ]


@pytest.mark.asyncio
async def test_a_reply_that_finished_stays_whole_without_a_truncate():
    dialog = build({"kind": "dialog"})
    await drain(
        dialog,
        [
            transcript_chunk("question", True),
            text_chunk(REPLY, 1),
            control_chunk(TURN_DONE, turn=1),
            lambda: dialog.control(
                control_chunk(REPLY_PROGRESS, turn=1, text_end=8, audio_ms=1000)
            ),
        ],
    )
    assert dialog.messages[2].content == REPLY


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resumed, note, history",
    [
        (
            "before_the_reply_opens",
            "",
            [("user", "Ten companies. That let you teach English online.")],
        ),
        (
            "before_its_audio",
            "",
            [("user", "Ten companies. That let you teach English online.")],
        ),
        (
            "after_its_first_sentence",
            f" {CUT_REPLY_NOTE}",
            [
                ("user", "Ten companies."),
                ("assistant", "One two.…"),
                ("user", "That let you teach English online."),
            ],
        ),
    ],
)
async def test_speech_that_resumes_after_the_utterance_ended_interrupts_its_reply(
    resumed, note, history
):
    dialog = build({"kind": "dialog"})
    question = transcript_chunk("Ten companies.", True)
    speech = control_chunk(SPEECH_STARTED, t_ms=1800)
    reply = {
        "before_the_reply_opens": [speech, question],
        "before_its_audio": [question, text_chunk(REPLY, 1), speech],
        "after_its_first_sentence": [
            question,
            text_chunk(REPLY, 1),
            lambda: dialog.control(
                control_chunk(REPLY_PROGRESS, turn=1, text_end=8, audio_ms=1000)
            ),
            speech,
            control_chunk(REPLY_DELIVERED, turn=1, audio_ms=1000),
        ],
    }[resumed]
    outputs = await drain(
        dialog,
        [
            control_chunk(SPEECH_STARTED, t_ms=0),
            control_chunk(SPEECH_STOPPED, t_ms=1000),
            *reply,
            control_chunk(SPEECH_STOPPED, t_ms=4000),
            transcript_chunk("That let you teach English online.", True),
        ],
    )
    assert [control_type(o) for o in outputs if o.turn == 1] == [
        TURN_QUEUED,
        TURN_STARTED,
        None,
        INTERRUPT,
        TURN_INTERRUPTED,
    ]
    assert [control_type(o) for o in outputs if o.turn == 2] == [
        TURN_QUEUED,
        TURN_STARTED,
        None,
    ]
    assert [(m.role, m.content) for m in messages_of(outputs[-1])] == [
        ("system", dialog.config.system_prompt + note),
        *history,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("end", [SPEECH_STOPPED, COMMIT])
async def test_a_reply_opens_once_the_speech_of_its_utterance_ended(end):
    dialog = build({"kind": "dialog"})
    outputs = await drain(
        dialog,
        [
            control_chunk(SPEECH_STARTED, t_ms=0),
            control_chunk(end, t_ms=1000),
            transcript_chunk("question", True),
            text_chunk("Hello world.", 1),
            control_chunk(TURN_DONE, turn=1),
        ],
    )
    assert [control_type(o) for o in outputs] == [TURN_QUEUED, TURN_STARTED, None]
    assert dialog.messages[-1].content == "Hello world."


@pytest.mark.asyncio
async def test_replies_forget_interrupted_turns_once_a_later_reply_starts():
    replies = Replies()
    for turn in (1, 2, 4):
        await replies.interrupt(turn)
    replies.start(3, asyncio.sleep(0))
    await replies.finish()
    assert replies.interrupted == {4}


def test_extra_body_cannot_set_the_fields_the_chat_kind_sets():
    with pytest.raises(ValueError, match="extra_body cannot set"):
        COMPONENTS.parse(
            {"kind": "openai_chat", **HTTP_SERVICE, "extra_body": {"stream": False}}
        )
    config = COMPONENTS.parse(
        {
            "kind": "openai_chat",
            **HTTP_SERVICE,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
    )
    assert config.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def declared(nodes=None, publish=None, **fields) -> StreamingCascadePipelineConfig:
    """The HTTP profile's declaration with `nodes` and `publish` replacing its own."""
    data = yaml.safe_load(HTTP_PROFILE.read_text())
    return StreamingCascadePipelineConfig(
        model_path="m",
        session_nodes={**data["session_nodes"], **(nodes or {})},
        session_publish={**data["session_publish"], **(publish or {})},
        **fields,
    )


HTTP_SERVICE = {"base_url": "http://127.0.0.1:8000/v1", "model": "test-model"}
DECLARATION_ERRORS = r"""
- [{nodes: {tts: {kind: openai_speech, base_url: "http://127.0.0.1:8000/v1", model: test-model, inputs: [input]}}}, produces]
- [{nodes: {dialog: {kind: dialog, inputs: ["asr:session", llm]}}}, cannot take]
- [{nodes: {asr: {kind: nope, inputs: [input]}}}, unknown component kind]
- [{nodes: {vad: {stages: [vad], inputs: [input], threshold: 0.5}}}, belong to a node of a kind]
- [{nodes: {asr: {kind: realtime_transcription, base_url: "ws://127.0.0.1:8000", model: test-asr, stages: [asr], inputs: [input]}}}, either stages or a kind]
- [{nodes: {asr: {stages: [ears], inputs: [input]}}}, no stage group]
- [{publish: {ghost: [session]}}, publish names unknown nodes]
- [{publish: {asr: [text_delta]}}, does not produce]
- [{publish: {tts: [pcm_f32le]}}, no published node ends turns]
- [{publish: {llm: [session]}}, turns end twice]
- [{publish: {dialog: []}}, open turns]
- [{nodes: {extra: {kind: tool, inputs: [llm], tools: {}}}}, unknown component kind]
"""


@pytest.mark.parametrize(
    "declaration, message",
    yaml.safe_load(DECLARATION_ERRORS),
)
def test_declaration_errors(declaration, message):
    with pytest.raises(ValueError, match=message):
        declared(**declaration)


def test_settings_merge_and_standard_names():
    settings = declared().component_graph()
    merged = settings.apply(
        {
            "instructions": "Be terse.",
            "sampling": {"temperature": 0.2, "max_new_tokens_per_unit": 64},
            "turn_detection": {"type": "server_vad", "silence_duration_ms": 800},
            "nodes": {"tts": {"voice": "b"}, "asr": {"timeout_s": 30}},
        }
    )
    assert merged.node("dialog").component.system_prompt == "Be terse."
    llm = merged.node("llm").component
    local = "http://127.0.0.1:8000/v1"
    assert (llm.temperature, llm.max_tokens, llm.base_url) == (0.2, 64, local)
    assert merged.node("tts").component.voice == "b"
    assert merged.node("asr").component.timeout_s == 30
    assert settings.node("dialog").component.system_prompt != "Be terse."
    asr = settings.apply(
        {"nodes": {"asr": {"kind": "openai_transcription", **HTTP_SERVICE}}}
    ).node("asr")
    assert (asr.component.base_url, asr.inputs) == (local, ("input", "vad:session"))
    assert ComponentGraphConfig.model_validate_json(merged.model_dump_json()) == merged
    for update, message in [
        ({"nodes": {"ghost": {}}}, "unknown node"),
        ({"nodes": {"llm": {"inputs": ["input"]}}}, "rewire"),
        ({"nodes": {"tts": {"voice": 3}}}, "voice"),
        ({"nodes": {"llm": {"base_url": "file:///tmp"}}}, "base_url"),
        ({"publish": {}}, "unknown cascade settings"),
        ({"sampling": {"top_k": 3}}, "unsupported sampling"),
        ({"instructions": 3}, "string"),
    ]:
        with pytest.raises(ValueError, match=message):
            settings.apply(update)


TEXT_ONLY = """
nodes:
  - {name: asr, kind: openai_transcription, base_url: "http://127.0.0.1:8000/v1", model: test-model, inputs: [input]}
  - {name: dialog, kind: dialog, inputs: [asr, llm]}
  - {name: llm, kind: openai_chat, base_url: "http://127.0.0.1:8000/v1", model: test-model, inputs: [dialog]}
publish: {dialog: [session], llm: [text_delta, session]}
"""


def test_profile_graph_turns_formats_and_ports():
    settings = declared().component_graph()
    assert [n.name for n in settings.nodes if n.opens_turns] == ["dialog"]
    assert [n.name for n in settings.nodes if n.ends_turns] == ["llm", "tts"]
    assert settings.publish["llm"] == [TEXT_DELTA]
    hot_settings = frozenset({"instructions", "sampling"})
    assert settings.ports(200) == replace(
        CASCADE_PORTS, hot_settings=hot_settings, turn_detection_types=()
    )
    assert declared(unit_ms=100).session_ports().unit_ms == 100
    text_only = ComponentGraphConfig.model_validate(yaml.safe_load(TEXT_ONLY)).ports(
        200
    )
    assert (text_only.output, text_only.hot_settings) == (None, hot_settings)
    chat = settings.node("llm")
    assert (chat.consumes, chat.required) == ({MESSAGES, SESSION_CONTROL}, {MESSAGES})
    dialog_formats = {TRANSCRIPT_FORMAT, TEXT_DELTA, SESSION_CONTROL}
    assert settings.node("dialog").required == dialog_formats


def test_a_dialog_without_the_input_interrupt_edge_grants_no_client_stop():
    """A client response.cancel travels the control edges from the input, and a graph without one fails the control, so its ports must not offer it."""
    dialog = yaml.safe_load(HTTP_PROFILE.read_text())["session_nodes"]["dialog"]
    input_interrupt = {"kind": "control", "source": "input", "event": "interrupt"}
    inputs = [entry for entry in dialog["inputs"] if entry != input_interrupt]
    deaf = declared(nodes={"dialog": {**dialog, "inputs": inputs}})
    assert INTERRUPT not in deaf.component_graph().ports(200).controls


def test_segmenter_gates_on_speech_controls():
    segmenter = UtteranceSegmenter()
    first = TimedChunk(AUDIO, 0, 100, 0, QUIET, format=PCM16)
    assert segmenter.step(first) == ["open", first]
    assert segmenter.step(control_chunk(SPEECH_STARTED, t_ms=0)) == []
    assert segmenter.step(control_chunk(SPEECH_STOPPED, t_ms=50)) == ["finish"]
    held = [
        TimedChunk(AUDIO, t, 100, 0, QUIET, format=PCM16) for t in range(100, 2600, 100)
    ]
    for chunk in held:
        assert segmenter.step(chunk) == []
    assert segmenter.pending_audio[0].t_start_ms == 600
    restart = control_chunk(SPEECH_STARTED, t_ms=2350)
    assert segmenter.step(restart) == ["open", *held[-3:]]
    assert segmenter.step(control_chunk(COMMIT)) == ["finish"]
    assert segmenter.step(control_chunk(COMMIT)) == ["empty"]
    assert segmenter.finish() == []


@pytest.mark.asyncio
async def test_realtime_transcription_streams_partials_then_final_per_utterance(
    realtime_transcription_server,
):
    base_url, appended, client_events = realtime_transcription_server
    frames = [bytes([index]) * 640 for index in range(4)]
    audio = [
        TimedChunk(AUDIO, index * 20, 20, index, frame, format=PCM16)
        for index, frame in enumerate(frames)
    ]
    spec = {
        "kind": "realtime_transcription",
        "base_url": base_url,
        "model": "nvidia/nemotron-3.5-asr-streaming-0.6b",
    }
    outputs = await drain(build(spec), [*audio, control_chunk(COMMIT)])
    assert [chunk.payload for chunk in outputs] == [
        {"text": "hello", "is_final": False},
        {"text": "hello there", "is_final": False},
        {"text": "hello there", "is_final": True},
    ]
    assert appended == list(enumerate(frames))
    assert client_events == [
        "session.update",
        *["input_audio_buffer.append"] * 4,
        "sglang.input_audio.end",
        "session.close",
    ]


@pytest.mark.asyncio
async def test_openai_speech_flushes_each_reply_before_turn_done(backends):
    settings, _, spoken, _ = backends
    tts = settings.component_graph().node("tts").component.model_dump()
    inputs = [
        text_chunk("Hello wor", 1),
        text_chunk("ld. Second sentence here", 1),
        control_chunk(TURN_DONE, turn=1),
        text_chunk("Again", 2),
        control_chunk(TURN_DONE, turn=2),
    ]
    outputs = await drain(build({**tts, "max_sentence_words": 24}), inputs)
    outputs = [o for o in outputs if control_type(o) != REPLY_PROGRESS]
    assert spoken == ["Hello world.", "Second sentence here", "Again"]
    labels = [control_type(o) or o.format for o in outputs]
    runs = [label for i, label in enumerate(labels) if i == 0 or labels[i - 1] != label]
    assert runs == [PCM_F32, TURN_DONE] * 2
    assert [o.turn for o in outputs if o.format == SESSION_CONTROL] == [1, 2]
    samples = [
        np.frombuffer(o.payload, dtype="<f4") for o in outputs if o.modality == AUDIO
    ]
    assert np.concatenate(samples).tolist() == [0.0, 0.5, -0.5] * 3


@pytest.mark.asyncio
async def test_openai_speech_reports_each_sentence_end_as_a_character_offset(backends):
    settings, _, spoken, _ = backends
    tts = settings.component_graph().node("tts").component.model_dump()
    deltas = ["Hello wor", "ld.\n  Second  sentence", " here. 你好。再见。"]
    outputs = await drain(
        build({**tts, "max_sentence_words": 24}),
        [*(text_chunk(delta, 1) for delta in deltas), control_chunk(TURN_DONE, turn=1)],
    )
    progress = [o.payload for o in outputs if control_type(o) == REPLY_PROGRESS]
    reply = "".join(deltas)
    assert spoken == ["Hello world.", "Second sentence here.", "你好。", "再见。"]
    assert [reply[: point["text_end"]] for point in progress] == [
        "Hello world.",
        "Hello world.\n  Second  sentence here.",
        "Hello world.\n  Second  sentence here. 你好。",
        reply,
    ]
    sentence_ms = 3 * 1000 / 24000
    assert [point["audio_ms"] for point in progress] == [
        sentence_ms * count for count in (1, 2, 3, 4)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "expected_spoken"),
    [
        pytest.param("太好了😄，祝你开心！🎉", ["太好了，祝你开心！"], id="emoji"),
        pytest.param(
            "哈哈😂😂，好！👍🏽✨", ["哈哈，好！"], id="skin_tone_and_symbols"
        ),
        pytest.param("加油\u2764\ufe0f！", ["加油！"], id="variation_selector"),
        pytest.param(
            "Great job! 🎉👨\u200d👩\u200d👧 See you.",
            ["Great job!", "See you."],
            id="joined_emoji",
        ),
        pytest.param("**重点**是多喝水。", ["重点是多喝水。"], id="emphasis"),
        pytest.param(
            "Run `pip install` and read [the docs](https://example.com).",
            ["Run pip install and read the docs."],
            id="code_and_link",
        ),
        pytest.param(
            "今天最高25℃，体感20°。", ["今天最高25℃，体感20°。"], id="unit_symbols_stay"
        ),
    ],
)
async def test_openai_speech_sends_only_the_spoken_words_to_the_speech_service(
    backends, reply, expected_spoken
):
    settings, _, spoken, _ = backends
    tts = settings.component_graph().node("tts").component.model_dump()
    await drain(
        build({**tts, "max_sentence_words": 24}),
        [text_chunk(reply, 1), control_chunk(TURN_DONE, turn=1)],
    )
    assert spoken == expected_spoken


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wordless_sentence",
    [
        pytest.param("😄😄", id="only_emoji"),
        pytest.param("🎉！", id="emoji_and_punctuation"),
        pytest.param("!?", id="only_punctuation"),
    ],
)
async def test_openai_speech_skips_a_sentence_without_words_but_still_reports_its_end(
    backends, wordless_sentence
):
    settings, _, spoken, _ = backends
    tts = settings.component_graph().node("tts").component.model_dump()
    outputs = await drain(
        build({**tts, "max_sentence_words": 24}),
        [
            text_chunk("好的。", 1),
            text_chunk(wordless_sentence, 1),
            control_chunk(TURN_DONE, turn=1),
        ],
    )
    sentence_ms = 3 * 1000 / 24000
    assert spoken == ["好的。"]
    assert spoken_progress(outputs) == [(3, sentence_ms), (5, sentence_ms)]


@pytest.mark.asyncio
@pytest.mark.parametrize("backends", ["english_tts"], indirect=True)
async def test_openai_speech_tags_each_sentence_with_its_language(backends):
    settings, _, spoken, _ = backends
    tts = settings.component_graph().node("tts").component.model_dump()
    inputs = [
        text_chunk("Hello world. 你好世界。", 1),
        control_chunk(TURN_DONE, turn=1),
    ]
    await drain(build({**tts, "language": "English"}), inputs)
    assert spoken[-1] == "你好世界。" and " ".join(spoken[:-1]) == "Hello world."


@pytest.mark.asyncio
@pytest.mark.parametrize("backends", ["english_tts"], indirect=True)
async def test_openai_speech_reads_the_digits_of_an_english_sentence_as_words(
    backends,
):
    settings, _, spoken, _ = backends
    tts = settings.component_graph().node("tts").component.model_dump()
    await drain(
        build({**tts, "language": "auto", "max_sentence_words": 24}),
        [
            text_chunk("This year is 2024.", 1, "English"),
            control_chunk(TURN_DONE, turn=1),
            text_chunk("今年是2024年。", 2, "Chinese"),
            control_chunk(TURN_DONE, turn=2),
        ],
    )
    assert spoken == ["This year is twenty twenty four.", "今年是2024年。"]


@pytest.mark.parametrize(
    "text, language, spoken, origin",
    [
        ("我有3个", "English", "我有 three 个", [0, 1] + [2] * 7 + [3]),
        ("我有3个", "Chinese", "我有 三 个", [0, 1, 2, 2, 2, 3]),
        ("3.14", "Chinese", "三点一四", [3, 3, 3, 3]),
        ("21", "Japanese", "21", [1, 1]),
        ("In 2024.", "English", "In twenty twenty four.", [0, 1, 2] + [6] * 18 + [7]),
    ],
)
def test_spoken_text_reads_numbers_in_the_sentence_language(
    text, language, spoken, origin
):
    assert spoken_text(text, language) == (spoken, origin)


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        pytest.param(
            "This year is 2024.", "This year is twenty twenty four.", id="year"
        ),
        pytest.param(
            "Built in 1887, 1900, 1905, 2000 and 2005.",
            "Built in eighteen eighty seven, nineteen hundred, nineteen oh five, two thousand and two thousand five.",
            id="years_ending_in_00_to_09",
        ),
        pytest.param(
            "The iPhone 17 costs 799 dollars, or 12,999 yuan.",
            "The iPhone seventeen costs seven hundred ninety nine dollars, or twelve thousand nine hundred ninety nine yuan.",
            id="cardinals_with_a_thousands_separator",
        ),
        pytest.param(
            "Call 138 1234 5678, 13812345678 or 0199.",
            "Call one three eight one two three four five six seven eight, one three eight one two three four five six seven eight or zero one nine nine.",
            id="grouped_long_and_zero_led_digits_are_spelled",
        ),
        pytest.param(
            "Over 21,000 people, about 3.5 million.",
            "Over twenty one thousand people, about three point five million.",
            id="thousands_and_decimals",
        ),
        pytest.param(
            "Codes 007, 001, 00 and 0.",
            "Codes zero zero seven, zero zero one, zero zero and zero.",
            id="multiple_leading_zeros",
        ),
        pytest.param(
            "Version 2.0.1.",
            "Version two point zero . one.",
            id="version_separators",
        ),
        pytest.param(
            "It takes 3-5 days.",
            "It takes three to five days.",
            id="range_separator",
        ),
        pytest.param(
            "The 21st century began on January 1st, 2001, in the 3rd millennium.",
            "The twenty first century began on January first, two thousand one, in the third millennium.",
            id="ordinals",
        ),
    ],
)
def test_spoken_text_reads_english_years_prices_and_phone_numbers(text, spoken):
    spoken_words, origin = spoken_text(text, "English")
    assert spoken_words == spoken
    assert len(origin) == len(spoken) and origin[-1] == len(text) - 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "digits, words",
    [("007", "zero zero seven"), ("001", "zero zero one"), ("00", "zero zero")],
)
async def test_openai_speech_preserves_multiple_leading_zeros(
    digits: str, words: str
) -> None:
    assert await spoken_requests(
        {"language": "English"}, [text_chunk(f"The code is {digits}.", 1, "English")]
    ) == [("English", f"The code is {words}.")]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("It takes 3-5 days.", "It takes three to five days."),
        ("Wait 3 - 5 DAYS.", "Wait three to five DAYS."),
        ("Wait 3\u20135 minutes.", "Wait three to five minutes."),
        ("Walk 3.5-5 kilometers.", "Walk three point five to five kilometers."),
        ("Pay 1,000-2,000 dollars.", "Pay one thousand to two thousand dollars."),
        ("In 2024-2025 years.", "In twenty twenty four to twenty twenty five years."),
        ("Compute 3-5 = -2.", "Compute three - five = - two."),
        ("Version 2.0.1.", "Version two point zero . one."),
        (
            "Call 138-1234-5678.",
            "Call one hundred thirty eight - one thousand two hundred thirty four - five thousand six hundred seventy eight.",
        ),
        ("Model X3-5 days.", "Model X three - five days."),
        ("Code 007-009 days.", "Code zero zero seven - zero zero nine days."),
    ],
)
def test_spoken_text_recognizes_quantity_ranges(text: str, expected: str) -> None:
    spoken, origin = spoken_text(text, "English")
    assert spoken == expected
    assert len(origin) == len(spoken) and origin == sorted(origin)
    assert origin[-1] == len(text) - 1


def test_spoken_range_maps_each_endpoint_and_to_back_to_source() -> None:
    spoken, origin = spoken_text("3-5 days.", "English")
    assert spoken == "three to five days."
    assert origin == [0] * len("three") + [1] * len(" to ") + [2] * len("five") + list(
        range(3, 9)
    )


@pytest.mark.parametrize("language", ["Chinese", "Japanese"])
def test_quantity_range_words_are_only_inserted_for_english(language: str) -> None:
    spoken, origin = spoken_text("3-5 days.", language)
    assert "to" not in spoken
    assert "-" in spoken and len(origin) == len(spoken)


@pytest.mark.asyncio
async def test_openai_speech_reads_a_quantity_range_as_to() -> None:
    assert await spoken_requests(
        {"language": "English"}, [text_chunk("It takes 3-5 days.", 1, "English")]
    ) == [("English", "It takes three to five days.")]


@pytest.mark.asyncio
@pytest.mark.parametrize("backends", ["unlabeled_rate"], indirect=True)
async def test_openai_speech_refuses_audio_of_an_unnamed_rate(backends):
    tts = backends[0].component_graph().node("tts").component.model_dump()
    with pytest.raises(ValueError, match="X-Sample-Rate"):
        await drain(
            build(tts), [text_chunk("Hello.", 1), control_chunk(TURN_DONE, turn=1)]
        )


@pytest.mark.asyncio
async def test_realtime_probe_requests_models_over_http(backends):
    """The loopback server answers no realtime endpoint, so a passing probe asked for `/v1/models` over HTTP."""
    base_url = backends[0].component_graph().node("asr").component.base_url
    ws_url = base_url.replace("http", "ws").removesuffix("/v1")
    await probe_realtime_service(
        COMPONENTS.parse(
            {"kind": "realtime_transcription", "base_url": ws_url, "model": "test-asr"}
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("backends", ["length_chat"], indirect=True)
async def test_the_chat_probe_streams_one_token_sampled_with_the_node_settings(
    backends, backend_requests
):
    llm = backends[0].component_graph().node("llm").component.model_dump()
    config = COMPONENTS.parse({**llm, "temperature": 0.7, "extra_body": {"top_p": 0.9}})
    await COMPONENTS.kind("openai_chat").probe(config)
    [request] = backend_requests
    assert [message["role"] for message in request.pop("messages")] == ["user"]
    assert request == {
        "model": CHAT_MODEL,
        "stream": True,
        "max_tokens": 1,
        "temperature": 0.7,
        "top_p": 0.9,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("minimum_tokens", [0, 1, 16])
@pytest.mark.parametrize("backends", ["length_chat"], indirect=True)
async def test_the_chat_probe_preserves_the_real_request_minimum_tokens(
    backends, backend_requests, minimum_tokens: int
) -> None:
    llm = backends[0].component_graph().node("llm").component.model_dump()
    config = COMPONENTS.parse(
        {
            **llm,
            "max_tokens": 32,
            "extra_body": {"min_tokens": minimum_tokens, "top_p": 0.9},
        }
    )
    await COMPONENTS.kind("openai_chat").probe(config)
    [request] = backend_requests
    assert request["max_tokens"] == 1 and request["min_tokens"] == 0
    assert request["top_p"] == 0.9
    assert config.max_tokens == 32
    assert config.extra_body == {"min_tokens": minimum_tokens, "top_p": 0.9}


@pytest.mark.asyncio
@pytest.mark.parametrize("backends", ["failed_probe"], indirect=True)
async def test_the_chat_probe_fails_the_start_when_the_service_cannot_complete(
    backends,
):
    llm = backends[0].component_graph().node("llm").component
    with pytest.raises(RuntimeError, match="LLM stopped"):
        await COMPONENTS.kind("openai_chat").probe(llm)


def loopback_aligner(backends) -> AlignmentConfig:
    """A forced alignment config whose worker is the loopback server."""
    base_url = backends[0].component_graph().node("asr").component.base_url
    return COMPONENTS.parse(
        {
            "kind": "alignment",
            "method": "forced",
            "alignment_url": base_url.removesuffix("/v1"),
        }
    )


@pytest.mark.asyncio
async def test_the_forced_alignment_probe_aligns_one_clip(backends, backend_requests):
    await COMPONENTS.kind("alignment").probe(loopback_aligner(backends))
    [request] = backend_requests
    assert len(base64.b64decode(request.pop("audio"))) == 2 * 24000
    assert request.pop("text")
    assert request == {"sample_rate_hz": 24000, "audio_format": "pcm16"}


@pytest.mark.asyncio
@pytest.mark.parametrize("backends", ["failed_align"], indirect=True)
async def test_the_forced_alignment_probe_fails_the_start_when_the_worker_cannot_align(
    backends,
):
    with pytest.raises(RuntimeError, match="answered 500"):
        await COMPONENTS.kind("alignment").probe(loopback_aligner(backends))


@pytest.mark.asyncio
async def test_running_transcript_cuts_utterances_at_speech_end_plus_tail():
    node = RunningTranscript(RunningTranscriptConfig(tail_ms=400))

    def text(t_ms: float, delta: str, full_text: str) -> TimedChunk:
        payload = {"text": delta, "full_text": full_text, "is_first_output": False}
        return TimedChunk("text", t_ms, 200, 0, payload, format="text")

    def unit(t_ms: float) -> TimedChunk:
        return TimedChunk(AUDIO, t_ms, 200, 0, b"", format=PCM16)

    async def inputs():
        yield text(0, "hello", "hello")
        yield control_chunk(SPEECH_STOPPED, t_ms=200)
        yield unit(200)
        yield text(400, " there", "hello there")
        yield unit(400)
        yield text(1000, "?", "hello there?")
        yield text(1200, " again", "hello there? again")
        for t_ms in (1200, 1400, 1600):
            yield unit(t_ms)
        yield control_chunk(SPEECH_STOPPED, t_ms=1300)

    outputs = [chunk async for chunk in node.run(inputs())]
    assert [(c.payload["text"], c.payload["is_final"]) for c in outputs] == [
        ("hello", False),
        ("hello there", False),
        ("hello there", True),
        ("again", False),
        ("again", True),
    ]
    assert all(c.format == TRANSCRIPT_FORMAT for c in outputs)
    replacement = RunningTranscript(RunningTranscriptConfig(tail_ms=0))
    replacement.adopt(node)
    assert replacement.running() == ""


@pytest.mark.asyncio
async def test_running_transcript_cuts_an_utterance_at_commit_plus_tail():
    node = RunningTranscript(RunningTranscriptConfig(tail_ms=400))

    async def inputs():
        payload = {"text": "hello", "full_text": "hello", "is_first_output": False}
        yield TimedChunk("text", 0, 200, 0, payload, format="text")
        yield TimedChunk(AUDIO, 0, 200, 0, b"", format=PCM16)
        yield control_chunk(COMMIT)
        for t_ms in (200, 400):
            yield TimedChunk(AUDIO, t_ms, 200, 0, b"", format=PCM16)

    outputs = [chunk async for chunk in node.run(inputs())]
    assert [(c.payload["text"], c.payload["is_final"]) for c in outputs] == [
        ("hello", False),
        ("hello", True),
    ]


@pytest.mark.parametrize("profile", PROFILE_NAMES)
def test_profiles_expand_components_into_independent_session_nodes(profile):
    config = profile_config(profile)
    graph = config.session_graph()
    assert set(graph.nodes) == set(config.session_nodes)
    assert graph.publish == {
        name: tuple(formats) for name, formats in config.session_publish.items()
    }
    assert config.session_ports() == CASCADE_PORTS
    assert (
        StreamingCascadePipelineConfig.model_validate(
            config.model_dump()
        ).session_graph()
        == graph
    )
    for name, node in config.session_nodes.items():
        if node.kind is not None:
            assert graph.nodes[name].stages == (name,)
            assert config.stage_named(name).process == "cascade"
            assert config.stage_factory_kwargs(name)["name"] == name
        else:
            assert graph.nodes[name].stages == tuple(node.stages)
    assert graph.nodes["dialog"].bound("llm").feedback
    assert graph.nodes["dialog"].bound("tts").feedback


ALIGNED_PROFILE = PROFILES / "qwen3-asr-qwen3-tts-vivian" / "launch.yaml"
CHUNK_MS = 500
SPOKEN = "One two three four. Five six."


def speech(turn: int = 1) -> TimedChunk:
    return audio_chunk(np.zeros(24000 * CHUNK_MS // 1000), 24000, turn)


def spoken_progress(outputs: list[TimedChunk]) -> list[tuple[int, float]]:
    return [
        (o.payload["text_end"], o.payload["audio_ms"])
        for o in outputs
        if control_type(o) == REPLY_PROGRESS
    ]


def aligner(method: str, **options) -> object:
    return build({"kind": "alignment", "method": method, **options})


def aligned_profile(**dialog_controls) -> StreamingCascadePipelineConfig:
    data = yaml.safe_load(ALIGNED_PROFILE.read_text())
    for source in dialog_controls.get("progress_from", ()):
        data["session_nodes"]["dialog"]["inputs"].append(
            {"kind": "control", "source": source, "event": REPLY_PROGRESS}
        )
    return StreamingCascadePipelineConfig(
        model_path="m",
        session_nodes=data["session_nodes"],
        session_publish=data["session_publish"],
    )


@pytest.mark.parametrize("profile", PROFILE_NAMES)
def test_every_profile_takes_reply_progress_from_a_forced_aligner(profile):
    settings = profile_config(profile).component_graph()
    assert settings.node("aligner").component.method == "forced"
    assert [
        edge.source
        for edge in settings.node("dialog").controls
        if edge.event == REPLY_PROGRESS
    ] == ["aligner"]


@pytest.mark.parametrize("profile", PROFILE_NAMES)
def test_every_profile_publishes_the_aligners_reply_progress(profile):
    # note (luojiaxuan): the client's sglang.response.progress, which moves the caption with playback, comes only from a published node.
    assert SESSION_CONTROL in profile_config(profile).session_publish["aligner"]


def test_the_reference_profile_routes_reply_progress_through_the_aligner():
    settings = aligned_profile().component_graph()
    aligner_node = settings.node("aligner")
    assert aligner_node.component.method == "forced"
    assert aligner_node.component.alignment_url == "http://127.0.0.1:8010"
    assert aligner_node.produces == {SESSION_CONTROL}
    assert [
        edge.source
        for edge in settings.node("dialog").controls
        if edge.event == REPLY_PROGRESS
    ] == ["aligner"]
    with pytest.raises(ValueError, match="one source reports it"):
        aligned_profile(progress_from=["tts"])
    with pytest.raises(ValueError, match=r"reads \['aligner:reply.progress'\] as data"):
        declared(
            nodes={
                "realigner": {
                    "kind": "alignment",
                    "inputs": ["llm:text_delta", "tts:pcm_f32le", "aligner:session"],
                }
            }
        )
    with pytest.raises(ValueError, match="needs alignment_url"):
        COMPONENTS.parse({"kind": "alignment", "method": "forced"})
    with pytest.raises(ValueError, match="belongs to the forced method"):
        COMPONENTS.parse({"kind": "alignment", "alignment_url": "http://127.0.0.1:1"})
    with pytest.raises(ValueError, match="the asr method needs heard_asr_url"):
        COMPONENTS.parse({"kind": "alignment", "method": "asr"})
    with pytest.raises(ValueError, match="go together"):
        COMPONENTS.parse(
            {"kind": "alignment", "method": "asr", "heard_asr_url": HEARD_ASR_URL}
        )
    with pytest.raises(ValueError, match="belongs to the asr and forced methods"):
        COMPONENTS.parse(
            {
                "kind": "alignment",
                "heard_asr_url": HEARD_ASR_URL,
                "heard_asr_model": "asr",
            }
        )


@pytest.mark.asyncio
async def test_the_sentence_method_passes_the_synthesizer_sentence_ends_through():
    outputs = await drain(
        aligner("sentence"),
        [
            text_chunk(SPOKEN, 1),
            *(speech() for _ in range(4)),
            control_chunk(REPLY_PROGRESS, turn=1, text_end=19, audio_ms=2000),
            speech(),
            control_chunk(REPLY_PROGRESS, turn=1, text_end=29, audio_ms=2500),
            control_chunk(TURN_DONE, turn=1),
        ],
    )
    assert spoken_progress(outputs) == [(19, 2000), (29, 2500)]
    assert all(o.turn == 1 for o in outputs)


def worker_reply(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    assert (body["sample_rate_hz"], body["audio_format"]) == (24000, "pcm16")
    samples = len(base64.b64decode(body["audio"])) // 2
    words = [word.end() for word in re.finditer(r"\S+", body["text"])]
    # Note (Junnan Li): The fake worker ends each word at an equal share of the sentence's audio.
    return httpx.Response(
        200,
        json={
            "boundaries": [
                {
                    "text_end_chars": end,
                    "audio_end_samples": samples * (index + 1) // len(words),
                }
                for index, end in enumerate(words)
            ]
        },
    )


async def forced_aligner(handler, **options) -> object:
    node = aligner("forced", alignment_url="http://aligner:9000", **options)
    await node.client.aclose()
    node.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return node


@pytest.mark.asyncio
async def test_the_forced_method_reports_the_worker_word_ends_and_nothing_of_an_interrupted_sentence():
    texts = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://aligner:9000/align"
        texts.append(json.loads(request.content)["text"])
        return worker_reply(request)

    outputs = await drain(
        await forced_aligner(handler),
        [
            text_chunk(SPOKEN, 1),
            *(speech() for _ in range(4)),
            control_chunk(REPLY_PROGRESS, turn=1, text_end=19, audio_ms=2000),
            speech(),
            speech(),
            control_chunk(TURN_INTERRUPTED, turn=1),
        ],
    )
    assert texts == ["One two three four."]
    progress = spoken_progress(outputs)
    assert progress[0] == (19, 2000)
    # Note (Junnan Li): "Five six." never ends, so the worker never sees it.
    assert sorted(progress[1:]) == [(3, 500), (7, 1000), (13, 1500), (19, 2000)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        httpx.ReadTimeout("slow"),
        httpx.Response(500),
        httpx.Response(
            200, json={"boundaries": [{"text_end_chars": 99, "audio_end_samples": 0}]}
        ),
    ],
)
async def test_the_forced_method_keeps_the_sentence_end_when_the_worker_fails(failure):
    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(failure, Exception):
            raise failure
        else:
            return failure

    outputs = await drain(
        await forced_aligner(handler),
        [
            text_chunk(SPOKEN, 1),
            *(speech() for _ in range(4)),
            control_chunk(REPLY_PROGRESS, turn=1, text_end=19, audio_ms=2000),
        ],
    )
    assert spoken_progress(outputs) == [(19, 2000)]


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["Chinese", None])
async def test_the_forced_method_sends_the_language_of_the_reply_deltas(language):
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return worker_reply(request)

    await drain(
        await forced_aligner(handler),
        [
            text_chunk("One two three four.", 1),
            text_chunk(" Five six.", 1, language),
            *(speech() for _ in range(4)),
            control_chunk(REPLY_PROGRESS, turn=1, text_end=19, audio_ms=2000),
            *(speech() for _ in range(2)),
            control_chunk(REPLY_PROGRESS, turn=1, text_end=29, audio_ms=3000),
        ],
    )
    assert [body["text"] for body in bodies] == ["One two three four.", "Five six."]
    if language is None:
        assert all("language" not in body for body in bodies)
    else:
        assert [body["language"] for body in bodies] == [language, language]


HEARD_ASR_URL = "http://asr:8002/v1"


def heard_frames(request: httpx.Request) -> int:
    """The number of samples in the WAV file of a transcription request."""
    with wave.open(
        io.BytesIO(request.content[request.content.index(b"RIFF") :])
    ) as wav:
        return wav.getnframes()


async def hearing_aligner(method: str, handler, **options) -> object:
    node = aligner(
        method, heard_asr_url=HEARD_ASR_URL, heard_asr_model="asr", **options
    )
    await node.client.aclose()
    node.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return node


def transcribing(transcript: str, requests: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url == f"{HEARD_ASR_URL}/audio/transcriptions":
            requests.append(request)
            return httpx.Response(200, json={"text": transcript})
        else:
            return worker_reply(request)

    return handler


def interrupted_speech(
    *after: object, language: str | None = None, text: str = SPOKEN
) -> list[object]:
    """Reply audio interrupted after its first sentence and one second of its second."""
    return [
        text_chunk(text, 1, language),
        *(speech() for _ in range(4)),
        control_chunk(REPLY_PROGRESS, turn=1, text_end=19, audio_ms=2000),
        speech(),
        speech(),
        control_chunk(TURN_INTERRUPTED, turn=1),
        *after,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["English", None])
async def test_the_asr_method_keeps_the_prefix_of_the_interrupted_sentence_its_delivered_audio_says(
    language,
):
    requests = []
    outputs = await drain(
        await hearing_aligner("asr", transcribing("five", requests)),
        interrupted_speech(
            control_chunk(REPLY_DELIVERED, turn=1, audio_ms=2600), language=language
        ),
    )
    # Note (Junnan Li): "Five" ends at character 24 of SPOKEN; the cut is 600 ms into " Five six.".
    assert spoken_progress(outputs) == [(19, 2000), (24, 2600)]
    assert [heard_frames(request) for request in requests] == [600 * 24]
    assert b'name="model"\r\n\r\nasr' in requests[0].content
    assert (b'name="language"' in requests[0].content) == (language is not None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sentence, transcript, prefix, language",
    [
        (
            "Buy 350 long-term boxes.",
            "buy three hundred fifty longterm",
            "Buy 350 long-term",
            "English",
        ),
        (
            "Buy 350 long-term boxes.",
            "Buy 350 long-term",
            "Buy 350 long-term",
            "English",
        ),
        ("Buy 21 apples.", "Buy twenty", "Buy", "English"),
        ("Buy twenty apples.", "Buy 21", "Buy", "English"),
        ("Buy 21 apples.", "Buy twenty-one", "Buy 21", "English"),
        (
            "Buy 350 apples.",
            "Buy three hundred fifty apples",
            "Buy 350 apples.",
            "English",
        ),
        ("Wait 3.14 seconds.", "Wait three fourteen", "Wait", "English"),
        ("Wait 3.14 seconds.", "Wait three point one four", "Wait 3.14", "English"),
        ("Buy 1,000 apples.", "Buy 1,00", "Buy", "English"),
        ("Wait 3-5 days.", "Wait three five", "Wait", "English"),
        ("Wait 3-5 days.", "Wait three to five days", "Wait 3-5 days.", "English"),
        ("预算21元。", "预算二十一", "预算21", "English"),
        ("预算3.14元。", "预算三点一", "预算", "Chinese"),
        ("预算3.14元。", "预算三点一四元", "预算3.14元。", "Chinese"),
        ("Count 8 9 10.", "Count 八九十", "Count 8 9 10.", "English"),
    ],
)
async def test_asr_normalization_preserves_complete_original_units(
    sentence: str, transcript: str, prefix: str, language: str
) -> None:
    """Number expansion lost reply offsets or committed a number before its last word."""
    earlier_text = "One two three four."
    outputs = await drain(
        await hearing_aligner("asr", transcribing(transcript, [])),
        interrupted_speech(
            control_chunk(REPLY_DELIVERED, turn=1, audio_ms=2600),
            language=language,
            text=f"{earlier_text} {sentence}",
        ),
    )
    assert spoken_progress(outputs) == [
        (len(earlier_text), 2000),
        (len(earlier_text) + 1 + len(prefix), 2600),
    ]


@pytest.mark.asyncio
async def test_a_client_truncate_before_the_delivered_audio_moves_the_heard_cut_back():
    requests = []
    outputs = await drain(
        await hearing_aligner("asr", transcribing("five", requests)),
        interrupted_speech(
            control_chunk(REPLY_DELIVERED, turn=1, audio_ms=2800),
            control_chunk(TRUNCATE, t_ms=2300),
            control_chunk(TRUNCATE, t_ms=2500),
        ),
    )
    assert [heard_frames(request) for request in requests] == [800 * 24, 300 * 24]
    assert spoken_progress(outputs) == [(19, 2000), (24, 2800), (24, 2300)]


@pytest.mark.asyncio
async def test_a_client_truncate_of_another_turn_leaves_the_heard_cut():
    requests = []
    outputs = await drain(
        await hearing_aligner("asr", transcribing("five", requests)),
        interrupted_speech(
            control_chunk(REPLY_DELIVERED, turn=1, audio_ms=2800),
            control_chunk(TRUNCATE, turn=0, t_ms=2300),
        ),
    )
    assert [heard_frames(request) for request in requests] == [800 * 24]
    assert spoken_progress(outputs) == [(19, 2000), (24, 2800)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        httpx.ReadTimeout("slow"),
        httpx.Response(500),
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, json={"text": "seven eight"}),
    ],
)
async def test_the_interrupted_sentence_stays_unheard_when_its_transcription_fails_or_differs(
    failure,
):
    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(failure, Exception):
            raise failure
        else:
            return failure

    outputs = await drain(
        await hearing_aligner("asr", handler),
        interrupted_speech(control_chunk(REPLY_DELIVERED, turn=1, audio_ms=2600)),
    )
    assert spoken_progress(outputs) == [(19, 2000)]


@pytest.mark.asyncio
@pytest.mark.parametrize("cut_ms, heard", [(1200, False), (2600, True)])
async def test_the_forced_method_recognizes_only_a_cut_its_word_ends_do_not_cover(
    cut_ms, heard
):
    requests = []
    outputs = await drain(
        await hearing_aligner(
            "forced",
            transcribing("five", requests),
            alignment_url="http://aligner:9000",
        ),
        interrupted_speech(control_chunk(REPLY_DELIVERED, turn=1, audio_ms=cut_ms)),
    )
    progress = spoken_progress(outputs)
    assert sorted(progress[1:5]) == [(3, 500), (7, 1000), (13, 1500), (19, 2000)]
    if heard:
        assert progress[5:] == [(24, 2600)]
        assert len(requests) == 1
    else:
        assert progress[5:] == []
        assert requests == []


@pytest.mark.asyncio
async def test_the_dialog_keeps_the_heard_sentences_and_the_heard_prefix_of_the_cut_one():
    outputs = await drain(
        await hearing_aligner("asr", transcribing("five", [])),
        interrupted_speech(control_chunk(REPLY_DELIVERED, turn=1, audio_ms=2600)),
    )
    sentence_end, heard = [o for o in outputs if control_type(o) == REPLY_PROGRESS]
    dialog = build({"kind": "dialog"})
    await drain(
        dialog,
        [
            transcript_chunk("count", True),
            text_chunk(SPOKEN, 1),
            lambda: dialog.control(sentence_end),
            control_chunk(SPEECH_STARTED),
            control_chunk(REPLY_DELIVERED, turn=1, audio_ms=2600),
        ],
    )
    assert dialog.messages[2].content == "One two three four."
    await dialog.control(heard)
    assert dialog.messages[2].content == "One two three four. Five"
    assert not dialog.messages[2].uncertain


@pytest.mark.parametrize(
    "profile",
    [
        "qwen3-asr-qwen3-tts-vivian",
        "qwen3-asr-qwen3.8-27b-vivian",
        "qwen3-asr-qwen3-tts-base-clone",
    ],
)
def test_the_qwen3_asr_profiles_hear_cut_sentences_with_their_own_recognizer(profile):
    settings = profile_config(profile).component_graph()
    recognizer = settings.node("asr").component
    aligner_config = settings.node("aligner").component
    assert (aligner_config.heard_asr_url, aligner_config.heard_asr_model) == (
        recognizer.base_url,
        recognizer.model,
    )
    # Note (Junnan Li): Without the client's truncate the aligner hears up to the delivered audio, past what the dialog keeps, and its prefix is dropped.
    assert "input" in profile_config(profile).session_nodes["aligner"].sources


@pytest.mark.asyncio
async def test_a_truncate_cuts_at_the_word_the_aligner_reports(backends):
    settings, _, spoken, _ = backends
    tts = settings.component_graph().node("tts").component.model_dump()
    deltas = [text_chunk("Hello world. Second sentence here.", 1)]
    speech_outputs = await drain(
        build({**tts, "max_sentence_words": 24}),
        [*deltas, control_chunk(TURN_DONE, turn=1)],
    )
    assert spoken == ["Hello world.", "Second sentence here."]
    node = await forced_aligner(worker_reply)
    progress = [
        o
        for o in await drain(node, [*deltas, *speech_outputs])
        if control_type(o) == REPLY_PROGRESS
    ]
    dialog = build({"kind": "dialog"})
    await drain(
        dialog,
        [
            transcript_chunk("question", True),
            *deltas,
            *(lambda point=point: dialog.control(point) for point in progress),
            control_chunk(TURN_DONE, turn=1),
        ],
    )
    sentence_ms = 3 * 1000 / 24000
    # Note (Junnan Li): The fake worker ends "Second" one third of the way into the second sentence's audio.
    heard_ms = sentence_ms + sentence_ms / 3
    await drain(dialog, [control_chunk(TRUNCATE, t_ms=heard_ms)])
    assert dialog.messages[-1].content == "Hello world. Second"
    assert not dialog.messages[-1].uncertain


@pytest.mark.asyncio
async def test_openai_transcription_carries_the_language_the_service_reports():
    forms = []

    def answer(request: httpx.Request) -> httpx.Response:
        forms.append(request.content)
        return httpx.Response(
            200, json={"text": " 一二三 ", "language": "Chinese", "duration": 0.1}
        )

    asr = build({"kind": "openai_transcription", **HTTP_SERVICE})
    asr.client = httpx.AsyncClient(transport=httpx.MockTransport(answer))
    audio = TimedChunk(AUDIO, 0, 100, 0, QUIET, format=PCM16)
    outputs = await drain(asr, [audio, control_chunk(COMMIT)])
    assert [o.payload for o in outputs] == [
        {"text": "一二三", "is_final": True, "language": "Chinese"}
    ]
    assert b"verbose_json" in forms[0]


@pytest.mark.asyncio
async def test_openai_transcription_leaves_the_language_out_when_the_service_names_none():
    def answer(request: httpx.Request) -> httpx.Response:
        assert b'name="response_format"\r\n\r\njson\r\n' in request.content
        return httpx.Response(200, json={"text": "hello"})

    asr = build(
        {"kind": "openai_transcription", **HTTP_SERVICE, "response_format": "json"}
    )
    asr.client = httpx.AsyncClient(transport=httpx.MockTransport(answer))
    audio = TimedChunk(AUDIO, 0, 100, 0, QUIET, format=PCM16)
    outputs = await drain(asr, [audio, control_chunk(COMMIT)])
    assert [o.payload for o in outputs] == [{"text": "hello", "is_final": True}]


@pytest.mark.asyncio
async def test_the_dialog_keeps_the_language_of_the_last_answered_turn():
    dialog = build({"kind": "dialog"})
    outputs = await drain(
        dialog,
        [
            transcript_chunk("数到三。", True, "Chinese"),
            text_chunk("1,2,3.", 1),
            control_chunk(TURN_DONE, turn=1),
            control_chunk(COMMIT),
            transcript_chunk("Again.", True),
            text_chunk("1,2,3.", 2),
            control_chunk(TURN_DONE, turn=2),
            control_chunk(COMMIT),
            transcript_chunk("Now in English.", True, "English"),
        ],
    )
    languages = [o.payload.get("language") for o in outputs if o.format == MESSAGES]
    assert languages == ["Chinese", "Chinese", "English"]
    replacement = build({"kind": "dialog"})
    replacement.adopt(dialog)
    assert replacement.language == "English"


@pytest.mark.asyncio
async def test_chat_deltas_carry_the_language_of_their_messages():
    generator = build({"kind": "openai_chat", **HTTP_SERVICE})

    async def complete(messages):
        for content in ("1, ", "2."):
            yield ChatDelta(content=content)

    generator.complete = complete
    outputs = await drain(
        generator,
        [
            messages_chunk([ChatMessage(role="user", content="数")], 1, "Chinese"),
            messages_chunk([ChatMessage(role="user", content="count")], 2),
        ],
    )
    deltas = [(o.turn, o.payload) for o in outputs if o.format == TEXT_DELTA]
    assert deltas == [
        (1, {"text": "1, ", "language": "Chinese"}),
        (1, {"text": "2.", "language": "Chinese"}),
        (2, {"text": "1, "}),
        (2, {"text": "2."}),
    ]


async def spoken_requests(
    spec: dict, deltas: list[TimedChunk]
) -> list[tuple[str, str]]:
    """The language tag and the input of each sentence the synthesizer speaks for the deltas of turn 1."""
    bodies = []

    def answer(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200, content=b"\x00\x00" * 4, headers={"X-Sample-Rate": "24000"}
        )

    tts = build({"kind": "openai_speech", **HTTP_SERVICE, **spec})
    tts.client = httpx.AsyncClient(transport=httpx.MockTransport(answer))
    await drain(tts, [*deltas, control_chunk(TURN_DONE, turn=1)])
    return [(body["language"], body["input"]) for body in bodies]


async def spoken_languages(spec: dict, deltas: list[TimedChunk]) -> list[str]:
    return [language for language, _ in await spoken_requests(spec, deltas)]


@pytest.mark.asyncio
async def test_openai_speech_keeps_the_digits_of_a_chinese_script_sentence_tagged_english():
    deltas = [
        text_chunk("This year is 2024. ", 1, "English"),
        text_chunk("今年是2024年。", 1, "English"),
    ]
    assert await spoken_requests({"language": "auto"}, deltas) == [
        ("English", "This year is twenty twenty four."),
        ("English", "今年是2024年。"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "language, user_language, tags",
    [
        ("auto", "Chinese", ["Chinese", "Chinese"]),
        ("auto", "Japanese", ["Japanese", "Japanese"]),
        ("auto", None, ["English", "Chinese"]),
        ("English", "Chinese", ["English", "Chinese"]),
    ],
)
async def test_openai_speech_auto_language_follows_the_user(
    language, user_language, tags
):
    deltas = [
        text_chunk("1, 2, 3. ", 1, user_language),
        text_chunk("你好。", 1, user_language),
    ]
    assert await spoken_languages({"language": language}, deltas) == tags


def test_openai_speech_keeps_the_user_language_across_a_replacement():
    spec = {"kind": "openai_speech", **HTTP_SERVICE, "language": "auto"}
    first = build(spec)
    first.session_language = "Chinese"
    replacement = build({**spec, "voice": "b"})
    replacement.adopt(first)
    assert replacement.sentence_language("1, 2, 3.") == "Chinese"


@pytest.mark.parametrize("profile", PROFILE_NAMES)
def test_profiles_speak_the_user_language(profile):
    config = profile_config(profile)
    assert config.component_graph().node("tts").component.language == "auto"


@pytest.mark.parametrize(
    ("flags", "admitted"), [([], 16), (["--max_sessions", "24"], 24)]
)
def test_node_stages_admit_max_sessions_and_refuse_the_next(flags, admitted):
    manager = ConfigManager(profile_config("qwen3-asr-qwen3.8-27b-vivian"))
    config = manager.merge_config(manager.parse_extra_args(flags))
    stages = {}
    for name in ("vad", "dialog"):
        stage = config.stage_named(name)
        stages[name] = import_string(stage.factory_path)(
            **resolve_stage_factory_args(stage, config)
        )
    assert stages["vad"].max_open_sessions == admitted
    dialog = stages["dialog"]
    sessions = [SessionIdentity(f"session-{index}") for index in range(admitted)]
    try:
        for session in sessions:
            dialog.open_session(session, OmniRequest(None))
        with pytest.raises(QueueFullError):
            dialog.open_session(SessionIdentity("one-more"), OmniRequest(None))
    finally:
        for session in sessions:
            dialog.close_session(session, dialog.open_sessions[session])
