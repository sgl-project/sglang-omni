# SPDX-License-Identifier: Apache-2.0
"""The thinker session keeps one native turn per append and refills what an interrupted reply's listener heard."""

import queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.qwen3_omni import thinker_session
from sglang_omni.models.qwen3_omni.session_input import QwenInputHooks
from sglang_omni.models.qwen3_omni.thinker_session import (
    CODEC_FRAME_MS,
    PREFILL_TEXT_ROWS,
    TRUNCATE_FORMAT,
    QwenThinkerSessionAdapter,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.proto.streams import (
    COMMIT,
    CONTROL,
    INTERRUPT,
    PCM16,
    REPLY_DELIVERED,
    REPLY_TRUNCATED,
    TRUNCATE,
    TURN_STARTED,
)
from sglang_omni.scheduling.session import SessionContext
from tests.unit_test.fixtures.qwen_fakes import make_dialog
from tests.unit_test.fixtures.session_outputs import control_chunk

IDENTITY = SessionIdentity("session")
END = 99
NEWLINE = 10
SYSTEM_PREFIX = [1, 2]
PROMPT_TEXT = "<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\n<audio><|im_end|>\n<|im_start|>assistant\n"
DELIVERED_TWO_FRAMES = control_chunk(
    REPLY_DELIVERED, turn=1, audio_ms=2.0 * CODEC_FRAME_MS
)


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> QwenThinkerSessionAdapter:
    tokenizer = Mock()
    tokenizer.convert_tokens_to_ids.return_value = END
    tokenizer.encode.side_effect = lambda text, add_special_tokens: (
        [NEWLINE] if text == "\n" else SYSTEM_PREFIX
    )

    def build_request(
        pipeline_state,
        *,
        params,
        tokenizer,
        vocab_size,
        request_id,
        capture_prompt_hidden_states,
    ):
        return SimpleNamespace(
            req=SimpleNamespace(
                origin_input_ids=pipeline_state.prompt["input_ids"].tolist()
            ),
            output_ids=[],
            stage_payload=None,
            finish_reason=None,
        )

    monkeypatch.setattr(thinker_session, "build_sglang_thinker_request", build_request)
    monkeypatch.setattr(
        thinker_session,
        "compute_mrope_positions",
        lambda ids, inputs, config: (torch.zeros(3, len(ids), dtype=torch.long), 0),
    )
    config = SimpleNamespace(image_token_id=100, video_token_id=101, audio_token_id=102)
    result = QwenThinkerSessionAdapter(tokenizer, 200, config)
    result.outbox = queue.Queue()
    result.open(IDENTITY, OmniRequest(None))
    return result


def unit_chunk(turn: int, carried_turns: tuple[int, ...] = ()) -> TimedChunk:
    return TimedChunk(
        CONTROL,
        0,
        0,
        0,
        {"carried_turns": list(carried_turns)},
        format="qwen.unit",
        turn=turn,
    )


def payload(
    turn: int, prompt_ids: list[int], audio_rows: list[float] | None = None
) -> StagePayload:
    model_inputs = (
        {"audio_embeds": torch.tensor([[row] for row in audio_rows])}
        if audio_rows is not None
        else {}
    )
    return StagePayload(
        f"unit-{turn}",
        OmniRequest(None, params={"max_new_tokens": 8}),
        {
            "session": {"turn": turn},
            "prompt": {
                "input_ids": torch.tensor(SYSTEM_PREFIX + prompt_ids),
                "prompt_text": PROMPT_TEXT,
            },
            "thinker_inputs": {"model_inputs": model_inputs},
        },
    )


def finish(adapter: QwenThinkerSessionAdapter, request_data, output_ids: list[int]):
    request_data.output_ids = output_ids
    adapter.result(IDENTITY, request_data)


def abort(adapter: QwenThinkerSessionAdapter, output_ids: list[int]) -> None:
    unit = SimpleNamespace(session_request=SimpleNamespace(output_ids=output_ids))
    adapter.on_abort(IDENTITY, unit)


def truncate(turn: int, **fields: object) -> TimedChunk:
    return TimedChunk(
        CONTROL, 0, 0, 0, {"turn": turn, **fields}, format=TRUNCATE_FORMAT
    )


def interrupted_dialog() -> tuple[QwenInputHooks, list[TimedChunk], SessionContext]:
    """A dialog whose first turn's reply was interrupted after it started; what it tells the thinker from here on are truncates."""
    hooks, outputs, context = make_dialog(IDENTITY)
    stage_payload = StagePayload("in", OmniRequest(None), {})
    hooks.append(
        TimedChunk("audio", 0, 20, 0, b"\x01\x00" * 320, format=PCM16),
        stage_payload,
        context,
    )
    hooks.append(control_chunk(COMMIT), stage_payload, context)
    hooks.append(control_chunk(TURN_STARTED, turn=1), stage_payload, context)
    hooks.control(IDENTITY, control_chunk(INTERRUPT))
    return hooks, outputs, context


def test_each_turn_is_one_native_append_after_the_last_reply(
    adapter: QwenThinkerSessionAdapter,
) -> None:
    first = adapter.build(IDENTITY, unit_chunk(1), payload(1, [5, 6]))
    assert first.req.origin_input_ids == SYSTEM_PREFIX + [5, 6]
    assert first.enforce_request_limits is True
    finish(adapter, first, [50, END])
    second = adapter.build(IDENTITY, unit_chunk(2), payload(2, [7]))
    assert second.req.origin_input_ids == [NEWLINE, 7]
    finish(adapter, second, [60])
    third = adapter.build(IDENTITY, unit_chunk(3), payload(3, [8]))
    assert third.req.origin_input_ids == [END, NEWLINE, 8]
    assert (
        adapter.prepare_unit(IDENTITY, unit_chunk(3), payload(3, [8])).reset_history
        is False
    )


def test_an_interrupted_turn_is_refilled_with_what_was_heard(
    adapter: QwenThinkerSessionAdapter,
) -> None:
    first = adapter.build(IDENTITY, unit_chunk(1), payload(1, [5]))
    finish(adapter, first, [50, END])
    adapter.build(IDENTITY, unit_chunk(2), payload(2, [7], audio_rows=[1.0, 2.0]))
    abort(adapter, [61, 62, 63, 64, 65, 66])
    heard_ms = 2 * CODEC_FRAME_MS
    adapter.control(IDENTITY, truncate(2, t_ms=heard_ms))
    third = adapter.build(IDENTITY, unit_chunk(3), payload(3, [8], audio_rows=[3.0]))
    heard = PREFILL_TEXT_ROWS + 2
    assert third.req.origin_input_ids == [
        NEWLINE,
        7,
        *[61, 62, 63, 64, 65, 66][:heard],
        END,
        NEWLINE,
        8,
    ]
    request = SimpleNamespace(origin_input_ids=third.req.origin_input_ids)
    third.req = request
    adapter.bind_request(IDENTITY, third, SimpleNamespace())
    assert request.omni_model_inputs["audio_embeds"].tolist() == [[1.0], [2.0], [3.0]]
    finish(adapter, third, [70, END])
    fourth = adapter.build(IDENTITY, unit_chunk(4), payload(4, [9]))
    assert fourth.req.origin_input_ids == [NEWLINE, 9]


def test_without_truncate_the_whole_generated_reply_stays(
    adapter: QwenThinkerSessionAdapter,
) -> None:
    adapter.build(IDENTITY, unit_chunk(1), payload(1, [5]))
    abort(adapter, [51, 52])
    second = adapter.build(IDENTITY, unit_chunk(2), payload(2, [7]))
    assert second.req.origin_input_ids == SYSTEM_PREFIX + [5, 51, 52, END, NEWLINE, 7]


@pytest.mark.parametrize(
    ("reports", "heard"),
    [
        pytest.param(
            [DELIVERED_TWO_FRAMES], PREFILL_TEXT_ROWS + 2, id="no_client_truncate"
        ),
        pytest.param(
            [DELIVERED_TWO_FRAMES, control_chunk(TRUNCATE, t_ms=CODEC_FRAME_MS)],
            PREFILL_TEXT_ROWS + 1,
            id="client_below_delivered",
        ),
        pytest.param(
            [DELIVERED_TWO_FRAMES, control_chunk(TRUNCATE, t_ms=4 * CODEC_FRAME_MS)],
            PREFILL_TEXT_ROWS + 2,
            id="client_above_delivered",
        ),
        pytest.param(
            [control_chunk(TRUNCATE, t_ms=4 * CODEC_FRAME_MS), DELIVERED_TWO_FRAMES],
            PREFILL_TEXT_ROWS + 2,
            id="client_first",
        ),
    ],
)
@pytest.mark.parametrize(
    "is_reply_finished",
    [pytest.param(False, id="aborted"), pytest.param(True, id="finished")],
)
def test_an_interrupted_reply_keeps_at_most_the_smaller_of_delivered_and_heard(
    adapter: QwenThinkerSessionAdapter,
    reports: list[TimedChunk],
    heard: int,
    is_reply_finished: bool,
) -> None:
    """The graph delivered two codec frames of the reply when the interrupt came: whether the thinker had finished the reply or was aborted, the history keeps those without a client truncate, fewer when the client heard less, and never more, whichever report reaches the dialog first."""
    hooks, dialog_outputs, context = interrupted_dialog()
    first = adapter.build(IDENTITY, unit_chunk(1), payload(1, [5]))
    generated = [51, 52, 53, 54, 55, 56, 57, 58]
    if is_reply_finished:
        finish(adapter, first, [*generated, END])
    else:
        abort(adapter, generated)
    for report in reports:
        hooks.append(report, StagePayload("in", OmniRequest(None), {}), context)
    for output in dialog_outputs:
        if output.format == TRUNCATE_FORMAT:
            adapter.control(IDENTITY, output)
        else:
            pass
    second = adapter.build(IDENTITY, unit_chunk(2), payload(2, [7]))
    assert second.req.origin_input_ids == SYSTEM_PREFIX + [
        5,
        *generated[:heard],
        END,
        NEWLINE,
        7,
    ]


def test_truncating_past_the_end_of_a_finished_reply_keeps_its_kv(
    adapter: QwenThinkerSessionAdapter,
) -> None:
    first = adapter.build(IDENTITY, unit_chunk(1), payload(1, [5]))
    finish(adapter, first, [51, 52, END])
    adapter.control(IDENTITY, truncate(1, t_ms=10 * CODEC_FRAME_MS))
    preparation = adapter.prepare_unit(IDENTITY, unit_chunk(2), payload(2, [7]))
    assert preparation.reset_history is False
    second = adapter.build(IDENTITY, unit_chunk(2), payload(2, [7]))
    assert second.req.origin_input_ids == [NEWLINE, 7]


def test_a_truncate_of_an_earlier_reply_keeps_the_history(
    adapter: QwenThinkerSessionAdapter,
) -> None:
    first = adapter.build(IDENTITY, unit_chunk(1), payload(1, [5]))
    finish(adapter, first, [51, 52, 53, END])
    second = adapter.build(IDENTITY, unit_chunk(2), payload(2, [7]))
    finish(adapter, second, [71, 72, 73, END])
    adapter.control(IDENTITY, truncate(1, t_ms=CODEC_FRAME_MS))
    preparation = adapter.prepare_unit(IDENTITY, unit_chunk(3), payload(3, [8]))
    assert preparation.reset_history is False
    third = adapter.build(IDENTITY, unit_chunk(3), payload(3, [8]))
    assert third.req.origin_input_ids == [NEWLINE, 8]


def test_truncating_a_finished_reply_rebuilds_the_native_session(
    adapter: QwenThinkerSessionAdapter,
) -> None:
    first = adapter.build(IDENTITY, unit_chunk(1), payload(1, [5], audio_rows=[1.0]))
    finish(adapter, first, [51, 52, 53, 54, 55, END])
    adapter.control(IDENTITY, truncate(1, t_ms=CODEC_FRAME_MS))
    preparation = adapter.prepare_unit(IDENTITY, unit_chunk(2), payload(2, [7]))
    assert preparation.reset_history is True
    second = adapter.build(IDENTITY, unit_chunk(2), payload(2, [7], audio_rows=[2.0]))
    heard = PREFILL_TEXT_ROWS + 1
    assert second.req.origin_input_ids == SYSTEM_PREFIX + [
        5,
        *[51, 52, 53, 54, 55][:heard],
        END,
        NEWLINE,
        7,
    ]
    request = SimpleNamespace(origin_input_ids=second.req.origin_input_ids)
    second.req = request
    adapter.bind_request(IDENTITY, second, SimpleNamespace())
    assert request.omni_model_inputs["audio_embeds"].tolist() == [[1.0], [2.0]]
    finish(adapter, second, [70])
    assert (
        adapter.prepare_unit(IDENTITY, unit_chunk(3), payload(3, [8])).reset_history
        is False
    )
    third = adapter.build(IDENTITY, unit_chunk(3), payload(3, [8]))
    assert third.req.origin_input_ids == [END, NEWLINE, 8]


def test_a_reply_cut_at_the_thinker_token_budget_tells_the_talker(
    adapter: QwenThinkerSessionAdapter,
) -> None:
    """The talker speaks the text it got and stops, so without this the turn ends as a natural stop."""
    request_data = adapter.build(IDENTITY, unit_chunk(1), payload(1, [5]))
    request_data.finish_reason = "length"
    finish(adapter, request_data, [50, 51])
    message = adapter.outbox.get_nowait()
    assert (message.request_id, message.target, message.type) == (
        "unit-1",
        "talker_ar",
        "stream",
    )
    assert message.metadata == {REPLY_TRUNCATED: True}
    assert message.data.numel() == 0
    assert adapter.outbox.empty()


def test_a_reply_that_ended_on_its_own_tells_the_talker_nothing(
    adapter: QwenThinkerSessionAdapter,
) -> None:
    request_data = adapter.build(IDENTITY, unit_chunk(1), payload(1, [5]))
    request_data.finish_reason = "stop"
    finish(adapter, request_data, [50, END])
    assert adapter.outbox.empty()


def test_system_prefix_must_match_the_tokenized_prompt(
    adapter: QwenThinkerSessionAdapter,
) -> None:
    finish(adapter, adapter.build(IDENTITY, unit_chunk(1), payload(1, [5])), [END])
    bad = payload(2, [7])
    bad.data["prompt"]["input_ids"] = torch.tensor([3, 4, 7])
    with pytest.raises(ValueError, match="system prefix"):
        adapter.build(IDENTITY, unit_chunk(2), bad)


def test_a_second_truncate_of_a_finished_reply_only_cuts_further(
    adapter: QwenThinkerSessionAdapter,
) -> None:
    first = adapter.build(IDENTITY, unit_chunk(1), payload(1, [5]))
    finish(adapter, first, [51, 52, 53, 54, 55, 56, 57, END])
    adapter.control(IDENTITY, truncate(1, t_ms=2 * CODEC_FRAME_MS))
    adapter.control(IDENTITY, truncate(1, t_ms=3 * CODEC_FRAME_MS))
    adapter.control(IDENTITY, truncate(1, t_ms=0))
    second = adapter.build(IDENTITY, unit_chunk(2), payload(2, [7]))
    heard = PREFILL_TEXT_ROWS
    assert second.req.origin_input_ids == SYSTEM_PREFIX + [
        5,
        *[51, 52, 53, 54, 55, 56, 57][:heard],
        END,
        NEWLINE,
        7,
    ]


def test_a_later_truncate_of_an_interrupted_reply_only_cuts_further(
    adapter: QwenThinkerSessionAdapter,
) -> None:
    adapter.build(IDENTITY, unit_chunk(1), payload(1, [5]))
    abort(adapter, [51, 52, 53, 54, 55, 56])
    adapter.control(IDENTITY, truncate(1, t_ms=CODEC_FRAME_MS))
    adapter.control(IDENTITY, truncate(1, t_ms=2 * CODEC_FRAME_MS))
    second = adapter.build(IDENTITY, unit_chunk(2), payload(2, [7]))
    heard = PREFILL_TEXT_ROWS + 1
    assert second.req.origin_input_ids == SYSTEM_PREFIX + [
        5,
        *[51, 52, 53, 54, 55, 56][:heard],
        END,
        NEWLINE,
        7,
    ]


def test_an_interrupted_turn_the_dialog_asks_again_leaves_only_what_preceded_it(
    adapter: QwenThinkerSessionAdapter,
) -> None:
    """The earlier interrupted turn refilled in front of it stays; the re-asked turn's own prompt and reply go."""
    adapter.build(IDENTITY, unit_chunk(1), payload(1, [5]))
    abort(adapter, [51])
    adapter.build(IDENTITY, unit_chunk(2), payload(2, [7], audio_rows=[1.0]))
    abort(adapter, [61, 62])
    third = adapter.build(
        IDENTITY, unit_chunk(3, (2,)), payload(3, [8], audio_rows=[3.0])
    )
    assert third.req.origin_input_ids == SYSTEM_PREFIX + [5, 51, END, NEWLINE, 8]
    request = SimpleNamespace(origin_input_ids=third.req.origin_input_ids)
    third.req = request
    adapter.bind_request(IDENTITY, third, SimpleNamespace())
    assert request.omni_model_inputs["audio_embeds"].tolist() == [[3.0]]


def test_a_committed_turn_the_dialog_asks_again_leaves_the_history(
    adapter: QwenThinkerSessionAdapter,
) -> None:
    """Its reply reached no client, so the turn is cut out and the native session rebuilt."""
    first = adapter.build(IDENTITY, unit_chunk(1), payload(1, [5]))
    finish(adapter, first, [50, END])
    second = adapter.build(IDENTITY, unit_chunk(2), payload(2, [7], audio_rows=[1.0]))
    finish(adapter, second, [60, END])
    carried = unit_chunk(3, (2,))
    assert adapter.prepare_unit(IDENTITY, carried, payload(3, [8])).reset_history
    third = adapter.build(IDENTITY, carried, payload(3, [8], audio_rows=[3.0]))
    assert third.req.origin_input_ids == SYSTEM_PREFIX + [5, 50, END, NEWLINE, 8]
    request = SimpleNamespace(origin_input_ids=third.req.origin_input_ids)
    third.req = request
    adapter.bind_request(IDENTITY, third, SimpleNamespace())
    assert request.omni_model_inputs["audio_embeds"].tolist() == [[3.0]]


def test_an_aborted_rebuild_keeps_the_history_for_the_next_unit(
    adapter: QwenThinkerSessionAdapter,
) -> None:
    first = adapter.build(IDENTITY, unit_chunk(1), payload(1, [5]))
    finish(adapter, first, [51, 52, 53, 54, 55, END])
    adapter.control(IDENTITY, truncate(1, t_ms=0))
    assert adapter.prepare_unit(IDENTITY, unit_chunk(2), payload(2, [7])).reset_history
    adapter.build(IDENTITY, unit_chunk(2), payload(2, [7]))
    abort(adapter, [71])
    assert adapter.prepare_unit(IDENTITY, unit_chunk(3), payload(3, [8])).reset_history
    third = adapter.build(IDENTITY, unit_chunk(3), payload(3, [8]))
    assert third.req.origin_input_ids == SYSTEM_PREFIX + [
        5,
        *[51, 52, 53, 54, 55][:PREFILL_TEXT_ROWS],
        END,
        NEWLINE,
        7,
        71,
        END,
        NEWLINE,
        8,
    ]


# What a third unit prefills once turn 1's reply, refilled by turn 2's unit, was cut to one codec frame and turn 2 replied 60.
CUT_REFILLED_REPLY_HISTORY = SYSTEM_PREFIX + [
    5,
    51,
    52,
    53,
    54,
    END,
    NEWLINE,
    7,
    60,
    END,
    NEWLINE,
    8,
]


@pytest.mark.parametrize(
    ("heard_ms", "is_next_reply_finished", "reset_history", "third_ids"),
    [
        pytest.param(
            CODEC_FRAME_MS, True, True, CUT_REFILLED_REPLY_HISTORY, id="finished"
        ),
        pytest.param(
            CODEC_FRAME_MS, False, False, CUT_REFILLED_REPLY_HISTORY, id="aborted"
        ),
        pytest.param(
            10 * CODEC_FRAME_MS, True, False, [NEWLINE, 8], id="past-the-reply"
        ),
    ],
)
def test_a_truncate_of_an_aborted_reply_the_unit_in_flight_refills_still_cuts_it(
    adapter: QwenThinkerSessionAdapter,
    heard_ms: float,
    is_next_reply_finished: bool,
    reset_history: bool,
    third_ids: list[int],
) -> None:
    """The next turn's unit went out with the whole interrupted reply in front of its prompt before the truncate of that reply came; the cut used to be lost, so the full reply stayed in the history once the unit finished and was refilled again once it was aborted. A truncate past the refilled reply changes nothing and costs no rebuild."""
    adapter.build(IDENTITY, unit_chunk(1), payload(1, [5]))
    abort(adapter, [51, 52, 53, 54, 55, 56])
    second = adapter.build(IDENTITY, unit_chunk(2), payload(2, [7]))
    adapter.control(IDENTITY, truncate(1, t_ms=heard_ms))
    if is_next_reply_finished:
        finish(adapter, second, [60, END])
    else:
        abort(adapter, [60])
    assert (
        adapter.prepare_unit(IDENTITY, unit_chunk(3), payload(3, [8])).reset_history
        is reset_history
    )
    third = adapter.build(IDENTITY, unit_chunk(3), payload(3, [8]))
    assert third.req.origin_input_ids == third_ids


def test_a_cut_refilled_reply_stays_cut_when_the_dialog_asks_the_next_turn_again(
    adapter: QwenThinkerSessionAdapter,
) -> None:
    """The truncate of the interrupted reply came after the next unit had refilled it; that unit is interrupted too and its utterance asked again, so only the cut reply precedes the re-asked prompt."""
    adapter.build(IDENTITY, unit_chunk(1), payload(1, [5]))
    abort(adapter, [51, 52, 53, 54, 55, 56])
    adapter.build(IDENTITY, unit_chunk(2), payload(2, [7]))
    adapter.control(IDENTITY, truncate(1, t_ms=CODEC_FRAME_MS))
    abort(adapter, [60])
    third = adapter.build(IDENTITY, unit_chunk(3, (2,)), payload(3, [8]))
    assert third.req.origin_input_ids == SYSTEM_PREFIX + [
        5,
        51,
        52,
        53,
        54,
        END,
        NEWLINE,
        8,
    ]


def test_a_truncate_between_a_units_preparation_and_its_build_rebuilds_the_next_unit(
    adapter: QwenThinkerSessionAdapter,
) -> None:
    """The scheduler prepares every append of a batch before it builds any, so a truncate in the same batch set the rebuild after the preparation had kept the KV, and the build sent the whole history onto that KV."""
    first = adapter.build(IDENTITY, unit_chunk(1), payload(1, [5]))
    finish(adapter, first, [51, 52, 53, 54, 55, END])
    preparation = adapter.prepare_unit(IDENTITY, unit_chunk(2), payload(2, [7]))
    assert preparation.reset_history is False
    adapter.control(IDENTITY, truncate(1, t_ms=CODEC_FRAME_MS))
    second = adapter.build(IDENTITY, unit_chunk(2), payload(2, [7]))
    assert second.req.origin_input_ids == [NEWLINE, 7]
    finish(adapter, second, [60, END])
    assert adapter.prepare_unit(IDENTITY, unit_chunk(3), payload(3, [8])).reset_history
    third = adapter.build(IDENTITY, unit_chunk(3), payload(3, [8]))
    heard = PREFILL_TEXT_ROWS + 1
    assert third.req.origin_input_ids == SYSTEM_PREFIX + [
        5,
        *[51, 52, 53, 54, 55][:heard],
        END,
        NEWLINE,
        7,
        60,
        END,
        NEWLINE,
        8,
    ]


@pytest.mark.parametrize(
    "is_next_reply_finished",
    [pytest.param(True, id="finished"), pytest.param(False, id="aborted")],
)
def test_a_truncate_of_a_committed_reply_while_the_next_unit_is_in_flight_outlives_that_unit(
    adapter: QwenThinkerSessionAdapter,
    is_next_reply_finished: bool,
) -> None:
    """The next unit was appended to the native session holding the full reply before the truncate came; its result used to clear the rebuild, so the KV kept the full reply the history had cut, and the next prompt got a second newline after the cut reply."""
    first = adapter.build(IDENTITY, unit_chunk(1), payload(1, [5]))
    finish(adapter, first, [51, 52, 53, 54, 55, END])
    second = adapter.build(IDENTITY, unit_chunk(2), payload(2, [7]))
    adapter.control(IDENTITY, truncate(1, t_ms=CODEC_FRAME_MS))
    if is_next_reply_finished:
        finish(adapter, second, [60, END])
    else:
        abort(adapter, [60])
    assert adapter.prepare_unit(IDENTITY, unit_chunk(3), payload(3, [8])).reset_history
    third = adapter.build(IDENTITY, unit_chunk(3), payload(3, [8]))
    heard = PREFILL_TEXT_ROWS + 1
    assert third.req.origin_input_ids == SYSTEM_PREFIX + [
        5,
        *[51, 52, 53, 54, 55][:heard],
        END,
        NEWLINE,
        7,
        60,
        END,
        NEWLINE,
        8,
    ]
