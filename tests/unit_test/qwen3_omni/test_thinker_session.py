# SPDX-License-Identifier: Apache-2.0
"""The thinker session keeps one native turn per append and refills what an interrupted reply's listener heard."""

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
        )

    monkeypatch.setattr(thinker_session, "build_sglang_thinker_request", build_request)
    monkeypatch.setattr(
        thinker_session,
        "compute_mrope_positions",
        lambda ids, inputs, config: (torch.zeros(3, len(ids), dtype=torch.long), 0),
    )
    config = SimpleNamespace(image_token_id=100, video_token_id=101, audio_token_id=102)
    result = QwenThinkerSessionAdapter(tokenizer, 200, config)
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
