# SPDX-License-Identifier: Apache-2.0
"""The floor pauses a reply under speech, resumes it when that speech was not addressed to the assistant, and cancels it like today otherwise."""

import math

import pytest

from sglang_omni.audio.floor import (
    AddresseeQuestion,
    Answer,
    Ask,
    Cancel,
    Floor,
    Pause,
    Resume,
)


def paused_floor() -> Floor[str]:
    floor: Floor[str] = Floor("pause", 0.0)
    assert floor.reply_started(1) == []
    assert floor.speech_started() == [Pause(1)]
    return floor


def test_speech_over_a_reply_pauses_it_and_a_not_addressed_verdict_resumes_it():
    floor = paused_floor()
    assert floor.speech_started() == []
    assert floor.speech_stopped() == []
    assert floor.heard("mm") == [Ask(0, 1, ("mm",))]
    assert floor.decided(0, -2.0) == [Resume(1)]
    assert floor.decided(0, -2.0) == []
    assert floor.held() == () and not floor.is_paused
    assert floor.heard("next") == [Answer(("next",))]


def test_a_second_overlap_while_deciding_waits_for_a_verdict_on_both():
    floor = paused_floor()
    floor.speech_stopped()
    assert floor.heard("a") == [Ask(0, 1, ("a",))]
    assert floor.speech_started() == []
    assert floor.decided(0, -2.0) == []
    assert floor.speech_stopped() == []
    assert floor.heard("b") == [Ask(1, 1, ("a", "b"))]
    assert floor.decided(0, 5.0) == []
    assert floor.decided(1, 5.0) == [Cancel(1), Answer(("a", "b"))]
    assert floor.reply is None and not floor.is_paused


def test_a_verdict_below_the_margin_waits_for_the_user_to_stop_speaking():
    floor: Floor[str] = Floor("pause", 1.5)
    floor.reply_started(1)
    floor.speech_started()
    floor.speech_stopped()
    floor.heard("a")
    floor.speech_started()
    assert floor.decided(0, 1.0) == []
    floor.speech_stopped()
    assert floor.heard("b", is_empty=False) == [Ask(1, 1, ("a", "b"))]
    assert floor.decided(1, 1.49) == [Resume(1)]


@pytest.mark.parametrize("verdict", [None, 0.0, 3.0])
def test_no_evidence_or_an_addressed_verdict_cancels_and_answers_what_was_heard(
    verdict,
):
    floor = paused_floor()
    floor.speech_stopped()
    floor.heard("stop")
    assert floor.decided(0, verdict) == [Cancel(1), Answer(("stop",))]
    assert floor.reply_ended(1) == []


def test_the_interrupt_policy_cancels_at_once_and_never_pauses():
    floor: Floor[str] = Floor("interrupt", 0.0)
    assert floor.reply_started(1) == []
    assert floor.speech_started() == [Cancel(1)]
    assert floor.reply is None and not floor.is_paused
    assert floor.heard("what") == [Answer(("what",))]
    assert floor.reply_started(2) == [Cancel(2)]
    floor.speech_stopped()
    assert floor.reply_started(3) == []


def test_a_client_cancel_while_paused_answers_what_was_heard_and_later_content_is_plain():
    floor = paused_floor()
    floor.speech_stopped()
    floor.heard("a")
    floor.speech_started()
    assert floor.reply_ended(1) == [Answer(("a",))]
    assert floor.reply_ended(1) == []
    assert floor.decided(0, 5.0) == []
    floor.speech_stopped()
    assert floor.heard("b") == [Answer(("b",))]


def test_a_reply_ending_while_paused_with_nothing_heard_answers_nothing():
    floor = paused_floor()
    assert floor.reply_ended(1) == []
    floor.speech_stopped()
    assert floor.heard("late") == [Answer(("late",))]


def test_content_without_words_resumes_without_asking():
    floor = paused_floor()
    floor.speech_stopped()
    assert floor.heard("", is_empty=True) == [Resume(1)]
    assert floor.speech_started() == [Pause(1)]
    assert floor.heard("", is_empty=True) == []
    assert floor.speech_stopped() == [Resume(1)]
    assert floor.speech_started() == [Pause(1)]
    floor.speech_stopped()
    floor.speech_started()
    assert floor.heard("", is_empty=True) == []
    floor.speech_stopped()
    assert floor.heard("hey", is_empty=False) == [Ask(3, 1, ("", "hey"))]


def test_a_reply_opening_over_speech_pauses_with_that_speech_over_it():
    floor: Floor[str] = Floor("pause", 0.0)
    assert floor.speech_started() == []
    floor.speech_stopped()
    assert floor.heard("earlier") == [Answer(("earlier",))]
    floor.speech_started()
    assert floor.reply_started(1) == [Pause(1)]
    assert floor.reply_started(1) == []
    floor.speech_stopped()
    assert floor.heard("over it") == [Ask(0, 1, ("over it",))]
    assert floor.decided(0, -1.0) == [Resume(1)]


def test_a_reply_opening_over_speech_already_transcribed_resumes_when_it_stops():
    floor: Floor[str] = Floor("pause", 0.0)
    floor.speech_started()
    assert floor.heard("earlier") == [Answer(("earlier",))]
    assert floor.reply_started(1) == [Pause(1)]
    assert floor.speech_stopped() == [Resume(1)]
    assert floor.heard("later") == [Answer(("later",))]


def test_a_commit_without_speech_keeps_the_fifo_aligned():
    floor = paused_floor()
    floor.speech_stopped()
    floor.empty_utterance()
    assert floor.heard("over") == [Ask(0, 1, ("over",))]
    assert floor.decided(0, -1.0) == [Resume(1)]
    assert floor.heard("") == [Answer(("",))]


def test_margin_sums_the_labels_of_each_side_and_a_missing_side_is_minus_infinity():
    question = AddresseeQuestion("Addressed?", ("yes", "Yes"), ("no", "No"))
    assert question.labels == {"yes", "Yes", "no", "No"}
    assert question.margin(
        {"yes": math.log(0.3), "Yes": math.log(0.2), "no": math.log(0.1)}
    ) == pytest.approx(math.log(0.5) - math.log(0.1))
    assert question.margin({"No": -0.1}) == -math.inf
    assert question.margin({"yes": -0.1, "maybe": -3.0}) == math.inf
    assert question.margin({"maybe": -3.0}) is None
