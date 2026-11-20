# SPDX-License-Identifier: Apache-2.0

from sglang_omni.models.nemotron_transcribe.speaker_attribution import (
    attribute_speakers,
    choose_tagged_transcript,
    decide_tagged_retry,
    diarization_segments,
    parse_speaker_tags,
    speaker_tag_request,
    strip_speaker_tags,
)


def test_two_tags_are_a_speaker_request_and_a_bare_tag_is_not() -> None:
    assert speaker_tag_request("Use <spk:0> and <spk:1>.")
    assert not speaker_tag_request("<spk:0>")
    assert not speaker_tag_request("Transcribe the audio.")


def test_tags_are_stripped_and_carried_onto_the_following_words() -> None:
    text = "hello <spk:1> there friend"
    assert strip_speaker_tags(text) == "hello there friend"
    assert parse_speaker_tags(text) == [None, 1, 1]


def test_transcript_tags_override_the_activity_slot() -> None:
    words = [
        {"word": "hello", "start": 0.0, "end": 0.1},
        {"word": "there", "start": 0.1, "end": 0.2},
    ]
    activity = [[0.9, 0.1], [0.1, 0.8]]
    attributed = attribute_speakers(
        words,
        "hello <spk:1> there",
        "Speakers <spk:0> <spk:1>",
        activity,
        0.1,
    )
    assert attributed[0]["speaker"] == "unknown"
    assert attributed[1]["speaker"] == "spk:1"
    assert attributed[1]["speaker_slot"] == 1
    assert attributed[1]["speaker_confidence"] == 0.8


def test_untagged_words_take_the_louder_slot() -> None:
    words = [{"word": "hello", "start": 0.0, "end": 0.2}]
    activity = [[0.2, 0.7], [0.1, 0.9]]
    attributed = attribute_speakers(
        words,
        "hello",
        "Speakers <spk:0> <spk:1>",
        activity,
        0.1,
    )
    assert attributed[0]["speaker"] == "spk:1"
    assert attributed[0]["speaker_slot"] == 1


def test_short_speech_is_dropped_by_the_diarization_postprocess() -> None:
    activity = [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]]
    assert diarization_segments(activity, frame_seconds=0.1, duration_s=1.0) == []


def test_a_long_tagless_answer_is_retried_once() -> None:
    prompt = "Tag speakers <spk:0> <spk:1>."
    assert not decide_tagged_retry(prompt, "short")
    long_text = " ".join(["word"] * 200)
    assert decide_tagged_retry(prompt, long_text)
    assert not decide_tagged_retry(prompt, long_text + " <spk:0>")
    assert choose_tagged_transcript(long_text, "no tag here") == long_text
    assert choose_tagged_transcript(long_text, "hello <spk:1> there") == (
        "hello <spk:1> there"
    )
