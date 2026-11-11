# SPDX-License-Identifier: Apache-2.0
"""Transcript cleanup removes non-speech markers and never eats a real transcript."""

from __future__ import annotations

from sglang_omni.models.nemotron_transcribe.transcript_cleanup import (
    clean_transcript,
    strip_refusal_tail,
)

REFUSAL = (
    "I'm sorry, I can't provide a transcription for this audio as it contains "
    "only silence."
)


def tagged_meeting(words: int = 3000) -> str:
    """A tagged meeting without punctuation: an apology early, 'audio' much later."""
    filler = "so we could put the label a bit higher up and then the remote"
    body: list[str] = []
    while sum(len(chunk.split()) for chunk in body) < words:
        body.append(f"<spk:{len(body) % 4}> {filler}")
    text = " ".join(body).split()
    text[40:40] = ["<spk:1>", "i'm", "sorry", "what", "you", "need"]
    text[-30:-30] = ["<spk:2>", "need", "a", "lot", "of", "audio"]
    return " ".join(text)


def test_reasoning_markers_and_unknown_runs_are_removed() -> None:
    assert clean_transcript("<think></think>Hello <unk><unk> world.") == (
        "Hello world."
    )
    assert clean_transcript("Caf\ufffd\ufffd au lait.") == "Caf au lait."


def test_speaker_and_event_tags_stay() -> None:
    text = "<spk:0> hello <ev:laughter> there <spk:1> hi"
    assert clean_transcript(text) == text


def test_a_long_tagged_meeting_with_an_early_apology_is_kept_whole() -> None:
    meeting = tagged_meeting()
    assert clean_transcript(meeting) == meeting
    assert strip_refusal_tail(meeting) == meeting


def test_a_refusal_tail_is_stripped_and_a_pure_refusal_empties() -> None:
    assert (
        clean_transcript("For some reason we were blocked from entering. " + REFUSAL)
        == "For some reason we were blocked from entering."
    )
    assert clean_transcript(REFUSAL) == ""
    assert (
        clean_transcript("<spk:0> for some reason we were blocked <spk:1> " + REFUSAL)
        == "<spk:0> for some reason we were blocked"
    )


def test_an_apology_about_audio_inside_a_meeting_is_speech() -> None:
    words = ["<spk:0>", "okay", "so", "i'm", "sorry", "the", "audio", "was", "out."]
    tail = ["<spk:1>", "right", "let's", "carry", "on", "with", "the", "design."] * 40
    text = " ".join(words + tail)
    assert clean_transcript(text) == text


def test_the_opener_to_object_span_is_bounded() -> None:
    short_gap = (
        "I'm sorry but I do think that I am not able to give you any kind of "
        "transcription here."
    )
    assert clean_transcript("Hello. " + short_gap) == "Hello."
    long_gap = (
        "I'm sorry what you need what green right oh this window here does it "
        "squeeze in like that yeah there we go okay right did you get to hand out "
        "the forms for the audio"
    )
    assert clean_transcript("Hello. " + long_gap) == "Hello. " + long_gap
