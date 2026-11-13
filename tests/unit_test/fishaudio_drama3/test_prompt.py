# SPDX-License-Identifier: Apache-2.0
"""Drama 3 prompt rows: the byte layout, speaker rules, rejections."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from sglang_omni.models.fishaudio_drama3.prompt import ReferencePart, build_prompt_rows

NUM_CODEBOOKS = 10
CODEBOOK_SIZE = 4096
# Protocol literals are written out here, not imported, so a changed literal fails.
SYSTEM = "<|im_start|>system\n"
USER = "<|im_start|>user\n"
VOICE_TURN = "<|im_start|>assistant\n<|voice|>"
END = "<|im_end|>\n"
TTS_INSTRUCTION = "convert the provided text to speech"
REFERENCE_INSTRUCTION = (
    "convert the provided text to speech reference to the following:"
)
DEFAULT_SYSTEM_TURN: list[str | torch.Tensor] = [SYSTEM, TTS_INSTRUCTION, END]
# Where the text under test sits in a request.
TextCarrier = Literal["reference", "input"]


def build_toy_tokenizer() -> Tokenizer:
    """Byte-level BPE whose merges join two newlines and "t[", so a newline seam
    tells per-segment encoding apart from encoding the joined string."""
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocabulary = {symbol: token_id for token_id, symbol in enumerate(alphabet)}
    # Ċ is the byte-level symbol of a newline.
    vocabulary["ĊĊ"] = len(vocabulary)
    vocabulary["t["] = len(vocabulary)
    tokenizer = Tokenizer(models.BPE(vocab=vocabulary, merges=[("Ċ", "Ċ"), ("t", "[")]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    tokenizer.add_special_tokens(
        ["<|im_start|>", "<|im_end|>", "<|voice|>"]
        + [f"<|speaker:{speaker_id}|>" for speaker_id in range(16)]
        + [f"<|semantic:{code}|>" for code in range(CODEBOOK_SIZE)]
    )
    return tokenizer


@pytest.fixture(scope="module")
def toy_tokenizer() -> Tokenizer:
    return build_toy_tokenizer()


@pytest.fixture(scope="module")
def real_tokenizer() -> Tokenizer:
    """The private tokenizer FISH_DRAMA3_TOKENIZER names; skipped when absent."""
    configured_path = os.environ.get("FISH_DRAMA3_TOKENIZER")
    if configured_path is None:
        pytest.skip("FISH_DRAMA3_TOKENIZER is not set")
    else:
        pass
    tokenizer_path = Path(configured_path).expanduser()
    if not tokenizer_path.is_file():
        pytest.skip(f"FISH_DRAMA3_TOKENIZER names no file: {tokenizer_path}")
    else:
        pass
    return Tokenizer.from_file(str(tokenizer_path))


def clip_codes(frame_count: int, seed: int) -> torch.Tensor:
    """[NUM_CODEBOOKS, frame_count] codes whose values name the clip."""
    codebook_offsets = 100 * torch.arange(NUM_CODEBOOKS).unsqueeze(1)
    return seed + codebook_offsets + torch.arange(frame_count)


def reference(transcript: str, audio_codes: torch.Tensor) -> ReferencePart:
    return ReferencePart(transcript=transcript, audio_codes=audio_codes)


def reference_header(reference_number: int, transcript_text: str) -> str:
    return f"\n\n[Reference {reference_number}]\nText: {transcript_text}\nSpeech: "


def build(
    tokenizer: Tokenizer, text: str, references: list[ReferencePart]
) -> torch.Tensor:
    return build_prompt_rows(
        text,
        references,
        tokenizer,
        num_codebooks=NUM_CODEBOOKS,
        codebook_size=CODEBOOK_SIZE,
    )


def expected_rows(
    tokenizer: Tokenizer, segments: list[str | torch.Tensor]
) -> torch.Tensor:
    """Rows of hand-written protocol segments: one per token, then one per frame."""
    semantic_begin_id = tokenizer.token_to_id("<|semantic:0|>")
    rows: list[list[int]] = []
    for segment in segments:
        if isinstance(segment, str):
            token_ids = tokenizer.encode(segment).ids
            rows += [[token_id] + [0] * NUM_CODEBOOKS for token_id in token_ids]
        else:
            frames = segment.T.tolist()
            rows += [[semantic_begin_id + frame[0], *frame] for frame in frames]
    return torch.tensor(rows, dtype=torch.long)


LAYOUT_CASES = [
    pytest.param(
        "hello",
        [],
        DEFAULT_SYSTEM_TURN + [USER, "<|speaker:0|> hello", END, VOICE_TURN],
        id="no-references-use-the-default-instruction",
    ),
    pytest.param(
        "say it",
        [reference("lone take", clip_codes(4, 1))],
        [SYSTEM, REFERENCE_INSTRUCTION, reference_header(1, "<|speaker:0|> lone take")]
        + [clip_codes(4, 1), END, USER, "<|speaker:0|> say it", END, VOICE_TURN],
        id="one-reference",
    ),
    pytest.param(
        "say it",
        [
            reference("opening take", clip_codes(4, 1)),
            reference("<|speaker:1|> answer take", clip_codes(5, 2)),
            reference("<|speaker:2|> closing take", clip_codes(6, 3)),
        ],
        [SYSTEM, REFERENCE_INSTRUCTION]
        + [reference_header(1, "<|speaker:0|> opening take"), clip_codes(4, 1)]
        + [reference_header(2, "<|speaker:1|> answer take"), clip_codes(5, 2)]
        + [reference_header(3, "<|speaker:2|> closing take"), clip_codes(6, 3)]
        + [END, USER, "<|speaker:0|> say it", END, VOICE_TURN],
        id="three-references-take-their-speakers-from-the-transcripts",
    ),
    pytest.param(
        "hello",
        [
            reference("A whole sample.", clip_codes(8, 10)),
            reference(
                "<|speaker:1|> One voice. <|speaker:0|> Another voice.",
                clip_codes(5, 40),
            ),
        ],
        [SYSTEM, REFERENCE_INSTRUCTION]
        + [reference_header(1, "<|speaker:0|> A whole sample."), clip_codes(8, 10)]
        + [reference_header(2, "<|speaker:1|> One voice. <|speaker:0|> Another voice.")]
        + [clip_codes(5, 40), END, USER, "<|speaker:0|> hello", END, VOICE_TURN],
        id="references-keep-an-inner-speaker-switch",
    ),
]


@pytest.mark.parametrize(("text", "references", "segments"), LAYOUT_CASES)
def test_prompt_layout(
    toy_tokenizer: Tokenizer,
    text: str,
    references: list[ReferencePart],
    segments: list[str | torch.Tensor],
) -> None:
    rows = build(toy_tokenizer, text, references)

    assert torch.equal(rows, expected_rows(toy_tokenizer, segments))


@pytest.mark.parametrize(
    ("text", "spoken_text"),
    [
        ("hello", "<|speaker:0|> hello"),
        ("<|speaker:15|> x", "<|speaker:15|> x"),
        (" <|speaker:1|> hi", " <|speaker:1|> hi"),
        ("\n<|speaker:1|> hi", "\n<|speaker:1|> hi"),
        (
            "\N{IDEOGRAPHIC SPACE}<|speaker:1|> hi",
            "\N{IDEOGRAPHIC SPACE}<|speaker:1|> hi",
        ),
        ("a <|speaker:3|> b", "<|speaker:0|> a <|speaker:3|> b"),
        # U+001C is not Unicode White_Space, so the tag does not lead the text.
        ("\x1c<|speaker:1|> hi", "<|speaker:0|> \x1c<|speaker:1|> hi"),
    ],
)
def test_untagged_input_is_spoken_by_speaker_0(
    toy_tokenizer: Tokenizer, text: str, spoken_text: str
) -> None:
    rows = build(toy_tokenizer, text, [])

    segments = DEFAULT_SYSTEM_TURN + [USER, spoken_text, END, VOICE_TURN]
    assert torch.equal(rows, expected_rows(toy_tokenizer, segments))


@pytest.mark.parametrize(
    ("transcript", "header_text"),
    [
        ("  padded  ", "<|speaker:0|>   padded  "),
        ("  <|speaker:2|> tagged", "<|speaker:2|>   tagged"),
        ("\n<|speaker:3|> line", "<|speaker:3|>\n line"),
        (
            "\N{IDEOGRAPHIC SPACE}<|speaker:4|> x",
            "<|speaker:4|>\N{IDEOGRAPHIC SPACE} x",
        ),
        # U+001C is not Unicode White_Space, so the tag is not leading.
        ("\x1c<|speaker:2|> x", "<|speaker:0|> \x1c<|speaker:2|> x"),
    ],
)
def test_reference_tag_moves_in_front_of_leading_white_space(
    toy_tokenizer: Tokenizer, transcript: str, header_text: str
) -> None:
    rows = build(toy_tokenizer, "hi", [reference(transcript, clip_codes(1, 1))])

    segments = [SYSTEM, REFERENCE_INSTRUCTION, reference_header(1, header_text)]
    segments += [clip_codes(1, 1), END, USER, "<|speaker:0|> hi", END, VOICE_TURN]
    assert torch.equal(rows, expected_rows(toy_tokenizer, segments))


def test_rows_are_cpu_int64_with_the_semantic_id_then_every_codebook(
    toy_tokenizer: Tokenizer,
) -> None:
    """Rows stay on the CPU even when a caller sets another default device."""
    # Codebook k holds frames [2k, 2k + 1].
    audio_codes = torch.arange(2 * NUM_CODEBOOKS).reshape(NUM_CODEBOOKS, 2)

    with torch.device("meta"):
        rows = build(toy_tokenizer, "say this", [reference("clip", audio_codes)])

    semantic_begin_id = toy_tokenizer.token_to_id("<|semantic:0|>")
    is_audio_row = rows[:, 1:].any(dim=1)
    assert rows.dtype == torch.long
    assert rows.device.type == "cpu"
    assert rows.shape[1] == 11
    assert rows[is_audio_row].tolist() == [
        [semantic_begin_id, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
        [semantic_begin_id + 1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    ]
    text_segments = [SYSTEM, REFERENCE_INSTRUCTION]
    text_segments += [reference_header(1, "<|speaker:0|> clip"), END, USER]
    text_segments += ["<|speaker:0|> say this", END, VOICE_TURN]
    text_token_ids = [toy_tokenizer.encode(segment).ids for segment in text_segments]
    assert rows[~is_audio_row, 0].tolist() == sum(text_token_ids, [])


def test_each_literal_and_part_is_tokenized_on_its_own(
    toy_tokenizer: Tokenizer,
) -> None:
    text = "\n<|speaker:0|> line"

    rows = build(toy_tokenizer, text, [])

    segments = DEFAULT_SYSTEM_TURN + [USER, text, END, VOICE_TURN]
    joined_segments = DEFAULT_SYSTEM_TURN + [USER + text, END, VOICE_TURN]
    assert torch.equal(rows, expected_rows(toy_tokenizer, segments))
    assert not torch.equal(rows, expected_rows(toy_tokenizer, joined_segments))


@pytest.mark.parametrize(
    ("text", "spoken_text"),
    [
        (
            "Direction: [quietly] hold back a little\n"
            "Line: <soft>We should go home now.</soft>",
            "<|speaker:0|> Direction: [quietly] hold back a little\n"
            "Line: <soft>We should go home now.</soft>",
        ),
        (
            "current part<|endofturn|>next part",
            "<|speaker:0|> current part<|endofturn|>next part",
        ),
        # U+001F is not Unicode White_Space, so this text is not empty.
        ("\x1f", "<|speaker:0|> \x1f"),
    ],
)
def test_other_markup_is_caller_content(
    toy_tokenizer: Tokenizer, text: str, spoken_text: str
) -> None:
    rows = build(toy_tokenizer, text, [])

    segments = DEFAULT_SYSTEM_TURN + [USER, spoken_text, END, VOICE_TURN]
    assert torch.equal(rows, expected_rows(toy_tokenizer, segments))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_codes_take_both_ends_of_the_codebook_in_int32_and_int64(
    toy_tokenizer: Tokenizer, dtype: torch.dtype
) -> None:
    audio_codes = torch.tensor([[0, CODEBOOK_SIZE - 1]] * NUM_CODEBOOKS, dtype=dtype)

    rows = build(toy_tokenizer, "hi", [reference("clip", audio_codes)])

    segments = [
        SYSTEM,
        REFERENCE_INSTRUCTION,
        reference_header(1, "<|speaker:0|> clip"),
    ]
    segments += [audio_codes, END, USER, "<|speaker:0|> hi", END, VOICE_TURN]
    assert rows.dtype == torch.long
    assert torch.equal(rows, expected_rows(toy_tokenizer, segments))


BLANK_TEXT = " \n\t\N{IDEOGRAPHIC SPACE}"


@pytest.mark.parametrize(
    ("text", "references", "error"),
    [
        pytest.param(BLANK_TEXT, [], "input: text must not be empty", id="blank-input"),
        pytest.param(
            "hi",
            [reference(BLANK_TEXT, clip_codes(1, 1))],
            "references.0: text must not be empty",
            id="blank-transcript",
        ),
    ],
)
def test_blank_text_is_rejected(
    toy_tokenizer: Tokenizer, text: str, references: list[ReferencePart], error: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(error)):
        build(toy_tokenizer, text, references)


def request_carrying(
    text: str, carrier: TextCarrier
) -> tuple[str, list[ReferencePart]]:
    transcript = text if carrier == "reference" else "clip"
    return (
        text if carrier == "input" else "again",
        [reference(transcript, clip_codes(1, 1))],
    )


STRUCTURAL_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<|voice|>",
    "<|text|>",
    "<|interleave|>",
    "<|semantic:3|>",
]
MALFORMED_SPEAKER_TEXTS = [
    "<|speaker:16|> x",
    "a <|speaker:99|> b",
    "<|speaker:01|> x",
    "<|speaker:|> x",
    "<|speaker0|> x",
    "hi <|speaker",
    "<|speaker:abc|> x",
    "<|speaker:1> x",
]


@pytest.mark.parametrize(
    ("carrier", "token"),
    [("reference", token) for token in STRUCTURAL_TOKENS]
    + [("input", STRUCTURAL_TOKENS[0])],
)
def test_structural_tokens_in_caller_text_are_rejected(
    toy_tokenizer: Tokenizer, carrier: TextCarrier, token: str
) -> None:
    text, references = request_carrying(f"hi {token} there", carrier)

    with pytest.raises(ValueError, match="is written by the prompt builder") as error:
        build(toy_tokenizer, text, references)
    assert token.removesuffix("3|>") in str(error.value)


@pytest.mark.parametrize(
    ("carrier", "text"),
    [("reference", text) for text in MALFORMED_SPEAKER_TEXTS]
    + [("input", MALFORMED_SPEAKER_TEXTS[0])],
)
def test_speaker_tags_the_tokenizer_lacks_are_rejected(
    toy_tokenizer: Tokenizer, carrier: TextCarrier, text: str
) -> None:
    request_text, references = request_carrying(text, carrier)

    with pytest.raises(ValueError, match="speaker tags must be written"):
        build(toy_tokenizer, request_text, references)


def codes_with_one_value(codebook_index: int, value: int) -> torch.Tensor:
    audio_codes = torch.zeros((NUM_CODEBOOKS, 2), dtype=torch.long)
    audio_codes[codebook_index, 1] = value
    return audio_codes


@pytest.mark.parametrize(
    ("audio_codes", "error"),
    [
        pytest.param(clip_codes(1, 1)[:-1], "int32 or int64", id="nine-codebooks"),
        pytest.param(
            torch.zeros((NUM_CODEBOOKS, 0), dtype=torch.long),
            "int32 or int64",
            id="no-frames",
        ),
        pytest.param(clip_codes(1, 1)[0], "int32 or int64", id="one-dimension"),
        pytest.param(clip_codes(1, 1).float(), "int32 or int64", id="float-codes"),
        pytest.param(codes_with_one_value(0, -1), "in [0, 4096)", id="negative"),
        pytest.param(
            codes_with_one_value(0, CODEBOOK_SIZE), "in [0, 4096)", id="codebook-size"
        ),
        pytest.param(
            codes_with_one_value(5, CODEBOOK_SIZE), "in [0, 4096)", id="later-codebook"
        ),
    ],
)
def test_malformed_reference_codes_are_rejected(
    toy_tokenizer: Tokenizer, audio_codes: torch.Tensor, error: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(error)):
        build(toy_tokenizer, "hi", [reference("clip", audio_codes)])


@pytest.mark.parametrize(
    ("num_codebooks", "codebook_size"), [(0, CODEBOOK_SIZE), (NUM_CODEBOOKS, 0)]
)
def test_codebook_parameters_must_be_positive(
    toy_tokenizer: Tokenizer, num_codebooks: int, codebook_size: int
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        build_prompt_rows(
            "hi",
            [],
            toy_tokenizer,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
        )


def test_real_tokenizer_special_tokens_are_single_contiguous_ids(
    real_tokenizer: Tokenizer,
) -> None:
    for token in ["<|im_start|>", "<|im_end|>", "<|voice|>", "<|endofturn|>"]:
        assert real_tokenizer.encode(token).ids == [real_tokenizer.token_to_id(token)]
    speaker_begin_id = real_tokenizer.token_to_id("<|speaker:0|>")
    for speaker_id in range(16):
        speaker_token_ids = real_tokenizer.encode(f"<|speaker:{speaker_id}|>").ids
        assert speaker_token_ids == [speaker_begin_id + speaker_id]
    semantic_begin_id = real_tokenizer.token_to_id("<|semantic:0|>")
    for code in range(CODEBOOK_SIZE):
        semantic_token_ids = real_tokenizer.encode(f"<|semantic:{code}|>").ids
        assert semantic_token_ids == [semantic_begin_id + code]
    assert real_tokenizer.token_to_id("<|speaker:16|>") is None
    assert real_tokenizer.token_to_id(f"<|semantic:{CODEBOOK_SIZE}|>") is None


def test_real_tokenizer_rows_follow_the_segment_boundaries(
    real_tokenizer: Tokenizer,
) -> None:
    header = reference_header(1, "<|speaker:0|> clip")
    text = "\n<|speaker:0|> first line"

    rows = build(real_tokenizer, text, [reference("clip", clip_codes(2, 1))])
    default_rows = build(real_tokenizer, "hello", [])

    segments = [SYSTEM, REFERENCE_INSTRUCTION, header, clip_codes(2, 1), END, USER]
    segments += [text, END, VOICE_TURN]
    assert torch.equal(rows, expected_rows(real_tokenizer, segments))
    joined_token_ids = real_tokenizer.encode(USER + text).ids
    split_token_ids = real_tokenizer.encode(USER).ids + real_tokenizer.encode(text).ids
    assert joined_token_ids != split_token_ids
    assert rows.shape[1] == NUM_CODEBOOKS + 1
    assert rows[0, 0] == real_tokenizer.token_to_id("<|im_start|>")
    assert rows[-1, 0] == real_tokenizer.token_to_id("<|voice|>")
    default_segments = DEFAULT_SYSTEM_TURN + [USER, "<|speaker:0|> hello", END]
    default_segments += [VOICE_TURN]
    assert torch.equal(default_rows, expected_rows(real_tokenizer, default_segments))
