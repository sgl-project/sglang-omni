# SPDX-License-Identifier: Apache-2.0
"""Build Drama 3 prompt rows in the reference-bracket-v1 layout."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from tokenizers import Tokenizer

TTS_INSTRUCTION = "convert the provided text to speech"
REFERENCE_INSTRUCTION = (
    "convert the provided text to speech reference to the following:"
)
MESSAGE_END = "<|im_end|>\n"
GENERATION_START = "<|im_start|>assistant\n<|voice|>"
SEMANTIC_BEGIN_TOKEN = "<|semantic:0|>"
# note (SunskyXH): only the builder writes these; other tokens in caller text
# are content.
STRUCTURAL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|voice|>",
    "<|text|>",
    "<|interleave|>",
    "<|semantic:",
)
MAX_SPEAKER_ID = 15
# note (SunskyXH): the spellings the tokenizer has, <|speaker:0|> to <|speaker:15|>.
SPEAKER_IDS = frozenset(str(speaker_id) for speaker_id in range(MAX_SPEAKER_ID + 1))
SPEAKER_TAG_PATTERN = re.compile(r"<\|speaker:([0-9]+)\|>")
# note (SunskyXH): the Unicode White_Space property. str.strip() with no argument also
# strips U+001C..U+001F, which would change the empty check and the speaker rule.
UNICODE_WHITE_SPACE = "".join(
    map(
        chr,
        (*range(0x9, 0xE), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B))
        + (0x2028, 0x2029, 0x202F, 0x205F, 0x3000),
    )
)


@dataclass(kw_only=True)
class ReferencePart:
    """A voice reference clip and its transcript."""

    transcript: str
    audio_codes: torch.Tensor


def validate_caller_text(text: str, location: str) -> None:
    """Reject empty text, structural tokens and speaker tags the tokenizer lacks."""
    if not text.strip(UNICODE_WHITE_SPACE):
        raise ValueError(f"{location}: text must not be empty")
    else:
        pass
    for token in STRUCTURAL_TOKENS:
        if token in text:
            raise ValueError(
                f"{location}: {token} is written by the prompt builder and must "
                f"not appear in caller text"
            )
        else:
            pass
    for speaker_marker in re.finditer(r"<\|speaker", text):
        speaker_tag = SPEAKER_TAG_PATTERN.match(text, speaker_marker.start())
        if speaker_tag is None or speaker_tag.group(1) not in SPEAKER_IDS:
            raise ValueError(
                f"{location}: speaker tags must be written <|speaker:N|> with N in "
                f"0..{MAX_SPEAKER_ID} and no leading zero; got {text!r}"
            )
        else:
            pass


def validate_audio_codes(
    audio_codes: torch.Tensor, num_codebooks: int, codebook_size: int, location: str
) -> None:
    if (
        audio_codes.dtype not in (torch.int32, torch.int64)
        or audio_codes.ndim != 2
        or audio_codes.shape[0] != num_codebooks
        or audio_codes.shape[1] == 0
    ):
        raise ValueError(
            f"{location}: audio codes must be an int32 or int64 [{num_codebooks}, "
            f"frames] tensor with at least one frame; got {audio_codes.dtype} "
            f"{tuple(audio_codes.shape)}"
        )
    elif audio_codes.min() < 0 or audio_codes.max() >= codebook_size:
        raise ValueError(f"{location}: audio codes must be in [0, {codebook_size})")
    else:
        pass


def with_speaker_tag(text: str, default_speaker_id: int) -> str:
    """Tag text with default_speaker_id unless a tag leads it after white space."""
    if SPEAKER_TAG_PATTERN.match(text.lstrip(UNICODE_WHITE_SPACE)) is None:
        return f"<|speaker:{default_speaker_id}|> {text}"
    else:
        return text


def build_prompt_rows(
    text: str,
    references: list[ReferencePart],
    tokenizer: Tokenizer,
    *,
    num_codebooks: int,
    codebook_size: int,
) -> torch.Tensor:
    """Encode text and its voice references as CPU int64 rows ending in the voice turn.

    Each literal and part is tokenized on its own, because tokenizing the joined
    string changes ids at the seams.
    """
    if num_codebooks <= 0 or codebook_size <= 0:
        raise ValueError(
            f"num_codebooks and codebook_size must be positive; got "
            f"{num_codebooks} and {codebook_size}"
        )
    else:
        pass
    semantic_begin_id = tokenizer.token_to_id(SEMANTIC_BEGIN_TOKEN)
    if semantic_begin_id is None:
        raise ValueError(f"the tokenizer has no {SEMANTIC_BEGIN_TOKEN} token")
    else:
        pass

    segments: list[str | torch.Tensor] = [
        "<|im_start|>system\n",
        REFERENCE_INSTRUCTION if references else TTS_INSTRUCTION,
    ]
    for reference_index, reference in enumerate(references):
        location = f"references.{reference_index}"
        validate_caller_text(reference.transcript, location)
        validate_audio_codes(
            reference.audio_codes, num_codebooks, codebook_size, location
        )
        # note (SunskyXH): the tag must follow "Text: " directly, so a tag
        # behind leading white space moves in front of it.
        tagged_transcript = with_speaker_tag(reference.transcript, 0)
        transcript_body = tagged_transcript.lstrip(UNICODE_WHITE_SPACE)
        leading_speaker_tag = SPEAKER_TAG_PATTERN.match(transcript_body)
        assert leading_speaker_tag is not None
        leading_white_space = tagged_transcript[
            : len(tagged_transcript) - len(transcript_body)
        ]
        transcript = (
            leading_speaker_tag.group(0)
            + leading_white_space
            + transcript_body[leading_speaker_tag.end() :]
        )
        segments += [
            f"\n\n[Reference {reference_index + 1}]\nText: {transcript}\nSpeech: ",
            reference.audio_codes,
        ]
    validate_caller_text(text, "input")
    segments += [MESSAGE_END, "<|im_start|>user\n", with_speaker_tag(text, 0)]
    segments += [MESSAGE_END, GENERATION_START]

    rows: list[torch.Tensor] = []
    for segment in segments:
        if isinstance(segment, str):
            token_ids = torch.tensor(
                tokenizer.encode(segment).ids, dtype=torch.long, device="cpu"
            )
            text_rows = torch.zeros(
                (token_ids.numel(), num_codebooks + 1), dtype=torch.long, device="cpu"
            )
            text_rows[:, 0] = token_ids
            rows.append(text_rows)
        else:
            audio_codes = segment.to(device="cpu", dtype=torch.long)
            rows.append(torch.cat([audio_codes[:1] + semantic_begin_id, audio_codes]).T)
    return torch.cat(rows)
