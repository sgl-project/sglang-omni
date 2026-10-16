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
INFILL_INSTRUCTION = (
    "fill in the missing part of the speech; the suffix follows <|fim_suffix|> "
    "and the prefix follows\n"
)
FIM_SUFFIX = "<|fim_suffix|>"
FIM_PREFIX = "<|fim_prefix|>"
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
    FIM_SUFFIX,
    FIM_PREFIX,
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


@dataclass(kw_only=True)
class SpokenTake:
    """One finished user turn and the audio the assistant spoke for it."""

    text: str
    audio_codes: torch.Tensor


@dataclass(kw_only=True)
class InfillSpan:
    """Audio kept on either side of the span the model regenerates.

    Either side may have zero frames. The suffix is written before the prefix,
    and generation continues from the prefix.
    """

    prefix_codes: torch.Tensor
    suffix_codes: torch.Tensor
    audio_token_count: int | None = None


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
    audio_codes: torch.Tensor,
    num_codebooks: int,
    codebook_size: int,
    location: str,
    *,
    allow_empty: bool = False,
) -> None:
    if (
        audio_codes.dtype not in (torch.int32, torch.int64)
        or audio_codes.ndim != 2
        or audio_codes.shape[0] != num_codebooks
        or (audio_codes.shape[1] == 0 and not allow_empty)
    ):
        raise ValueError(
            f"{location}: audio codes must be an int32 or int64 [{num_codebooks}, "
            f"frames] tensor with at least one frame; got {audio_codes.dtype} "
            f"{tuple(audio_codes.shape)}"
        )
    elif audio_codes.shape[1] == 0:
        pass
    elif audio_codes.min() < 0 or audio_codes.max() >= codebook_size:
        raise ValueError(f"{location}: audio codes must be in [0, {codebook_size})")
    else:
        pass


def audio_frame_rows(audio_codes: torch.Tensor, semantic_begin_id: int) -> torch.Tensor:
    """Prompt rows for one audio span: semantic id, then every codebook."""
    audio_codes = audio_codes.to(device="cpu", dtype=torch.long)
    return torch.cat([audio_codes[:1] + semantic_begin_id, audio_codes]).T


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
    previous_takes: list[SpokenTake] | None = None,
    infill_span: InfillSpan | None = None,
) -> torch.Tensor:
    """Encode one conversation as CPU int64 rows ending at the voice turn.

    Each literal and part is tokenized on its own, because tokenizing the joined
    string changes ids at the seams. A previous take is a finished user turn and
    the assistant audio that followed it. An infill span writes the suffix, then
    the prefix, and leaves the model to speak the middle.
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
    takes = [] if previous_takes is None else previous_takes
    if infill_span is not None and (references or takes):
        raise ValueError("infill does not take references or previous takes")
    else:
        pass

    segments: list[str | torch.Tensor] = ["<|im_start|>system\n"]
    if infill_span is not None:
        for token in (FIM_SUFFIX, FIM_PREFIX):
            if tokenizer.token_to_id(token) is None:
                raise ValueError(f"the tokenizer has no {token} token")
            else:
                pass
        validate_audio_codes(
            infill_span.suffix_codes,
            num_codebooks,
            codebook_size,
            "infill.suffix",
            allow_empty=True,
        )
        validate_audio_codes(
            infill_span.prefix_codes,
            num_codebooks,
            codebook_size,
            "infill.prefix",
            allow_empty=True,
        )
        if (
            infill_span.suffix_codes.shape[1] == 0
            and infill_span.prefix_codes.shape[1] == 0
        ):
            raise ValueError("infill needs audio before or after the edit")
        else:
            pass
        validate_caller_text(text, "input")
        spoken = with_speaker_tag(text, 0)
        if infill_span.audio_token_count is not None:
            spoken = f"{spoken} [audio tokens: {infill_span.audio_token_count}]"
        else:
            pass
        segments += [INFILL_INSTRUCTION, FIM_PREFIX, MESSAGE_END, "<|im_start|>user\n"]
        segments += [spoken + "\n", FIM_SUFFIX]
        if infill_span.suffix_codes.shape[1] > 0:
            segments.append(infill_span.suffix_codes)
        else:
            pass
        segments.append(FIM_PREFIX)
        if infill_span.prefix_codes.shape[1] > 0:
            segments.append(infill_span.prefix_codes)
        else:
            pass
        segments += [MESSAGE_END, GENERATION_START]
    else:
        segments.append(REFERENCE_INSTRUCTION if references else TTS_INSTRUCTION)
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
        for take_index, take in enumerate(takes):
            location = f"previous_takes.{take_index}"
            validate_caller_text(take.text, location)
            validate_audio_codes(
                take.audio_codes, num_codebooks, codebook_size, location
            )
            segments += [
                MESSAGE_END,
                "<|im_start|>user\n",
                with_speaker_tag(take.text, 0),
                MESSAGE_END,
                GENERATION_START,
                take.audio_codes,
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
            rows.append(audio_frame_rows(segment, semantic_begin_id))
    return torch.cat(rows)
