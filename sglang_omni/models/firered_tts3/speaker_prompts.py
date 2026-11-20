# SPDX-License-Identifier: Apache-2.0
"""Tokenize Speaker prompts and validate their required special tokens."""

from __future__ import annotations

import regex
from transformers import PreTrainedTokenizerBase

SPEAKER_TEXT_HOLDER = "<|text_holder|>"
SPEAKER_TEXT_PAD = "<|text_pad|>"
SPEAKER_AUDIO_START = "<|audio_start|>"
SPACE_BEFORE_HAN_PATTERN = regex.compile(r"\s+(?=\p{Han})")


def compose_speaker_chatml(
    speaker_token: str,
    *,
    style: str = "",
    text_token_count: int = 0,
    dual_stream: bool = False,
    interleave_ratio: tuple[int, int] = (2, 1),
) -> str:
    """Build the speaker prefix through its audio-start token."""
    if dual_stream:
        user_prompt = f"{style} Align the interleaved text and speech chunks.".strip()
        text_steps, audio_steps = interleave_ratio
        if text_steps < 1 or audio_steps < 1:
            raise ValueError(
                f"FireRedTTS3 interleave_ratio {interleave_ratio!r} invalid; "
                "expected [text_steps, audio_steps] with both >= 1"
            )
        else:
            pass
    else:
        user_prompt = f"{style} Convert text to speech.".strip()
        user_prompt = f"{user_prompt}\n{SPEAKER_TEXT_HOLDER * text_token_count}"
    return (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        f"{speaker_token}{SPEAKER_AUDIO_START}"
    )


def is_han_character(character: str) -> bool:
    return (
        "一" <= character <= "鿿"
        or "㐀" <= character <= "䶿"
        or "豈" <= character <= "﫿"
        or "\U00020000" <= character <= "\U0002a6df"
    )


def encode_speaker_text(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    """Expand multi-character Han tokens while preserving unencodable words."""
    text = SPACE_BEFORE_HAN_PATTERN.sub("", text)
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    unknown_token_id = tokenizer.unk_token_id
    text_token_ids: list[int] = []
    for token_id in token_ids:
        decoded_text = tokenizer.decode([int(token_id)]).strip()
        if len(decoded_text) < 2 or not all(
            is_han_character(character) for character in decoded_text
        ):
            text_token_ids.append(int(token_id))
            continue
        else:
            pass
        character_token_ids: list[int] = []
        for character in decoded_text:
            encoded_character_ids = tokenizer(character, add_special_tokens=False)[
                "input_ids"
            ]
            if (
                unknown_token_id is not None
                and unknown_token_id in encoded_character_ids
            ):
                character_token_ids = []
                break
            else:
                pass
            character_token_ids.extend(
                int(token_id) for token_id in encoded_character_ids
            )
        if character_token_ids:
            text_token_ids.extend(character_token_ids)
        else:
            text_token_ids.append(int(token_id))
    return text_token_ids


def resolve_speaker_token_id(tokenizer: PreTrainedTokenizerBase, token: str) -> int:
    """Reject a tokenizer missing a required speaker token."""
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or (
        token_id == tokenizer.unk_token_id and token not in tokenizer.get_vocab()
    ):
        raise ValueError(
            f"FireRedTTS3 speaker requests require a tokenizer with the "
            f"{token} token — this model's tokenizer has none (the speaker "
            "variant is a separate checkpoint)"
        )
    else:
        pass
    return int(token_id)


def encode_speaker_prefill_ids(
    tokenizer: PreTrainedTokenizerBase,
    *,
    text: str,
    speaker_token: str,
    style: str | None = None,
) -> list[int]:
    """Replace speaker text holders with the character-level token stream."""
    text_token_ids = encode_speaker_text(tokenizer, text)
    if not text_token_ids:
        raise ValueError("FireRedTTS3 speaker text encoded to zero tokens")
    else:
        pass
    chatml_prompt = compose_speaker_chatml(
        speaker_token,
        style="" if style is None else style,
        text_token_count=len(text_token_ids),
    )
    template_token_ids = encode_speaker_text(tokenizer, chatml_prompt)
    text_holder_token_id = resolve_speaker_token_id(tokenizer, SPEAKER_TEXT_HOLDER)
    prompt_token_ids: list[int] = []
    consumed_text_token_count = 0
    for template_token_id in template_token_ids:
        if int(template_token_id) == text_holder_token_id:
            prompt_token_ids.append(text_token_ids[consumed_text_token_count])
            consumed_text_token_count += 1
        else:
            prompt_token_ids.append(int(template_token_id))
    if consumed_text_token_count != len(text_token_ids):
        raise ValueError(
            f"FireRedTTS3 speaker template carries {consumed_text_token_count} text slots "
            f"for {len(text_token_ids)} text tokens — composition bug"
        )
    else:
        pass
    return prompt_token_ids


def encode_speaker_dual_stream_prompt(
    tokenizer: PreTrainedTokenizerBase,
    *,
    text: str,
    speaker_token: str,
    style: str | None = None,
    interleave_ratio: tuple[int, int] = (2, 1),
) -> tuple[list[int], list[int], int]:
    """Return the speaker prefix, interleaved text tokens, and text-padding id."""
    if not text.strip():
        raise ValueError("FireRedTTS3 speaker requests require non-empty text")
    else:
        pass
    text_token_ids = encode_speaker_text(tokenizer, text)
    if not text_token_ids:
        raise ValueError("FireRedTTS3 speaker text encoded to zero tokens")
    else:
        pass
    chatml_prompt = compose_speaker_chatml(
        speaker_token,
        style="Use a 中性 tone." if style is None else style,
        dual_stream=True,
        interleave_ratio=interleave_ratio,
    )
    prompt_token_ids = encode_speaker_text(tokenizer, chatml_prompt)
    text_padding_token_id = resolve_speaker_token_id(tokenizer, SPEAKER_TEXT_PAD)
    return prompt_token_ids, text_token_ids, text_padding_token_id
