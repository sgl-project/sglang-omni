# SPDX-License-Identifier: Apache-2.0
"""Deterministic tokenization and bounded segmentation for Ming speech."""

from __future__ import annotations

import re
import unicodedata
from math import ceil

from sglang_omni.models.ming3_audio.text_processing_boundaries import (
    choose_fragment_end,
)
from sglang_omni.models.ming3_audio.text_processing_contract import (
    TextSegmentationError,
    TextToken,
    TextTokenKind,
    TextTooLargeError,
    TextUnbreakableError,
)
from sglang_omni.models.ming3_audio.text_processing_markdown import (
    URL_PATTERN,
    split_url_suffix,
)

ASCII_SEMANTIC_CHAR_GROUP = 8
DATE_PATTERN = re.compile(r"[+-]?\d{1,4}(?:[-/]\d{1,2}){1,2}")
NUMBER_PATTERN = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:[.,]\d+)*(?:%|％)?")
VERSION_PATTERN = re.compile(r"[A-Za-z]+\d+(?:[.-][A-Za-z0-9]+)+")
ABBREVIATION_PATTERN = re.compile(r"(?:[A-Za-z]\.){2,}")
TITLE_ABBREVIATION_PATTERN = re.compile(r"(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc)\.")
WORD_PATTERN = re.compile(r"[A-Za-z]+(?:['’_-][A-Za-z0-9]+)*")


def tokenize_text(text: str) -> list[TextToken]:
    tokens: list[TextToken] = []
    position = 0
    while position < len(text):
        keycap_end = find_keycap_end(text, position)
        if keycap_end is not None:
            tokens.append(make_token(text[position:keycap_end], "emoji"))
            position = keycap_end
            continue
        else:
            pass

        url_match = URL_PATTERN.match(text, position)
        if url_match is not None:
            url_text, _ = split_url_suffix(url_match.group(0))
            if url_text:
                tokens.append(make_token(url_text, "url"))
                position += len(url_text)
                continue
            else:
                pass
        else:
            pass

        date_match = DATE_PATTERN.match(text, position)
        if date_match is not None:
            token_end = extend_token_end(text, date_match.end())
            tokens.append(make_token(text[position:token_end], "number"))
            position = token_end
            continue
        else:
            pass

        number_match = NUMBER_PATTERN.match(text, position)
        if number_match is not None:
            token_end = extend_token_end(text, number_match.end())
            tokens.append(make_token(text[position:token_end], "number"))
            position = token_end
            continue
        else:
            pass

        version_match = VERSION_PATTERN.match(text, position)
        if version_match is not None:
            token_end = extend_token_end(text, version_match.end())
            tokens.append(make_token(text[position:token_end], "word"))
            position = token_end
            continue
        else:
            pass

        abbreviation_match = ABBREVIATION_PATTERN.match(text, position)
        if abbreviation_match is not None:
            token_end = extend_token_end(text, abbreviation_match.end())
            tokens.append(make_token(text[position:token_end], "word"))
            position = token_end
            continue
        else:
            pass

        title_abbreviation_match = TITLE_ABBREVIATION_PATTERN.match(text, position)
        if title_abbreviation_match is not None:
            token_end = extend_token_end(text, title_abbreviation_match.end())
            tokens.append(make_token(text[position:token_end], "word"))
            position = token_end
            continue
        else:
            pass

        word_match = WORD_PATTERN.match(text, position)
        if word_match is not None:
            token_end = extend_token_end(text, word_match.end())
            tokens.append(make_token(text[position:token_end], "word"))
            position = token_end
            continue
        else:
            pass

        character = text[position]
        if character == "\n":
            tokens.append(make_token(character, "whitespace"))
            position += 1
        elif character in " \t\f\v":
            whitespace_end = position + 1
            while whitespace_end < len(text) and text[whitespace_end] in " \t\f\v":
                whitespace_end += 1
            tokens.append(make_token(text[position:whitespace_end], "whitespace"))
            position = whitespace_end
        elif is_cjk_character(character):
            grapheme_end = consume_grapheme_cluster(text, position)
            tokens.append(make_token(text[position:grapheme_end], "cjk"))
            position = grapheme_end
        elif character.isalpha():
            word_end = position + 1
            while word_end < len(text) and (
                text[word_end].isalpha() or is_grapheme_extend(text[word_end])
            ):
                word_end += 1
            tokens.append(make_token(text[position:word_end], "word"))
            position = word_end
        else:
            grapheme_end = consume_grapheme_cluster(text, position)
            grapheme = text[position:grapheme_end]
            if unicodedata.category(character) == "So":
                token_kind: TextTokenKind = "emoji"
            elif unicodedata.category(character).startswith(("P", "S", "Z", "C")):
                token_kind = "punctuation"
            else:
                token_kind = "other"
            tokens.append(make_token(grapheme, token_kind))
            position = grapheme_end
    return tokens


def extend_token_end(text: str, token_end: int) -> int:
    while token_end < len(text) and is_grapheme_extend(text[token_end]):
        token_end += 1
    return token_end


def find_keycap_end(text: str, start: int) -> int | None:
    if text[start] not in "#*0123456789":
        return None
    else:
        pass
    end = start + 1
    if end < len(text) and text[end] == "\ufe0f":
        end += 1
    else:
        pass
    if end < len(text) and text[end] == "\u20e3":
        return end + 1
    else:
        return None


def make_token(text: str, kind: TextTokenKind) -> TextToken:
    if kind in ("whitespace", "punctuation"):
        semantic_length = 0
    elif kind == "cjk":
        semantic_length = sum(is_cjk_character(character) for character in text)
    else:
        semantic_length = max(1, ceil(len(text) / ASCII_SEMANTIC_CHAR_GROUP))
    return TextToken(text=text, kind=kind, semantic_length=semantic_length)


def consume_grapheme_cluster(text: str, start: int) -> int:
    end = start + 1
    if (
        is_regional_indicator(text[start])
        and end < len(text)
        and is_regional_indicator(text[end])
    ):
        end += 1
    else:
        pass
    while end < len(text):
        if is_grapheme_extend(text[end]):
            end += 1
        elif text[end] == "\u200d" and end + 1 < len(text):
            end += 2
        else:
            break
    return end


def is_cjk_character(character: str) -> bool:
    return "\u3400" <= character <= "\u9fff"


def is_variation_selector(character: str) -> bool:
    return (
        "\ufe00" <= character <= "\ufe0f" or "\U000e0100" <= character <= "\U000e01ef"
    )


def is_grapheme_extend(character: str) -> bool:
    return (
        unicodedata.category(character).startswith("M")
        or is_variation_selector(character)
        or "\U0001f3fb" <= character <= "\U0001f3ff"
        or "\U000e0020" <= character <= "\U000e007f"
    )


def is_regional_indicator(character: str) -> bool:
    return "\U0001f1e6" <= character <= "\U0001f1ff"


def has_meaningful_content(text: str) -> bool:
    for character in text:
        category = unicodedata.category(character)
        if category[0] in ("L", "N"):
            return True
        else:
            pass
    return False


def validate_tokens(
    tokens: list[TextToken], max_fragment_chars: int, max_semantic_length: int
) -> None:
    for token in tokens:
        if len(token.text) > max_fragment_chars:
            raise TextUnbreakableError(
                f"protected token has {len(token.text)} characters; fragment limit is {max_fragment_chars}"
            )
        else:
            pass
        if token.semantic_length > max_semantic_length:
            raise TextUnbreakableError(
                "protected token exceeds the semantic fragment limit"
            )
        else:
            pass


def segment_tokens(
    tokens: list[TextToken],
    *,
    max_semantic_length: int,
    max_fragment_chars: int,
    max_fragments: int,
) -> list[str]:
    raw_fragments: list[str] = []
    fragment_start = 0
    while fragment_start < len(tokens):
        semantic_length = 0
        character_length = 0
        fit_end = fragment_start
        cursor = fragment_start
        while cursor < len(tokens):
            token = tokens[cursor]
            next_semantic_length = semantic_length + token.semantic_length
            next_character_length = character_length + len(token.text)
            if fit_end > fragment_start and (
                next_character_length > max_fragment_chars
                or next_semantic_length > max_semantic_length
            ):
                break
            elif fit_end == fragment_start and (
                next_character_length > max_fragment_chars
                or next_semantic_length > max_semantic_length
            ):
                raise TextUnbreakableError(
                    "the next atomic token cannot fit in a fragment"
                )
            else:
                semantic_length = next_semantic_length
                character_length = next_character_length
                fit_end = cursor + 1
                cursor += 1
        if fit_end == fragment_start:
            raise TextSegmentationError("no safe segment boundary was found")
        else:
            pass
        fragment_end = choose_fragment_end(
            tokens,
            fragment_start,
            fit_end,
            max_semantic_length,
        )
        raw_fragments.append(
            "".join(token.text for token in tokens[fragment_start:fragment_end])
        )
        fragment_start = fragment_end

    fragments = merge_punctuation_only_fragments(
        raw_fragments,
        max_fragment_chars,
        max_semantic_length,
    )
    if len(fragments) > max_fragments:
        fragments = coalesce_adjacent_fragments(
            fragments,
            max_fragment_chars,
            max_semantic_length,
        )
    else:
        pass
    if len(fragments) > max_fragments:
        raise TextTooLargeError(
            f"planned {len(fragments)} fragments; limit is {max_fragments}"
        )
    else:
        pass
    for fragment in fragments:
        if not has_meaningful_content(fragment):
            raise TextSegmentationError(
                "punctuation-only fragment escaped consolidation"
            )
        else:
            pass
        if not fits_fragment_limits(
            fragment,
            max_fragment_chars,
            max_semantic_length,
        ):
            raise TextSegmentationError("consolidation exceeded fragment limits")
        else:
            pass
    return fragments


def fragment_semantic_length(fragment: str) -> int:
    return sum(token.semantic_length for token in tokenize_text(fragment))


def fits_fragment_limits(
    fragment: str, max_fragment_chars: int, max_semantic_length: int
) -> bool:
    return (
        len(fragment) <= max_fragment_chars
        and fragment_semantic_length(fragment) <= max_semantic_length
    )


def coalesce_adjacent_fragments(
    fragments: list[str], max_fragment_chars: int, max_semantic_length: int
) -> list[str]:
    """Greedily merge neighboring fragments that fit both configured limits."""
    coalesced: list[str] = []
    for fragment in fragments:
        if coalesced and fits_fragment_limits(
            coalesced[-1] + fragment,
            max_fragment_chars,
            max_semantic_length,
        ):
            coalesced[-1] += fragment
        else:
            coalesced.append(fragment)
    return coalesced


def merge_punctuation_only_fragments(
    raw_fragments: list[str], max_fragment_chars: int, max_semantic_length: int
) -> list[str]:
    fragments: list[str] = []
    pending_prefix = ""
    for raw_fragment in raw_fragments:
        if has_meaningful_content(raw_fragment):
            if pending_prefix and fragments:
                preceding_fragment = fragments[-1]
                if fits_fragment_limits(
                    preceding_fragment + pending_prefix,
                    max_fragment_chars,
                    max_semantic_length,
                ):
                    fragments[-1] = preceding_fragment + pending_prefix
                    pending_prefix = ""
                else:
                    pass
            else:
                pass
            candidate_fragment = pending_prefix + raw_fragment
            if not fits_fragment_limits(
                candidate_fragment,
                max_fragment_chars,
                max_semantic_length,
            ):
                raise TextSegmentationError(
                    "punctuation-only content cannot be merged within fragment limits"
                )
            else:
                pass
            fragments.append(candidate_fragment)
            pending_prefix = ""
        else:
            pending_prefix += raw_fragment
    if pending_prefix:
        if fragments and fits_fragment_limits(
            fragments[-1] + pending_prefix,
            max_fragment_chars,
            max_semantic_length,
        ):
            fragments[-1] += pending_prefix
        elif fragments:
            raise TextSegmentationError(
                "trailing punctuation-only content exceeds fragment limits"
            )
        else:
            return []
    else:
        pass
    return fragments
