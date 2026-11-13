# SPDX-License-Identifier: Apache-2.0
"""How much of a known text a transcript of its speech says, as a strict prefix of that text."""

import re
from dataclasses import dataclass

ENGLISH_NUMBERS: tuple[str, ...] = tuple(
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
)
CHINESE_DIGITS: dict[str, int] = dict(
    zip("零〇一二两三四五六七八九", (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9))
)
CHINESE_MAGNITUDES: dict[str, int] = {"十": 10, "百": 100, "千": 1000}
TEXT_UNITS = re.compile(
    r"\d+(?:[,.\-\u2013\u2014]\d+)*|[零〇一二两三四五六七八九十百千]+|"
    r"[a-zA-Z]+(?:[-.\u2010-\u2015’'/][a-zA-Z]+)*|[^\W_]"
)
WORD_JOINER = re.compile(r"[-.\u2010-\u2015’'/]")
TRAILING_PUNCTUATION = re.compile(r"[^\w\s]*")


@dataclass(frozen=True)
class NormalizedUnit:
    text: str
    end_chars: int
    has_joiner: bool = False
    is_complete: bool = True


def chinese_number(token: str) -> str:
    """The value of an ordinary Chinese number below ten thousand such as 三百零五; the token itself when it is not one."""
    number = digit = 0
    previous_magnitude = 10000
    after_digit = False
    is_number = True
    for char in token:
        if char in CHINESE_DIGITS:
            is_number &= not after_digit or digit == 0
            digit = CHINESE_DIGITS[char]
            after_digit = True
        else:
            magnitude = CHINESE_MAGNITUDES[char]
            is_number &= magnitude < previous_magnitude and (
                after_digit or (char == "十" and number == 0)
            )
            number += (digit or 1) * magnitude
            digit = 0
            previous_magnitude = magnitude
            after_digit = False
    if is_number:
        return str(number + digit)
    else:
        return token


def normalized_number(token: str) -> str:
    """Use English words through twenty and digits for larger numeric units."""
    if token.isdecimal():
        number = int(token)
        return ENGLISH_NUMBERS[number] if number < len(ENGLISH_NUMBERS) else str(number)
    else:
        return token


def normalized_units(text: str) -> list[NormalizedUnit]:
    """Comparable units with original offsets and explicit joining permission."""
    units: list[NormalizedUnit] = []
    for match in TEXT_UNITS.finditer(text):
        token = match.group().casefold()
        if token.isdecimal():
            units.append(NormalizedUnit(normalized_number(token), match.end()))
        elif all(char in CHINESE_DIGITS for char in token):
            units.extend(
                NormalizedUnit(
                    normalized_number(str(CHINESE_DIGITS[char])),
                    match.start() + index + 1,
                )
                for index, char in enumerate(token)
            )
        elif (
            token.endswith("十")
            and len(token) > 2
            and all(char in CHINESE_DIGITS for char in token[:-1])
            and [CHINESE_DIGITS[char] for char in token[:-1]]
            == list(range(CHINESE_DIGITS[token[0]], 10))
        ):
            units.extend(
                NormalizedUnit(
                    normalized_number(str(CHINESE_DIGITS[char])),
                    match.start() + index + 1,
                )
                for index, char in enumerate(token[:-1])
            )
            units.append(NormalizedUnit("ten", match.end()))
        elif all(
            char in CHINESE_DIGITS or char in CHINESE_MAGNITUDES for char in token
        ):
            units.append(
                NormalizedUnit(normalized_number(chinese_number(token)), match.end())
            )
        else:
            has_joiner = bool(WORD_JOINER.search(token)) and token[0].isalpha()
            units.append(
                NormalizedUnit(
                    WORD_JOINER.sub("", token) if has_joiner else token,
                    match.end(),
                    has_joiner,
                )
            )
    return units


def spoken_prefix_end(text: str, transcript: str) -> int:
    """The longest spoken prefix's character end, including following punctuation.

    Matching tolerates joined words, case and number spellings, but not omissions.
    """
    end_chars = matched_prefix_end(normalized_units(text), normalized_units(transcript))
    return TRAILING_PUNCTUATION.match(text, end_chars).end() if end_chars else 0


def matched_prefix_end(
    source_units: list[NormalizedUnit], heard_units: list[NormalizedUnit]
) -> int:
    """Match units in order, committing only at complete boundaries on both sides."""
    source_unit_index = heard_unit_index = 0
    source_text = heard_text = ""
    end_chars = 0
    while source_unit_index < len(source_units) and heard_unit_index < len(heard_units):
        source = source_units[source_unit_index]
        heard = heard_units[heard_unit_index]
        source_text = source_text or source.text
        heard_text = heard_text or heard.text
        if source_text == heard_text:
            if source.is_complete and heard.is_complete:
                end_chars = source.end_chars
            else:
                pass
            source_unit_index += 1
            heard_unit_index += 1
            source_text = heard_text = ""
        elif heard_text.startswith(source_text) and heard.has_joiner:
            source_unit_index += 1
            if source_unit_index < len(source_units):
                source_text += source_units[source_unit_index].text
            else:
                break
        elif source_text.startswith(heard_text) and source.has_joiner:
            heard_unit_index += 1
            if heard_unit_index < len(heard_units):
                heard_text += heard_units[heard_unit_index].text
            else:
                break
        else:
            break
    return end_chars
