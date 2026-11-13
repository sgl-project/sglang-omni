# SPDX-License-Identifier: Apache-2.0
"""Numbers read out as words, the same way in the text the speech service reads and in the text the forced aligner aligns."""

import re

NUMERAL_PATTERN = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
QUANTITY_UNIT_PATTERN = (
    r"seconds?|minutes?|hours?|days?|weeks?|months?|years?|"
    r"millimeters?|millimetres?|centimeters?|centimetres?|meters?|metres?|"
    r"kilometers?|kilometres?|miles?|inch(?:es)?|feet|grams?|kilograms?|pounds?|"
    r"dollars?|euros?|percent|degrees?"
)
NUMBER = re.compile(rf"{NUMERAL_PATTERN}(?:(?P<ordinal_suffix>st|nd|rd|th)\b)?")
ENGLISH_NUMBER = re.compile(
    rf"(?<![\w.+\-\u2013])(?P<range_start>(?!0\d){NUMERAL_PATTERN})"
    rf"\s*(?P<range_separator>[-\u2013])\s*"
    rf"(?P<range_end>(?!0\d){NUMERAL_PATTERN})(?=\s+(?i:{QUANTITY_UNIT_PATTERN})\b)"
    rf"|{NUMBER.pattern}"
)
ADJACENT_WORD = re.compile(r"\S*\w")
GROUP_BEFORE = re.compile(r"\d \Z")
GROUP_AFTER = re.compile(r"\A \d")
CHINESE_DIGITS = "零一二三四五六七八九"
CHINESE_UNITS = ("", "十", "百", "千")
ENGLISH_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
ENGLISH_TENS = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)
ENGLISH_SCALES = ((10**6, "million"), (1000, "thousand"))
ENGLISH_ORDINALS = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}
MAX_CARDINAL_DIGITS = 9
FIRST_SPOKEN_YEAR = 1500
LAST_SPOKEN_YEAR = 2099


def chinese_numeral(digits: str) -> str:
    """The digits read out in Chinese: a number up to 9999 with its place units, anything longer digit by digit."""
    if len(digits) > 4:
        return "".join(CHINESE_DIGITS[int(digit)] for digit in digits)
    elif int(digits) == 0:
        return CHINESE_DIGITS[0]
    else:
        spoken = ""
        for position, digit in enumerate(reversed(digits)):
            if digit == "0":
                spoken = (
                    "零" if spoken and not spoken.startswith("零") else ""
                ) + spoken
            elif position == 1 and digit == "1" and len(digits) == 2:
                spoken = "十" + spoken
            else:
                spoken = CHINESE_DIGITS[int(digit)] + CHINESE_UNITS[position] + spoken
        return spoken.rstrip("零")


def english_cardinal(number: int) -> str:
    """The number, below one billion, in English words."""
    if number < 20:
        return ENGLISH_ONES[number]
    elif number < 100:
        tens, ones = divmod(number, 10)
        return (
            ENGLISH_TENS[tens]
            if ones == 0
            else f"{ENGLISH_TENS[tens]} {ENGLISH_ONES[ones]}"
        )
    elif number < 1000:
        hundreds, remainder = divmod(number, 100)
        hundred_words = f"{ENGLISH_ONES[hundreds]} hundred"
        return (
            hundred_words
            if remainder == 0
            else f"{hundred_words} {english_cardinal(remainder)}"
        )
    else:
        scale, scale_name = next(
            (scale, scale_name)
            for scale, scale_name in ENGLISH_SCALES
            if number >= scale
        )
        scale_count, remainder = divmod(number, scale)
        scaled_words = f"{english_cardinal(scale_count)} {scale_name}"
        return (
            scaled_words
            if remainder == 0
            else f"{scaled_words} {english_cardinal(remainder)}"
        )


def english_ordinal(number: int) -> str:
    """The number as an English ordinal, such as twenty first."""
    *leading_words, last_word = english_cardinal(number).split()
    if last_word in ENGLISH_ORDINALS:
        ordinal_word = ENGLISH_ORDINALS[last_word]
    elif last_word.endswith("y"):
        ordinal_word = f"{last_word[:-1]}ieth"
    else:
        ordinal_word = f"{last_word}th"
    return " ".join([*leading_words, ordinal_word])


def english_year(number: int) -> str:
    """A year as two pairs: nineteen oh five, nineteen hundred, twenty twenty four."""
    century, year_in_century = divmod(number, 100)
    if year_in_century == 0:
        return f"{english_cardinal(century)} hundred"
    elif year_in_century < 10:
        return f"{english_cardinal(century)} oh {english_cardinal(year_in_century)}"
    else:
        return f"{english_cardinal(century)} {english_cardinal(year_in_century)}"


def english_numeral(digits: str, is_grouped: bool) -> str:
    """The digits read out in English.

    Digit by digit in a group of a longer number such as a phone number, with
    a leading zero or beyond nine digits; a year from 1500 to 2099 as two
    pairs, except 2000 to 2009; anything else as a cardinal.
    """
    number = int(digits)
    if (
        is_grouped
        or len(digits) > MAX_CARDINAL_DIGITS
        or (len(digits) > 1 and digits.startswith("0"))
    ):
        return " ".join(ENGLISH_ONES[int(digit)] for digit in digits)
    elif len(digits) == 4 and FIRST_SPOKEN_YEAR <= number <= LAST_SPOKEN_YEAR:
        return (
            english_cardinal(number) if 2000 <= number < 2010 else english_year(number)
        )
    else:
        return english_cardinal(number)


def number_words(
    numeral: str, language: str, is_grouped: bool, ordinal_suffix: str | None
) -> str:
    """The numeral, a digit run with optional thousands separators or a decimal, read out in the language; another language keeps its digits."""
    whole, point, fraction = numeral.replace(",", "").partition(".")
    if language == "Chinese":
        words = chinese_numeral(whole)
        fraction_words = "".join(CHINESE_DIGITS[int(digit)] for digit in fraction)
        return words if not point else f"{words}点{fraction_words}"
    elif language == "English" and ordinal_suffix is not None and not point:
        return english_ordinal(int(whole))
    elif language == "English":
        words = english_numeral(whole, is_grouped)
        fraction_words = " ".join(ENGLISH_ONES[int(digit)] for digit in fraction)
        return words if not point else f"{words} point {fraction_words}"
    else:
        return numeral


def spoken_text(text: str, language: str) -> tuple[str, list[int]]:
    """The text with each number read out as words, and for each spoken character the original character it stands for.

    English quantity ranges use to; digit runs one space apart are groups of a longer number, such as a phone number.
    """
    spoken: list[str] = []
    origin: list[int] = []
    cursor = 0
    number_pattern = ENGLISH_NUMBER if language == "English" else NUMBER
    for match in number_pattern.finditer(text):
        before, after = text[: match.start()], text[match.end() :]
        is_grouped = bool(GROUP_BEFORE.search(before) or GROUP_AFTER.match(after))
        if language == "English" and match.group("range_start") is not None:
            start_words = number_words(
                match.group("range_start"), language, False, None
            )
            end_words = number_words(match.group("range_end"), language, False, None)
            words = f"{start_words} to {end_words}"
            word_origin = (
                [match.end("range_start") - 1] * len(start_words)
                + [match.start("range_separator")] * len(" to ")
                + [match.end("range_end") - 1] * len(end_words)
            )
        else:
            ordinal_suffix = match.group("ordinal_suffix")
            numeral = match.group()[: -len(ordinal_suffix) if ordinal_suffix else None]
            words = number_words(numeral, language, is_grouped, ordinal_suffix)
            word_origin = [match.end() - 1] * len(words)
        # Note (Junnan Li): Number words joined to their neighbours without a space would read as one token to the aligner.
        if before and not before[-1].isspace():
            words = f" {words}"
            word_origin.insert(0, word_origin[0])
        else:
            pass
        if ADJACENT_WORD.match(after):
            words = f"{words} "
            word_origin.append(word_origin[-1])
        else:
            pass
        spoken.append(text[cursor : match.start()])
        origin.extend(range(cursor, match.start()))
        spoken.append(words)
        origin.extend(word_origin)
        cursor = match.end()
    spoken.append(text[cursor:])
    origin.extend(range(cursor, len(text)))
    return "".join(spoken), origin
