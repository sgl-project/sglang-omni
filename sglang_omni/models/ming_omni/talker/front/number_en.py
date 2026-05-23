import re

try:
    import inflect

    _inflect = inflect.engine()
    _HAS_INFLECT = True
except ImportError:
    _inflect = None
    _HAS_INFLECT = False

_comma_number_re = re.compile(r"([0-9][0-9,]+[0-9])")
_percent_number_re = re.compile(r"(-?[0-9.,]*[0-9]+)%")
_pounds_re = re.compile(r"\u00a3(-?[0-9,]*[0-9]+(?:\.[0-9]+)?)")
_dollars_re = re.compile(r"\$(-?[0-9.,]*[0-9]+(?:\.[0-9]+)?)")
_fraction_re = re.compile(r"([0-9]+)\/([0-9]+)")
_ordinal_re = re.compile(r"\b[0-9]+(st|nd|rd|th)\b")
_number_re = re.compile(r"\b-?[0-9]+(?:\.[0-9]+)?\b")
_unit_re = re.compile(
    r"\b(-?\d+(?:\.\d+)?)\s*(ms|s|Hz|kHz|MHz|GHz|kb|mb|gb|tb|KB|MB|GB|TB|bps|kbps|Mbps|Gbps|cm|km|kg|V|A|W|\u00b0C|\u00b0F)\b",
    re.IGNORECASE,
)
_version_re = re.compile(r"\b([a-zA-Z]+)([-]?)([0-9]+(?:\.[0-9]+)?)\b")
_whitespace_re = re.compile(r"\s+")

_unit_mapping = {
    "ms": "milliseconds",
    "s": "seconds",
    "hz": "hertz",
    "khz": "kilohertz",
    "mhz": "megahertz",
    "ghz": "gigahertz",
    "kb": "kilobytes",
    "mb": "megabytes",
    "gb": "gigabytes",
    "tb": "terabytes",
    "kbps": "kilobits per second",
    "mbps": "megabits per second",
    "gbps": "gigabits per second",
    "bps": "bits per second",
    "cm": "centimeters",
    "km": "kilometers",
    "kg": "kilograms",
    "v": "volts",
    "a": "amperes",
    "w": "watts",
    "\u00b0c": "degrees celsius",
    "\u00b0f": "degrees fahrenheit",
}


def _num_to_words(num_str):
    """Convert a number string to words. Returns None on failure."""
    if not _HAS_INFLECT:
        return None
    try:
        is_negative = num_str.startswith("-")
        clean_num = num_str.lstrip("-") or "0"

        if "." in clean_num:
            parts = clean_num.split(".", 1)
            integer_part = parts[0] or "0"
            decimal_part = parts[1]
            if not integer_part.isdigit() or not decimal_part.isdigit():
                return None
            integer_word = (
                _inflect.number_to_words(int(integer_part), andword="")
                if integer_part != "0"
                else "zero"
            )
            decimal_words = " ".join(
                _inflect.number_to_words(int(d), andword="")
                for d in decimal_part
                if d.isdigit()
            )
            num_word = f"{integer_word} point {decimal_words}"
        else:
            if not clean_num.isdigit():
                return None
            num_word = _inflect.number_to_words(int(clean_num), andword="")

        if is_negative:
            num_word = f"minus {num_word}"
        return num_word
    except Exception:
        return None


def _remove_commas(m):
    return m.group(1).replace(",", "")


def _expand_unit(m):
    num_str, unit = m.group(1), m.group(2).lower()
    unit_word = _unit_mapping.get(unit, unit)
    word = _num_to_words(num_str)
    return f" {word} {unit_word} " if word else f" {num_str} {unit} "


def _expand_percent(m):
    word = _num_to_words(m.group(1))
    return f" {word} percent " if word else f" {m.group(1)} percent "


def _expand_dollars(m):
    match = m.group(1)
    word = _num_to_words(match)
    if word:
        clean = match.lstrip("-") or "0"
        unit = "dollar" if abs(float(clean)) == 1.0 else "dollars"
        return f" {word} {unit} "
    return f" {match} dollars "


def _expand_pounds(m):
    num_str = m.group(1)
    word = _num_to_words(num_str)
    if word:
        clean = num_str.lstrip("-") or "0"
        unit = "pound" if abs(float(clean)) == 1.0 else "pounds"
        return f" {word} {unit} "
    return f" {num_str} pounds "


def _expand_fraction(m):
    if not _HAS_INFLECT:
        return m.group(0)
    try:
        n, d = int(m.group(1)), int(m.group(2))
        if n == 1 and d == 2:
            return " one half "
        if n == 1 and d == 4:
            return " one quarter "
        ordinal = _inflect.ordinal(_inflect.number_to_words(d))
        return f" {_inflect.number_to_words(n)} {ordinal} "
    except Exception:
        return m.group(0)


def _expand_ordinal(m):
    if not _HAS_INFLECT:
        return m.group(0)
    try:
        num = int(re.sub(r"(st|nd|rd|th)", "", m.group(0)))
        return f" {_inflect.number_to_words(num)} "
    except Exception:
        return m.group(0)


def _expand_number(m):
    word = _num_to_words(m.group(0))
    return f" {word} " if word else f" {m.group(0)} "


def _expand_version(m):
    if not _HAS_INFLECT:
        return m.group(0)
    prefix, sep, num_str = m.group(1), m.group(2), m.group(3)
    try:
        if "." in num_str:
            parts = num_str.split(".", 1)
            if not parts[0].isdigit() or not parts[1].isdigit():
                return m.group(0)
            integer_word = _inflect.number_to_words(int(parts[0]))
            decimal_words = " ".join(
                _inflect.number_to_words(int(d)) for d in parts[1] if d.isdigit()
            )
            word = f"{integer_word} point {decimal_words}"
        else:
            if not num_str.isdigit():
                return m.group(0)
            word = _inflect.number_to_words(int(num_str))
    except Exception:
        return m.group(0)
    return f"{prefix} {word}"


def normalize_numbers(text):
    if not _HAS_INFLECT:
        return text
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_unit_re, _expand_unit, text)
    text = re.sub(_pounds_re, _expand_pounds, text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_fraction_re, _expand_fraction, text)
    text = re.sub(_percent_number_re, _expand_percent, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_version_re, _expand_version, text)
    text = re.sub(_number_re, _expand_number, text)
    text = re.sub(_whitespace_re, " ", text)
    return text.strip()
