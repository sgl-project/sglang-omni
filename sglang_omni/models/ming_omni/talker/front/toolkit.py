# Vendored from Ming/front/toolkit.py
import re
from typing import Iterable

TOKENIZE_PATTERN = (
    r"(?:[a-zA-Z]\.)+|[a-zA-Z]+(?:['\-][a-zA-Z]+)*|\d+(?:\.\d+)?|[\u4e00-\u9fff]|\s+|\S"
)


def tokenize_mixed_text(text: str):
    return re.findall(TOKENIZE_PATTERN, text)


def tokenize_mixed_text_iterator(text_iterator: Iterable[str]):
    for chunk in text_iterator:
        for match in re.finditer(TOKENIZE_PATTERN, chunk):
            yield match.group(0)
