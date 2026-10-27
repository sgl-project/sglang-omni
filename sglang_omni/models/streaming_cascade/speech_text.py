"""Limited inline speech cleanup, retaining original reply offsets."""

import re
from bisect import bisect_left

import marko
from marko import block, inline
from pydantic import BaseModel, ConfigDict, Field, model_validator

UNSPOKEN_SYMBOLS = re.compile(
    "[\U0001f000-\U0001faff\u2600-\u27bf\u2b50\u2b55\ufe0e\ufe0f\u200d\u20e3]"
)


class SpeechSpan(BaseModel):
    """One speech request before numeral expansion, with absolute reply character ends."""

    model_config = ConfigDict(extra="forbid", strict=True)
    text: str
    source_start_chars: int = Field(ge=0)
    source_end_chars: int = Field(ge=0)
    original_ends_chars: list[int]
    language: str | None = None

    @model_validator(mode="after")
    def check_offsets(self) -> "SpeechSpan":
        if (
            self.source_end_chars < self.source_start_chars
            or len(self.original_ends_chars) != len(self.text)
            or self.original_ends_chars != sorted(self.original_ends_chars)
            or any(
                not self.source_start_chars < end_chars <= self.source_end_chars
                for end_chars in self.original_ends_chars
            )
        ):
            raise ValueError(
                "speech character ends must map monotonically into the source span"
            )
        else:
            return self


def speech_span(text: str, source_start_chars: int) -> SpeechSpan:
    """Remove supported inline syntax without searching rendered words in the source."""
    characters: list[str] = []
    original_ends_chars: list[int] = []
    for match in re.finditer(r"\r\n|\r|[^\r]", text, re.DOTALL):
        characters.append("\n" if match.group().startswith("\r") else match.group())
        original_ends_chars.append(source_start_chars + match.end())
    normalized = "".join(characters)
    removed = [False] * len(normalized)
    closing_spans: list[tuple[int, int]] = []
    pending: list[block.BlockElement | inline.InlineElement] = [marko.parse(normalized)]
    while pending:
        element = pending.pop()
        if isinstance(
            element,
            (
                inline.Emphasis,
                inline.StrongEmphasis,
                inline.CodeSpan,
                inline.Link,
                inline.Literal,
            ),
        ):
            for start_chars, end_chars in element.syntax_spans or ():
                for index in range(start_chars, min(end_chars, len(removed))):
                    removed[index] = True
                if (
                    element.source_span is not None
                    and end_chars == element.source_span[1]
                ):
                    closing_spans.append((start_chars, end_chars))
                else:
                    pass
        else:
            pass
        if isinstance(element.children, str):
            pass
        else:
            pending.extend(element.children)
    kept_characters: list[str] = []
    kept_ends_chars: list[int] = []
    kept_indices: list[int] = []
    for index, character in enumerate(normalized):
        if removed[index]:
            pass
        elif UNSPOKEN_SYMBOLS.fullmatch(character):
            if kept_ends_chars:
                kept_ends_chars[-1] = original_ends_chars[index]
            else:
                pass
        else:
            kept_characters.append(character)
            kept_ends_chars.append(original_ends_chars[index])
            kept_indices.append(index)
    for start_chars, end_chars in closing_spans:
        preceding_index = bisect_left(kept_indices, start_chars) - 1
        if preceding_index >= 0:
            kept_ends_chars[preceding_index] = max(
                kept_ends_chars[preceding_index], original_ends_chars[end_chars - 1]
            )
        else:
            pass
    words: list[str] = []
    speech_ends_chars: list[int] = []
    for match in re.finditer(r"\S+", "".join(kept_characters)):
        if words:
            speech_ends_chars.append(kept_ends_chars[match.start() - 1])
        else:
            pass
        words.append(match.group())
        speech_ends_chars.extend(kept_ends_chars[match.start() : match.end()])
    return SpeechSpan(
        text=" ".join(words),
        source_start_chars=source_start_chars,
        source_end_chars=source_start_chars + len(text),
        original_ends_chars=speech_ends_chars,
    )
