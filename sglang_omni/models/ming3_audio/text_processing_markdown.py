# SPDX-License-Identifier: Apache-2.0
"""Conservative Markdown cleanup for Ming speech text.

Markdown link labels are spoken and bare HTTP(S) URLs are replaced by the short
spoken word link so generated responses cannot fail only because a URL is long.
"""

from __future__ import annotations

import re

from sglang_omni.models.ming3_audio.text_processing_contract import UnsupportedTextError

URL_PATTERN = re.compile(r"https?://(?:\[[^\[\]\s<>]+\]|[^\s<>\[\]{}])+", re.IGNORECASE)
AUTOLINK_PATTERN = re.compile(r"<(https?://[^\s<>]+)>", re.IGNORECASE)
AUTOLINK_START_PATTERN = re.compile(r"<https?://", re.IGNORECASE)
FENCE_PATTERN = re.compile(r"^\s*(`{3,}|~{3,})(?:\s*[A-Za-z0-9_+.-]+)?\s*$")
FENCE_CLOSE_PATTERN = re.compile(r"^\s*(`{3,}|~{3,})\s*$")
HEADING_PREFIX_PATTERN = re.compile(r"^\s{0,3}#{1,6}(?:\s+|$)")
HEADING_SUFFIX_PATTERN = re.compile(r"\s+#{1,6}\s*$")
LIST_PREFIX_PATTERN = re.compile(r"^(?P<indent>\s*)(?:[-+*])\s+")
HORIZONTAL_RULE_PATTERN = re.compile(r"^(?:[-*_]\s*){3,}$")
LINK_START_PATTERN = re.compile(
    r"(?P<image>!?)\[(?P<label>(?:[^\[\]\n]|\[[^\[\]\n]*\])*)\]\("
)
LINK_DESTINATION_START_PATTERN = re.compile(r"\]\(\s*<?https?://", re.IGNORECASE)
REFERENCE_LINK_PATTERN = re.compile(r"\[[^\]\n]*\]\[[^\]\n]*\]")
HTML_TAG_PATTERN = re.compile(r"</?[A-Za-z][^>\n]*>")
INLINE_CODE_PATTERN = re.compile(r"`([^`\n]+)`")
EMPHASIS_PATTERNS = (
    re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*"),
    re.compile(r"__(?=\S)(.+?)(?<=\S)__"),
    re.compile(r"~~(?=\S)(.+?)(?<=\S)~~"),
    re.compile(r"(?<!\w)\*(?=\S)(.+?)(?<=\S)\*(?!\w)"),
    re.compile(r"(?<!\w)_(?=\S)(.+?)(?<=\S)_(?!\w)"),
)
SPEAKABLE_URL_REPLACEMENT = "link"
TRAILING_URL_PUNCTUATION = ".,!?:;，。！？；：、\"'”’»"


def clean_markdown(raw_text: str) -> str:
    """Remove supported presentation syntax while preserving visible text."""
    normalized_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines: list[str] = []
    inside_code_block = False
    fence_character = ""
    fence_length = 0
    for line in normalized_text.split("\n"):
        fence_match = FENCE_PATTERN.fullmatch(line)
        if inside_code_block:
            closing_fence_match = FENCE_CLOSE_PATTERN.fullmatch(line)
            if (
                closing_fence_match is not None
                and closing_fence_match.group(1)[0] == fence_character
                and len(closing_fence_match.group(1)) >= fence_length
            ):
                inside_code_block = False
                fence_character = ""
                fence_length = 0
            else:
                code_line = normalize_bare_urls(replace_autolinks(line.strip()))
                if code_line:
                    cleaned_lines.append(code_line)
                else:
                    pass
        elif fence_match is not None:
            inside_code_block = True
            fence_character = fence_match.group(1)[0]
            fence_length = len(fence_match.group(1))
        else:
            cleaned_line = clean_markdown_line(line)
            if cleaned_line or not line.strip():
                cleaned_lines.append(cleaned_line)
            else:
                pass
    if inside_code_block:
        raise UnsupportedTextError("input contains an unclosed code fence")
    else:
        pass

    return "\n".join(cleaned_lines).strip()


def replace_autolinks(line: str) -> str:
    return AUTOLINK_PATTERN.sub(r"\1", line)


def normalize_bare_urls(text: str) -> str:
    protected_parts: list[str] = []
    source_end = 0
    for url_match in URL_PATTERN.finditer(text):
        candidate = url_match.group(0)
        url_text, trailing_text = split_url_suffix(candidate)
        if url_text:
            protected_parts.append(text[source_end : url_match.start()])
            protected_parts.append(SPEAKABLE_URL_REPLACEMENT)
            protected_parts.append(trailing_text)
            source_end = url_match.end()
        else:
            pass
    protected_parts.append(text[source_end:])
    return "".join(protected_parts)


def split_url_suffix(candidate: str) -> tuple[str, str]:
    url_end = len(candidate)
    while url_end > 0 and candidate[url_end - 1] in TRAILING_URL_PUNCTUATION:
        url_end -= 1
    while (
        url_end > 0
        and candidate[url_end - 1] == ")"
        and candidate[:url_end].count(")") > candidate[:url_end].count("(")
    ):
        url_end -= 1
    while (
        url_end > 0
        and candidate[url_end - 1] == "]"
        and candidate[:url_end].count("]") > candidate[:url_end].count("[")
    ):
        url_end -= 1
    return candidate[:url_end], candidate[url_end:]


def clean_markdown_line(line: str) -> str:
    line = replace_autolinks(line)
    if AUTOLINK_START_PATTERN.search(line) is not None:
        raise UnsupportedTextError("malformed HTTP(S) autolink")
    elif HTML_TAG_PATTERN.search(line) is not None:
        raise UnsupportedTextError("HTML markup is not supported")
    elif REFERENCE_LINK_PATTERN.search(line) is not None:
        raise UnsupportedTextError("reference-style links are not supported")
    else:
        pass

    heading_match = HEADING_PREFIX_PATTERN.match(line)
    line = HEADING_PREFIX_PATTERN.sub("", line, count=1)
    if heading_match is None:
        line = LIST_PREFIX_PATTERN.sub(r"\g<indent>", line, count=1)
    else:
        pass
    line = HORIZONTAL_RULE_PATTERN.sub("", line, count=1)
    line = replace_links(line)
    line = INLINE_CODE_PATTERN.sub(r"\1", line)
    if "`" in line:
        raise UnsupportedTextError("unpaired inline code delimiter")
    else:
        pass
    for emphasis_pattern in EMPHASIS_PATTERNS:
        line = emphasis_pattern.sub(r"\1", line)
    if heading_match is not None:
        line = HEADING_SUFFIX_PATTERN.sub("", line)
    else:
        pass
    line = normalize_bare_urls(line)
    leading_match = re.match(r"^[ \t]*", line)
    assert leading_match is not None
    leading_whitespace = leading_match.group(0)
    return (
        leading_whitespace
        + re.sub(r"[ \t]+", " ", line[len(leading_whitespace) :]).rstrip()
    )


def replace_links(line: str) -> str:
    replacement_parts: list[str] = []
    source_start = 0
    search_start = 0
    while True:
        link_match = LINK_START_PATTERN.search(line, search_start)
        if link_match is None:
            break
        else:
            pass
        link_end = find_link_end(line, link_match.end())
        if link_end is None:
            raise UnsupportedTextError("malformed Markdown link")
        else:
            pass
        target_text = line[link_match.end() : link_end].strip()
        validate_link_target(target_text)
        replacement_parts.append(line[source_start : link_match.start()])
        replacement_parts.append(link_match.group("label"))
        source_start = link_end + 1
        search_start = source_start
    if source_start == 0:
        replacement_line = line
    else:
        replacement_parts.append(line[source_start:])
        replacement_line = "".join(replacement_parts)
    if re.search(r"!\[|\[[^\]\n]*\]\(", replacement_line) is not None:
        raise UnsupportedTextError("malformed Markdown link")
    elif LINK_DESTINATION_START_PATTERN.search(replacement_line) is not None:
        raise UnsupportedTextError("malformed Markdown link")
    else:
        pass
    return replacement_line


def find_link_end(line: str, target_start: int) -> int | None:
    parenthesis_depth = 1
    character_position = target_start
    while character_position < len(line):
        character = line[character_position]
        if character == "\\":
            if character_position + 1 < len(line):
                character_position += 2
            else:
                character_position += 1
        elif character == "(":
            parenthesis_depth += 1
            character_position += 1
        elif character == ")":
            parenthesis_depth -= 1
            if parenthesis_depth == 0:
                return character_position
            else:
                character_position += 1
        else:
            character_position += 1
    return None


def validate_link_target(target_text: str) -> None:
    target_match = re.fullmatch(
        r"(?:<(?P<angle_destination>[^<>]+)>|(?P<plain_destination>\S+))"
        r"(?:\s+(?P<title>.*))?",
        target_text,
    )
    if target_match is None:
        raise UnsupportedTextError("malformed Markdown link target")
    else:
        pass
    destination = target_match.group("angle_destination")
    if destination is None:
        destination = target_match.group("plain_destination")
    else:
        pass
    title = target_match.group("title")
    if title is not None and not has_valid_markdown_title(title):
        raise UnsupportedTextError("malformed Markdown link title")
    elif not destination:
        raise UnsupportedTextError("link target is empty")
    elif URL_PATTERN.fullmatch(destination) is None:
        raise UnsupportedTextError("only HTTP(S) Markdown links are supported")
    else:
        pass


def has_valid_markdown_title(title: str) -> bool:
    if len(title) < 2:
        return False
    else:
        pass
    if title[0] in "\"'":
        if title[-1] != title[0]:
            return False
        else:
            pass
        escaped = False
        for character in title[1:-1]:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == title[0]:
                return False
            else:
                pass
        return not escaped
    elif title[0] == "(":
        depth = 0
        escaped = False
        for index, character in enumerate(title):
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0 and index != len(title) - 1:
                    return False
                else:
                    pass
            else:
                pass
        return depth == 0 and not escaped
    else:
        return False


def has_presentation_markup(raw_text: str) -> bool:
    normalized_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    for line in normalized_text.split("\n"):
        if (
            FENCE_PATTERN.fullmatch(line) is not None
            or HEADING_PREFIX_PATTERN.match(line) is not None
            or LIST_PREFIX_PATTERN.match(line) is not None
            or HORIZONTAL_RULE_PATTERN.fullmatch(line.strip()) is not None
        ):
            return True
        else:
            pass
    return bool(
        HTML_TAG_PATTERN.search(normalized_text)
        or LINK_START_PATTERN.search(normalized_text)
        or AUTOLINK_PATTERN.search(normalized_text)
        or URL_PATTERN.search(normalized_text)
        or REFERENCE_LINK_PATTERN.search(normalized_text)
        or INLINE_CODE_PATTERN.search(normalized_text)
        or re.search(r"`|~~|\*\*|__", normalized_text) is not None
        or re.search(r"(?<!\w)\*(?=\S).+?(?<=\S)\*(?!\w)", normalized_text) is not None
        or re.search(r"(?<!\w)_(?=\S).+?(?<=\S)_(?!\w)", normalized_text) is not None
    )
