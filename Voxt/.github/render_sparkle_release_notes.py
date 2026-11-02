#!/usr/bin/env python3

import argparse
import html
import json
import os
import re


LOCALIZED_HEADING_TITLES = {
    "en": {"Added": "Added", "Changed": "Changed", "Fixed": "Fixed"},
    "zh-Hans": {"Added": "新增", "Changed": "改进", "Fixed": "修复"},
    "ja": {"Added": "追加", "Changed": "変更", "Fixed": "修正"},
}

LOCALE_SECTION_TITLES = {
    "english": "en",
    "en": "en",
    "简体中文": "zh-Hans",
    "zh-hans": "zh-Hans",
    "日本語": "ja",
    "ja": "ja",
    "japanese": "ja",
}

CATEGORY_TITLES = {
    "added": "Added",
    "changed": "Changed",
    "fixed": "Fixed",
    "新增": "Added",
    "改进": "Changed",
    "修复": "Fixed",
    "追加": "Added",
    "変更": "Changed",
    "修正": "Fixed",
}

LANGUAGE_PREFIXES = [
    ("en", "EN:"),
    ("zh-Hans", "简体中文："),
    ("zh-Hans", "简体中文:"),
    ("ja", "日本語："),
    ("ja", "日本語:"),
]


def markdown_to_html(markdown: str) -> str:
    html_lines: list[str] = []
    in_list = False

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            html_lines.append("</ul>")
            in_list = False

    for raw_line in markdown.splitlines():
        stripped = raw_line.rstrip().strip()

        if re.match(r"^## \[", stripped):
            continue

        if not stripped:
            close_list()
            continue

        if stripped.startswith("### "):
            close_list()
            html_lines.append(f"<h3>{html.escape(stripped[4:])}</h3>")
            continue

        if stripped.startswith("#### "):
            close_list()
            html_lines.append(f"<h4>{html.escape(stripped[5:])}</h4>")
            continue

        if stripped.startswith("- "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{html.escape(stripped[2:])}</li>")
            continue

        close_list()
        html_lines.append(f"<p>{html.escape(stripped)}</p>")

    close_list()
    return "\n".join(html_lines)


def heading_for_locale(locale: str, heading: str | None) -> str | None:
    if not heading:
        return None

    heading_title = heading.strip()
    localized_title = LOCALIZED_HEADING_TITLES.get(locale, {}).get(
        heading_title, heading_title
    )
    return f"### {localized_title}"


def build_localized_html(release_body: str) -> dict[str, str]:
    localized_sections: dict[str, list[list[str | None | list[str]]]] = {
        "en": [],
        "zh-Hans": [],
        "ja": [],
    }
    current_heading: str | None = None
    current_locale: str | None = None

    for raw_line in release_body.splitlines():
        stripped = raw_line.rstrip().strip()

        if re.match(r"^## \[", stripped):
            continue

        if not stripped:
            continue

        if stripped.startswith("### "):
            section_title = stripped[4:].strip()
            current_locale = LOCALE_SECTION_TITLES.get(section_title.lower())
            if current_locale:
                current_heading = None
                continue

            current_heading = CATEGORY_TITLES.get(section_title.lower())
            continue

        if stripped.startswith("#### "):
            section_title = stripped[5:].strip()
            current_heading = CATEGORY_TITLES.get(section_title.lower())
            continue

        if not stripped.startswith("- "):
            continue

        item_text = stripped[2:].strip()
        if current_locale:
            localized_heading = heading_for_locale(current_locale, current_heading)
            locale_sections = localized_sections[current_locale]
            if not locale_sections or locale_sections[-1][0] != localized_heading:
                locale_sections.append([localized_heading, []])
            locale_sections[-1][1].append(f"- {item_text}")
            continue

        for locale, prefix in LANGUAGE_PREFIXES:
            if item_text.startswith(prefix):
                localized_heading = heading_for_locale(locale, current_heading)
                locale_sections = localized_sections[locale]
                if not locale_sections or locale_sections[-1][0] != localized_heading:
                    locale_sections.append([localized_heading, []])
                locale_sections[-1][1].append(f"- {item_text[len(prefix):].strip()}")
                break

    return {
        locale: markdown_to_html(
            "\n".join(
                line
                for heading, items in sections
                for line in ([heading] if heading else []) + items
            )
        )
        for locale, sections in localized_sections.items()
        if sections
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--localized-output", required=True)
    args = parser.parse_args()

    release_body = os.environ.get("RELEASE_BODY", "")
    localized_html = build_localized_html(release_body)

    with open(args.localized_output, "w", encoding="utf-8") as handle:
        json.dump(localized_html, handle, ensure_ascii=False, separators=(",", ":"))

    print(markdown_to_html(release_body), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
