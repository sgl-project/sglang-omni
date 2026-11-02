# SPDX-License-Identifier: Apache-2.0
"""Summarize replies, judgments and live turns as a markdown report.

Length is in words for English and, for Chinese, in Chinese characters plus each run of Latin letters or digits. The comparison against a baseline prompt resamples whole conversations, since the turns of one conversation share its history.
"""

from __future__ import annotations

import collections
import math
import random
import re
import statistics
from pathlib import Path
from typing import Literal, get_args

from benchmarks.cascade_replies.judge import (
    Axis,
    AxisVerdict,
    JudgedReply,
    LengthVerdict,
    Verdicts,
)
from benchmarks.cascade_replies.live import SpokenTurn
from benchmarks.cascade_replies.replies import (
    CONVERSATIONS_PATH,
    ConversationSet,
    Language,
    PromptName,
    ReplyRow,
    read_reply_rows,
)

LANGUAGES: tuple[Language, ...] = get_args(Language)
LENGTH_VERDICTS: tuple[LengthVerdict, ...] = get_args(LengthVerdict)
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0

EN_WORD = re.compile(r"[A-Za-z0-9]+(?:['.,][A-Za-z0-9]+)*")
ZH_CHARACTER = re.compile("[一-鿿]")
LATIN_OR_DIGIT_RUN = re.compile(r"[A-Za-z0-9]+(?:[.,][0-9]+)*")
# note (Haoling Pu): text a listener would hear read out as symbols: markdown, list lines, table pipes, URLs, emoji, formula signs and code.
SYMBOLS = re.compile(
    r"\*\*|__|`|^[ \t]*#{1,6}[ \t]|^[ \t]*(?:[-*•+]|\d+[.)、])[ \t]*\S|\||https?://|www\."
    r"|[\U0001f000-\U0001faff☀-➿]|[=^×÷√²³]|[{};]|\bdef |\bimport |\w\(\)",
    re.MULTILINE,
)

LengthBand = Literal["under", "in", "over"]


def length_units(text: str, language: Language) -> int:
    if language == "en":
        return len(EN_WORD.findall(text))
    else:
        return len(ZH_CHARACTER.findall(text)) + len(LATIN_OR_DIGIT_RUN.findall(text))


def share(count: int, total: int) -> str:
    return f"{100 * count / total:.0f}%" if total else "-"


def nearest_rank(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[max(math.ceil(fraction * len(ordered)) - 1, 0)]


def table(header: list[str], rows: list[list[str | int | float]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def axis_cell(verdicts: list[AxisVerdict]) -> str:
    """Accepted over resolved, with resolved over all in brackets."""
    accepted = verdicts.count("accept")
    resolved = accepted + verdicts.count("fail")
    return f"{share(accepted, resolved)} ({share(resolved, len(verdicts))})"


def length_band(row: ReplyRow, conversation_set: ConversationSet) -> LengthBand:
    low, high = conversation_set.bands[row.language][row.category]
    units = length_units(row.reply, row.language)
    if units < low:
        return "under"
    elif units > high:
        return "over"
    else:
        return "in"


def grouped(
    rows: list[ReplyRow],
) -> dict[tuple[str, PromptName, Language], list[ReplyRow]]:
    """Rows by llm, prompt and language: llms and prompts in the order each first appears, then English before Chinese."""
    groups: dict[tuple[str, PromptName, Language], list[ReplyRow]] = (
        collections.defaultdict(list)
    )
    for row in rows:
        groups[(row.llm, row.prompt, row.language)].append(row)
    pairs = list(dict.fromkeys((row.llm, row.prompt) for row in rows))
    return dict(
        sorted(
            groups.items(),
            key=lambda group: (pairs.index(group[0][:2]), LANGUAGES.index(group[0][2])),
        )
    )


def length_rows(
    rows: list[ReplyRow], conversation_set: ConversationSet
) -> list[list[str | int | float]]:
    table_rows: list[list[str | int | float]] = []
    for (llm, prompt, language), group in grouped(rows).items():
        units = [length_units(row.reply, row.language) for row in group]
        bands = [length_band(row, conversation_set) for row in group]
        total = len(group)
        table_rows.append(
            [
                llm,
                prompt,
                language,
                total,
                statistics.median(units),
                nearest_rank(units, 0.9),
                share(bands.count("under"), total),
                share(bands.count("in"), total),
                share(bands.count("over"), total),
                share(sum(bool(SYMBOLS.search(row.reply)) for row in group), total),
                share(sum(not row.reply.strip() for row in group), total),
                share(sum(row.finish_reason == "length" for row in group), total),
            ]
        )
    return table_rows


def judge_rows(
    rows: list[ReplyRow], judged: dict[str, JudgedReply]
) -> list[list[str | int | float]]:
    table_rows: list[list[str | int | float]] = []
    for (llm, prompt, language), group in grouped(rows).items():
        verdicts = [
            judged[row.reply_id].verdicts for row in group if row.reply_id in judged
        ]
        lengths = [
            verdict.length_fit for verdict in verdicts if verdict.length_fit is not None
        ]
        table_rows.append(
            [
                llm,
                prompt,
                language,
                f"{len(verdicts)}/{len(group)}",
                *(
                    axis_cell([verdict.by_axis()[axis] for verdict in verdicts])
                    for axis in get_args(Axis)
                ),
                axis_cell([verdict.joint() for verdict in verdicts]),
                *(share(lengths.count(name), len(lengths)) for name in LENGTH_VERDICTS),
            ]
        )
    return table_rows


def joint_and_too_short(
    rows: list[ReplyRow], judged: dict[str, JudgedReply]
) -> tuple[float, float]:
    """The share of replies whose joint verdict accepts, and of length verdicts that are too_short."""
    verdicts: list[Verdicts] = [judged[row.reply_id].verdicts for row in rows]
    lengths = [
        verdict.length_fit for verdict in verdicts if verdict.length_fit is not None
    ]
    joint_share = sum(verdict.joint() == "accept" for verdict in verdicts) / len(
        verdicts
    )
    return joint_share, lengths.count("too_short") / len(lengths)


def baseline_rows(
    rows: list[ReplyRow], judged: dict[str, JudgedReply], baseline: PromptName
) -> list[list[str | int | float]]:
    """Each prompt's joint accept and too_short minus the baseline's, in points, with a 95% cluster bootstrap interval over conversations."""
    chooser = random.Random(BOOTSTRAP_SEED)
    groups: dict[tuple[str, PromptName], dict[str, list[ReplyRow]]] = (
        collections.defaultdict(lambda: collections.defaultdict(list))
    )
    for row in rows:
        if row.reply_id in judged:
            groups[(row.llm, row.prompt)][row.conversation].append(row)
        else:
            pass
    compared = {
        (llm, prompt): by_conversation
        for (llm, prompt), by_conversation in groups.items()
        if prompt != baseline and (llm, baseline) in groups
    }
    table_rows: list[list[str | int | float]] = []
    for (llm, prompt), by_conversation in compared.items():
        baseline_by_conversation = groups[(llm, baseline)]
        conversations = list(by_conversation)
        joint_deltas = []
        too_short_deltas = []
        for resample in range(BOOTSTRAP_RESAMPLES + 1):
            if resample == 0:
                picked = conversations
            else:
                picked = chooser.choices(conversations, k=len(conversations))
            candidate_joint, candidate_too_short = joint_and_too_short(
                [row for name in picked for row in by_conversation[name]], judged
            )
            baseline_joint, baseline_too_short = joint_and_too_short(
                [row for name in picked for row in baseline_by_conversation[name]],
                judged,
            )
            joint_deltas.append(100 * (candidate_joint - baseline_joint))
            too_short_deltas.append(100 * (candidate_too_short - baseline_too_short))
        cells: list[str | int | float] = [llm, prompt, len(conversations)]
        for deltas in (joint_deltas, too_short_deltas):
            spread = sorted(deltas[1:])
            low = spread[int(0.025 * BOOTSTRAP_RESAMPLES)]
            high = spread[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]
            cells.append(f"{deltas[0]:+.1f} [{low:+.1f}, {high:+.1f}]")
        table_rows.append(cells)
    return table_rows


def category_rows(
    group: list[ReplyRow],
    judged: dict[str, JudgedReply],
    conversation_set: ConversationSet,
    prompts: list[PromptName],
) -> list[list[str | int | float]]:
    table_rows: list[list[str | int | float]] = []
    for category in conversation_set.categories:
        cells: list[str | int | float] = [category]
        for prompt in prompts:
            rows = [
                row
                for row in group
                if row.category == category and row.prompt == prompt
            ]
            lengths = [
                judged[row.reply_id].verdicts.length_fit
                for row in rows
                if row.reply_id in judged
                and judged[row.reply_id].verdicts.length_fit is not None
            ]
            cells += [
                statistics.median(
                    length_units(row.reply, row.language) for row in rows
                ),
                share(
                    sum(length_band(row, conversation_set) == "in" for row in rows),
                    len(rows),
                ),
                " / ".join(
                    share(lengths.count(name), len(lengths)) for name in LENGTH_VERDICTS
                ),
            ]
        table_rows.append(cells)
    return table_rows


def expand_rows(
    rows: list[ReplyRow], conversation_set: ConversationSet
) -> list[list[str | int | float]]:
    turns_by_conversation = {
        conversation.id: conversation.turns
        for conversation in conversation_set.conversations
    }
    by_id = {row.reply_id: row for row in rows}
    table_rows: list[list[str | int | float]] = []
    for (llm, prompt, language), group in grouped(rows).items():
        ratios = []
        for row in group:
            expands = turns_by_conversation[row.conversation][row.turn].expands
            if expands is not None:
                expanded = by_id[
                    f"{row.llm}/{row.prompt}/{row.conversation}/{row.sample}/{expands}"
                ]
                expanded_units = length_units(expanded.reply, expanded.language)
                ratios.append(
                    length_units(row.reply, row.language) / max(expanded_units, 1)
                )
            else:
                pass
        table_rows.append(
            [
                llm,
                prompt,
                language,
                len(ratios),
                f"{statistics.median(ratios):.2f}",
                share(sum(ratio > 1 for ratio in ratios), len(ratios)),
            ]
        )
    return table_rows


def live_rows(
    turns_by_label: dict[str, list[SpokenTurn]]
) -> list[list[str | int | float]]:
    table_rows: list[list[str | int | float]] = []
    for label, turns in turns_by_label.items():
        for language in LANGUAGES:
            group = [turn for turn in turns if turn.language == language]
            spoken = [turn for turn in group if turn.audio_s > 0]
            first_audio = [
                turn.first_audio_ms
                for turn in spoken
                if turn.first_audio_ms is not None
            ]
            after_stop = [
                turn.first_audio_ms - turn.speech_stopped_ms
                for turn in spoken
                if turn.first_audio_ms is not None
                and turn.speech_stopped_ms is not None
            ]
            table_rows.append(
                [
                    label,
                    language,
                    len(group),
                    len(group) - len(spoken),
                    statistics.median(
                        length_units(turn.reply, language) for turn in group
                    ),
                    f"{statistics.median(turn.audio_s for turn in group):.2f}",
                    f"{max(turn.audio_s for turn in group):.2f}",
                    statistics.median(first_audio) if first_audio else "-",
                    nearest_rank(first_audio, 0.9) if first_audio else "-",
                    statistics.median(after_stop) if after_stop else "-",
                ]
            )
    return table_rows


def live_category_rows(
    turns_by_label: dict[str, list[SpokenTurn]], conversation_set: ConversationSet
) -> list[list[str | int | float]]:
    table_rows: list[list[str | int | float]] = []
    for category in conversation_set.categories:
        cells: list[str | int | float] = [category]
        for turns in turns_by_label.values():
            for language in LANGUAGES:
                seconds = [
                    turn.audio_s
                    for turn in turns
                    if turn.category == category and turn.language == language
                ]
                cells.append(f"{statistics.median(seconds):.2f}" if seconds else "-")
        table_rows.append(cells)
    return table_rows


def write_summary(
    reply_paths: list[Path],
    judged_path: Path,
    report_path: Path,
    baseline: PromptName | None,
    live_dirs: list[Path],
) -> None:
    conversation_set = ConversationSet.model_validate_json(
        CONVERSATIONS_PATH.read_text()
    )
    rows = read_reply_rows(reply_paths)
    judged = {
        judged_reply.id: judged_reply
        for judged_reply in map(
            JudgedReply.model_validate_json, judged_path.read_text().splitlines()
        )
    }
    parts = [
        "# Cascade replies",
        "",
        "## Length",
        "",
        "Bands are per category in conversations.json; symbols are markdown, list lines, table pipes, URLs, emoji, formula signs or code.",
        "",
        table(
            [
                "llm",
                "prompt",
                "lang",
                "replies",
                "median",
                "p90",
                "under band",
                "in band",
                "over band",
                "symbols",
                "empty",
                "max_tokens hit",
            ],
            length_rows(rows, conversation_set),
        ),
        "",
        "## Judge",
        "",
        "Each axis is accepted over resolved, with resolved over all in brackets; joint accepts when all four axes accept and fails when any fails.",
        "",
        table(
            [
                "llm",
                "prompt",
                "lang",
                "judged",
                *get_args(Axis),
                "joint",
                *LENGTH_VERDICTS,
            ],
            judge_rows(rows, judged),
        ),
        "",
    ]
    if baseline is not None:
        parts += [
            f"## Against {baseline}",
            "",
            "Points of joint accept over all replies and of too_short over the length verdicts, minus the baseline's, with a 95% bootstrap interval over conversations.",
            "",
            table(
                ["llm", "prompt", "conversations", "joint accept", "too_short"],
                baseline_rows(rows, judged, baseline),
            ),
            "",
        ]
    else:
        pass
    parts += [
        "## By category",
        "",
        "Per prompt: median length, share in band, and too_short / right / too_long.",
        "",
    ]
    for llm in dict.fromkeys(row.llm for row in rows):
        prompts = list(dict.fromkeys(row.prompt for row in rows if row.llm == llm))
        header = ["category"] + [
            f"{prompt} {column}"
            for prompt in prompts
            for column in ("median", "in band", "length_fit")
        ]
        for language in LANGUAGES:
            group = [row for row in rows if row.llm == llm and row.language == language]
            parts += [
                f"### {llm}, {language}",
                "",
                table(header, category_rows(group, judged, conversation_set, prompts)),
                "",
            ]
    parts += [
        "## Expand turns",
        "",
        "Length of the reply to a request for more detail over the reply it expands.",
        "",
        table(
            ["llm", "prompt", "lang", "turns", "median ratio", "longer"],
            expand_rows(rows, conversation_set),
        ),
        "",
    ]
    if live_dirs:
        turns_by_label = {
            path.name: [
                SpokenTurn.model_validate_json(line)
                for line in (path / "turns.jsonl").read_text().splitlines()
            ]
            for path in live_dirs
        }
        parts += [
            "## Live",
            "",
            "Spoken turns through a running cascade, one session per conversation. first audio is from the end of the user's speech, and after stop from the speech_stopped event, which leaves out the turn detection's wait.",
            "",
            table(
                [
                    "run",
                    "lang",
                    "turns",
                    "no audio",
                    "median length",
                    "median audio s",
                    "max audio s",
                    "first audio ms",
                    "p90",
                    "after stop ms",
                ],
                live_rows(turns_by_label),
            ),
            "",
            "Median reply audio seconds per category:",
            "",
            table(
                [
                    "category",
                    *(
                        f"{label} {language}"
                        for label in turns_by_label
                        for language in LANGUAGES
                    ),
                ],
                live_category_rows(turns_by_label, conversation_set),
            ),
            "",
        ]
    else:
        pass
    report_path.write_text("\n".join(parts) + "\n")
    print(f"wrote {report_path}")
