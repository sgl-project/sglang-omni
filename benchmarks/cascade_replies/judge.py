# SPDX-License-Identifier: Apache-2.0
"""Judge replies on five axes with an LLM, the model and system prompt behind each reply withheld.

relevance, completeness, directness and listenability are accept, fail or unresolved with a quote from the reply; a verdict whose quote is not in the reply becomes unresolved. length_fit is too_short, right or too_long. An empty reply fails every axis and is too_short without a request, and a judgment cut at max_tokens leaves every axis unresolved.
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import json
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from benchmarks.cascade_replies.replies import (
    CONVERSATIONS_PATH,
    Completion,
    ConversationSet,
    Language,
    PromptMessage,
    check_run_settings,
    read_reply_rows,
)

ASSETS = Path(__file__).parent
PROMPT_PATH = ASSETS / "judge-prompt.txt"
SCHEMA_PATH = ASSETS / "judge-response.schema.json"
CONTROLS_PATH = ASSETS / "judge-controls.jsonl"
JUDGE_SEED = 1

AxisVerdict = Literal["accept", "fail", "unresolved"]
LengthVerdict = Literal["too_short", "right", "too_long"]
Axis = Literal["relevance", "completeness", "directness", "listenability"]


class AxisJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    reason: str
    quote: str
    verdict: AxisVerdict


class LengthJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    reason: str
    verdict: LengthVerdict


class Judgment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    relevance: AxisJudgment
    completeness: AxisJudgment
    directness: AxisJudgment
    listenability: AxisJudgment
    length_fit: LengthJudgment


class Verdicts(BaseModel):
    relevance: AxisVerdict
    completeness: AxisVerdict
    directness: AxisVerdict
    listenability: AxisVerdict
    length_fit: LengthVerdict | None

    def by_axis(self) -> dict[Axis, AxisVerdict]:
        return {
            "relevance": self.relevance,
            "completeness": self.completeness,
            "directness": self.directness,
            "listenability": self.listenability,
        }

    def joint(self) -> AxisVerdict:
        """Accepts when all four axes accept and fails when any fails."""
        axis_verdicts = list(self.by_axis().values())
        if "fail" in axis_verdicts:
            return "fail"
        elif all(verdict == "accept" for verdict in axis_verdicts):
            return "accept"
        else:
            return "unresolved"


class JudgedReply(BaseModel):
    id: str
    status: Literal["judged", "empty_reply", "truncated"]
    verdicts: Verdicts
    unquoted_axes: list[Axis] = Field(default_factory=list)
    judgment: Judgment | None = None
    completion_tokens: int | None = None


class JudgeCase(BaseModel):
    """What the judge reads about one reply."""

    language: Language
    category: str
    category_expectation: str
    key_points: list[str]
    conversation: list[PromptMessage]
    user_turn: str
    reply: str
    note: str | None
    expanded_reply: str | None


class JudgeControl(BaseModel):
    """A hand-written reply with the verdicts a sound judge gives it."""

    model_config = ConfigDict(extra="forbid")

    id: str
    language: Language
    category: str
    points: list[str]
    conversation: list[PromptMessage]
    user: str
    reply: str
    note: str | None = None
    expanded_reply: str | None = None
    expected: dict[Axis | Literal["length_fit"], AxisVerdict | LengthVerdict]


def verdicts_of(judgment: Judgment, reply: str) -> tuple[Verdicts, list[Axis]]:
    """The verdict of every axis, and the axes whose quote was not found in the reply."""
    spaced_reply = " ".join(reply.split())
    axis_verdicts: dict[Axis, AxisVerdict] = {}
    unquoted_axes: list[Axis] = []
    for axis, axis_judgment in (
        ("relevance", judgment.relevance),
        ("completeness", judgment.completeness),
        ("directness", judgment.directness),
        ("listenability", judgment.listenability),
    ):
        quote = " ".join(axis_judgment.quote.split())
        if axis_judgment.verdict == "unresolved":
            axis_verdicts[axis] = "unresolved"
        elif quote and quote in spaced_reply:
            axis_verdicts[axis] = axis_judgment.verdict
        else:
            axis_verdicts[axis] = "unresolved"
            unquoted_axes.append(axis)
    return (
        Verdicts(**axis_verdicts, length_fit=judgment.length_fit.verdict),
        unquoted_axes,
    )


async def judge_case(
    client: httpx.AsyncClient,
    base_url: str,
    served_model: str,
    decoding: dict[str, JsonValue],
    judge_prompt: str,
    schema: dict[str, JsonValue],
    case_id: str,
    case: JudgeCase,
) -> JudgedReply:
    if not case.reply.strip():
        return JudgedReply(
            id=case_id,
            status="empty_reply",
            verdicts=Verdicts(
                relevance="fail",
                completeness="fail",
                directness="fail",
                listenability="fail",
                length_fit="too_short",
            ),
        )
    else:
        pass
    case_text = json.dumps(
        case.model_dump(exclude_none=True), ensure_ascii=False, indent=1
    )
    body: dict[str, JsonValue] = {
        "model": served_model,
        "messages": [
            {"role": "system", "content": judge_prompt},
            {
                "role": "user",
                "content": f"Response JSON schema:\n{json.dumps(schema)}\n\nCase:\n{case_text}",
            },
        ],
        **decoding,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "cascade_reply_judgment", "schema": schema},
        },
    }
    response = await client.post(f"{base_url}/chat/completions", json=body)
    response.raise_for_status()
    completion = Completion.model_validate_json(response.content)
    choice = completion.choices[0]
    if choice.finish_reason != "stop":
        return JudgedReply(
            id=case_id,
            status="truncated",
            verdicts=Verdicts(
                relevance="unresolved",
                completeness="unresolved",
                directness="unresolved",
                listenability="unresolved",
                length_fit=None,
            ),
            completion_tokens=completion.usage.completion_tokens,
        )
    else:
        judgment = Judgment.model_validate_json(choice.message.content or "")
        verdicts, unquoted_axes = verdicts_of(judgment, case.reply)
        return JudgedReply(
            id=case_id,
            status="judged",
            verdicts=verdicts,
            unquoted_axes=unquoted_axes,
            judgment=judgment,
            completion_tokens=completion.usage.completion_tokens,
        )


async def judge_all(
    cases: dict[str, JudgeCase],
    output_path: Path,
    base_url: str,
    served_model: str,
    enable_thinking: bool,
    max_tokens: int,
    concurrency: int,
    timeout_s: float,
) -> dict[str, JudgedReply]:
    """Judge the cases not yet in output_path, appending each judgment as it arrives."""
    judge_prompt = PROMPT_PATH.read_text()
    schema = json.loads(SCHEMA_PATH.read_text())
    resolved_base_url = base_url.rstrip("/")
    decoding: dict[str, JsonValue] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "seed": JUDGE_SEED,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    check_run_settings(
        output_path,
        {
            "served_model": served_model,
            "base_url": resolved_base_url,
            "decoding": decoding,
            "prompt_sha256": hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest(),
            "schema_sha256": hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest(),
        },
    )
    lines = output_path.read_text().splitlines() if output_path.is_file() else []
    results = {
        judged.id: judged for judged in map(JudgedReply.model_validate_json, lines)
    }
    pending = [case_id for case_id in cases if case_id not in results]
    print(f"{len(pending)} cases to judge, {len(results)} already judged")
    limit = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=timeout_s) as client:

        async def run_case(case_id: str) -> None:
            async with limit:
                judged = await judge_case(
                    client,
                    resolved_base_url,
                    served_model,
                    decoding,
                    judge_prompt,
                    schema,
                    case_id,
                    cases[case_id],
                )
            results[case_id] = judged
            with output_path.open("a") as stream:
                stream.write(judged.model_dump_json() + "\n")
            if len(results) % 100 == 0:
                print(f"{len(results)}/{len(cases)} judged")
            else:
                pass

        await asyncio.gather(*(run_case(case_id) for case_id in pending))
    return results


async def judge_replies(
    reply_paths: list[Path],
    output_path: Path,
    base_url: str,
    served_model: str,
    enable_thinking: bool,
    max_tokens: int,
    concurrency: int,
    timeout_s: float,
) -> None:
    conversation_set = ConversationSet.model_validate_json(
        CONVERSATIONS_PATH.read_text()
    )
    rows = read_reply_rows(reply_paths)
    turns_by_conversation = {
        conversation.id: conversation.turns
        for conversation in conversation_set.conversations
    }
    reply_by_id = {row.reply_id: row.reply for row in rows}
    cases = {}
    for row in rows:
        conversation_turn = turns_by_conversation[row.conversation][row.turn]
        if conversation_turn.expands is not None:
            expanded_reply = reply_by_id[
                f"{row.llm}/{row.prompt}/{row.conversation}/{row.sample}/{conversation_turn.expands}"
            ]
        else:
            expanded_reply = None
        cases[row.reply_id] = JudgeCase(
            language=row.language,
            category=row.category,
            category_expectation=conversation_set.categories[row.category],
            key_points=conversation_turn.points,
            conversation=row.messages[1:-1],
            user_turn=row.messages[-1].content,
            reply=row.reply,
            note=conversation_turn.note,
            expanded_reply=expanded_reply,
        )
    results = await judge_all(
        cases,
        output_path,
        base_url,
        served_model,
        enable_thinking,
        max_tokens,
        concurrency,
        timeout_s,
    )
    print(dict(collections.Counter(judged.status for judged in results.values())))


async def judge_controls(
    output_path: Path,
    base_url: str,
    served_model: str,
    enable_thinking: bool,
    max_tokens: int,
    concurrency: int,
    timeout_s: float,
) -> None:
    """Judge the controls and print how many expected verdicts the judge matched, so a judge setting is checked before its verdicts are used."""
    conversation_set = ConversationSet.model_validate_json(
        CONVERSATIONS_PATH.read_text()
    )
    controls = [
        JudgeControl.model_validate_json(line)
        for line in CONTROLS_PATH.read_text().splitlines()
    ]
    cases = {
        control.id: JudgeCase(
            language=control.language,
            category=control.category,
            category_expectation=conversation_set.categories[control.category],
            key_points=control.points,
            conversation=control.conversation,
            user_turn=control.user,
            reply=control.reply,
            note=control.note,
            expanded_reply=control.expanded_reply,
        )
        for control in controls
    }
    results = await judge_all(
        cases,
        output_path,
        base_url,
        served_model,
        enable_thinking,
        max_tokens,
        concurrency,
        timeout_s,
    )
    matched = 0
    expected_count = 0
    for control in controls:
        verdicts = results[control.id].verdicts
        judged_verdicts = {**verdicts.by_axis(), "length_fit": verdicts.length_fit}
        misses = {
            axis: (expected, judged_verdicts[axis])
            for axis, expected in control.expected.items()
            if judged_verdicts[axis] != expected
        }
        matched += len(control.expected) - len(misses)
        expected_count += len(control.expected)
        print(f"{control.id}: {misses or 'ok'}")
    print(f"matched {matched} of {expected_count} expected verdicts")
