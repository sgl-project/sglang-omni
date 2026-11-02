# SPDX-License-Identifier: Apache-2.0
"""Replay the reply conversations through one chat server under named system prompts.

The history is the cascade dialog's: each user turn is appended as said and each reply in full, prompt_messages gives what the model reads, and an empty reply is dropped with the user turns around it joined, as the dialog does. The request is the profile's openai_chat request, unstreamed. A conversation sample is a chain of requests whose rows are appended once it ends, so a rerun skips the chains already done.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Literal

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, JsonValue

from sglang_omni.models.streaming_cascade.base import ChatMessage
from sglang_omni.models.streaming_cascade.dialog import DialogConfig, prompt_messages
from sglang_omni.models.streaming_cascade.generators import OpenAIChatGeneratorConfig

CONVERSATIONS_PATH = Path(__file__).with_name("conversations.json")
NO_LENGTH_RULE_PROMPT = "You are a helpful voice assistant. Answer in the language the user speaks. Do not use markdown or emoji."
ONE_SENTENCE_PROMPT = "You are a helpful voice assistant. Answer in one spoken sentence of at most 25 words, in the language the user speaks. Do not use markdown or emoji."
CAP_WITH_EXCEPTIONS_PROMPT = "You are a helpful voice assistant. Answer in one spoken sentence of at most 25 words, in the language the user speaks. Do not use markdown or emoji. If the user asks for steps or more detail, use up to four short sentences instead. If the request is unclear, ask a short question. Never mention a word limit."
FITTED_LENGTH_SMALL_TALK_PROMPT = "You are a helpful voice assistant. Answer in the language the user speaks, as plain speech with no markdown, lists or emoji. Give the answer in your first sentence. Most replies need one sentence of at most 25 words. When the user asks for steps, details or an explanation, or the question needs steps, use up to three short sentences. When the user asks for a short answer, give just that. For small talk, reply briefly and warmly, without advice. If the request is unclear, ask one short question instead of guessing. For long content such as lists, tables, code or itineraries, say the two or three most useful items. Never mention these rules."

PromptName = Literal[
    "profile",
    "dialog_default",
    "no_length_rule",
    "one_sentence",
    "cap_with_exceptions",
    "fitted_length_small_talk",
]
Language = Literal["en", "zh"]


class ConversationTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    user: str
    points: list[str]
    expands: int | None = None
    note: str | None = None


class Conversation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    language: Language
    topic: str
    turns: list[ConversationTurn]


class ConversationSet(BaseModel):
    """conversations.json: per category what a good spoken reply does, and its length band per language."""

    model_config = ConfigDict(extra="forbid")

    version: int
    units: dict[Language, str]
    categories: dict[str, str]
    bands: dict[Language, dict[str, tuple[int, int]]]
    conversations: list[Conversation]


class PromptMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class CompletionMessage(BaseModel):
    content: str | None


class CompletionChoice(BaseModel):
    message: CompletionMessage
    finish_reason: str


class CompletionUsage(BaseModel):
    completion_tokens: int


class Completion(BaseModel):
    """The fields read from an OpenAI-compatible chat completion."""

    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage


class ReplyRow(BaseModel):
    llm: str
    served_model: str
    prompt: str
    conversation: str
    language: Language
    sample: int
    turn: int
    category: str
    user: str
    messages: list[PromptMessage]
    reply: str
    finish_reason: str
    completion_tokens: int
    latency_s: float

    @property
    def reply_id(self) -> str:
        return f"{self.llm}/{self.prompt}/{self.conversation}/{self.sample}/{self.turn}"


def read_reply_rows(paths: list[Path]) -> list[ReplyRow]:
    return [
        ReplyRow.model_validate_json(line)
        for path in paths
        for line in path.read_text().splitlines()
    ]


def system_prompts(profile_dialog: DialogConfig) -> dict[PromptName, str]:
    return {
        "profile": profile_dialog.system_prompt,
        "dialog_default": DialogConfig().system_prompt,
        "no_length_rule": NO_LENGTH_RULE_PROMPT,
        "one_sentence": ONE_SENTENCE_PROMPT,
        "cap_with_exceptions": CAP_WITH_EXCEPTIONS_PROMPT,
        "fitted_length_small_talk": FITTED_LENGTH_SMALL_TALK_PROMPT,
    }


async def replay_conversation(
    client: httpx.AsyncClient,
    llm: OpenAIChatGeneratorConfig,
    base_url: str,
    llm_label: str,
    prompt_name: PromptName,
    system_prompt: str,
    conversation: Conversation,
    sample: int,
) -> list[ReplyRow]:
    messages = [ChatMessage(role="system", content=system_prompt)]
    rows = []
    for turn, spec in enumerate(conversation.turns):
        messages.append(ChatMessage(role="user", content=spec.user.strip()))
        if messages[-2].role == "assistant" and not messages[-2].content:
            joined = f"{messages[-3].content} {messages[-1].content}"
            messages = [*messages[:-3], ChatMessage(role="user", content=joined)]
        else:
            pass
        prompt = [
            PromptMessage(role=message.role, content=message.content)
            for message in prompt_messages(messages)
        ]
        body: dict[str, JsonValue] = {
            "model": llm.model,
            "messages": [message.model_dump() for message in prompt],
            "stream": False,
            "max_tokens": llm.max_tokens,
        }
        if llm.temperature is not None:
            body["temperature"] = llm.temperature
        else:
            pass
        body.update(llm.extra_body)
        started = time.perf_counter()
        response = await client.post(f"{base_url}/chat/completions", json=body)
        response.raise_for_status()
        latency_s = time.perf_counter() - started
        completion = Completion.model_validate_json(response.content)
        choice = completion.choices[0]
        reply = choice.message.content or ""
        messages.append(ChatMessage(role="assistant", content=reply))
        rows.append(
            ReplyRow(
                llm=llm_label,
                served_model=completion.model,
                prompt=prompt_name,
                conversation=conversation.id,
                language=conversation.language,
                sample=sample,
                turn=turn,
                category=spec.category,
                user=spec.user,
                messages=prompt,
                reply=reply,
                finish_reason=choice.finish_reason,
                completion_tokens=completion.usage.completion_tokens,
                latency_s=round(latency_s, 3),
            )
        )
    return rows


def check_run_settings(out: Path, settings: dict[str, JsonValue]) -> None:
    """Record a run's settings beside its output, and refuse to append a run with other settings."""
    settings_path = out.with_suffix(".meta.json")
    if not settings_path.is_file():
        settings_path.write_text(
            json.dumps(settings, ensure_ascii=False, indent=2) + "\n"
        )
    elif json.loads(settings_path.read_text()) != settings:
        raise SystemExit(
            f"{settings_path}: this run's settings differ from the recorded ones; use a new --out"
        )
    else:
        pass


async def run_replies(
    profile_path: Path,
    llm_label: str,
    out: Path,
    base_url: str | None,
    served_model: str | None,
    prompt_names: list[PromptName],
    extra_body: dict[str, JsonValue],
    samples: int,
    concurrency: int,
) -> None:
    profile = yaml.safe_load(profile_path.read_text())
    session_nodes = profile["session_nodes"]
    llm = OpenAIChatGeneratorConfig.model_validate(
        {
            name: value
            for name, value in session_nodes["llm"].items()
            if name != "inputs"
        }
    )
    profile_dialog = DialogConfig.model_validate(
        {
            name: value
            for name, value in session_nodes["dialog"].items()
            if name != "inputs"
        }
    )
    llm = llm.model_copy(
        update={
            "model": served_model or llm.model,
            "extra_body": {**llm.extra_body, **extra_body},
        }
    )
    resolved_base_url = (base_url or llm.base_url).rstrip("/")
    prompts = system_prompts(profile_dialog)
    conversations = ConversationSet.model_validate_json(
        CONVERSATIONS_PATH.read_text()
    ).conversations
    # note (Haoling Pu): the chosen prompts and the sample count stay out of the settings, so a rerun can add prompts or samples to the same output.
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    check_run_settings(
        out,
        {
            "llm": llm_label,
            "profile": str(profile_path),
            "commit": commit,
            "base_url": resolved_base_url,
            "request": {
                "model": llm.model,
                "max_tokens": llm.max_tokens,
                "temperature": llm.temperature,
                **llm.extra_body,
            },
            "prompts": dict(prompts),
            "conversations_sha256": hashlib.sha256(
                CONVERSATIONS_PATH.read_bytes()
            ).hexdigest(),
        },
    )
    lines = out.read_text().splitlines() if out.is_file() else []
    done = {
        (row.prompt, row.conversation, row.sample)
        for row in map(ReplyRow.model_validate_json, lines)
    }
    chains = [
        (prompt_name, conversation, sample)
        for prompt_name in prompt_names
        for conversation in conversations
        for sample in range(samples)
        if (prompt_name, conversation.id, sample) not in done
    ]
    print(f"{len(chains)} chains to run, {len(done)} already done")
    limit = asyncio.Semaphore(concurrency)
    finished = 0
    async with httpx.AsyncClient(timeout=llm.timeout_s) as client:

        async def run_chain(
            prompt_name: PromptName, conversation: Conversation, sample: int
        ) -> None:
            nonlocal finished
            async with limit:
                rows = await replay_conversation(
                    client,
                    llm,
                    resolved_base_url,
                    llm_label,
                    prompt_name,
                    prompts[prompt_name],
                    conversation,
                    sample,
                )
            with out.open("a") as stream:
                stream.write("".join(row.model_dump_json() + "\n" for row in rows))
            finished += 1
            if finished % 20 == 0 or finished == len(chains):
                print(f"{finished}/{len(chains)} chains")
            else:
                pass

        await asyncio.gather(*(run_chain(*chain) for chain in chains))
