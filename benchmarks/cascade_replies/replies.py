# SPDX-License-Identifier: Apache-2.0
"""Replay the reply conversations through one chat server under named system prompts.

The history is the cascade dialog's: each user turn is appended as said and each reply in full, prompt_messages gives what the model reads, and an empty reply is dropped with the user turns around it joined, as the dialog does. The request body is the one the profile's openai_chat generator builds, sent unstreamed. A conversation sample is a chain of requests whose rows are appended once it ends, so a rerun skips the chains already done.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.streaming_cascade.base import ChatMessage
from sglang_omni.models.streaming_cascade.config import StreamingCascadePipelineConfig
from sglang_omni.models.streaming_cascade.dialog import DialogConfig, prompt_messages
from sglang_omni.models.streaming_cascade.generators import (
    OpenAIChatGenerator,
    OpenAIChatGeneratorConfig,
)

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
    prompt: PromptName
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


async def replay_conversation(
    generator: OpenAIChatGenerator,
    llm_label: str,
    prompt_name: PromptName,
    system_prompt: str,
    conversation: Conversation,
    sample: int,
) -> list[ReplyRow]:
    history = [ChatMessage(role="system", content=system_prompt)]
    rows = []
    for turn, conversation_turn in enumerate(conversation.turns):
        history.append(ChatMessage(role="user", content=conversation_turn.user.strip()))
        if history[-2].role == "assistant" and not history[-2].content:
            joined = f"{history[-3].content} {history[-1].content}"
            history = [*history[:-3], ChatMessage(role="user", content=joined)]
        else:
            pass
        messages = prompt_messages(history)
        started_at = time.perf_counter()
        response = await generator.client.post(
            f"{generator.config.base_url.rstrip('/')}/chat/completions",
            json={**generator.request_body(messages), "stream": False},
        )
        response.raise_for_status()
        latency_s = time.perf_counter() - started_at
        completion = Completion.model_validate_json(response.content)
        choice = completion.choices[0]
        reply = choice.message.content or ""
        history.append(ChatMessage(role="assistant", content=reply))
        rows.append(
            ReplyRow(
                llm=llm_label,
                served_model=completion.model,
                prompt=prompt_name,
                conversation=conversation.id,
                language=conversation.language,
                sample=sample,
                turn=turn,
                category=conversation_turn.category,
                user=conversation_turn.user,
                messages=[
                    PromptMessage(role=message.role, content=message.content)
                    for message in messages
                ],
                reply=reply,
                finish_reason=choice.finish_reason,
                completion_tokens=completion.usage.completion_tokens,
                latency_s=round(latency_s, 3),
            )
        )
    return rows


def check_run_settings(output_path: Path, settings: dict[str, JsonValue]) -> None:
    """Record a run's settings beside its output, and refuse to append a run with other settings."""
    settings_path = output_path.with_suffix(".meta.json")
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
    output_path: Path,
    base_url: str | None,
    served_model: str | None,
    prompt_names: list[PromptName],
    extra_body: dict[str, JsonValue],
    samples: int,
    concurrency: int,
) -> None:
    profile = ConfigManager.from_file(str(profile_path)).config
    if not isinstance(profile, StreamingCascadePipelineConfig):
        raise SystemExit(f"{profile_path} is not a streaming cascade profile")
    else:
        pass
    graph = profile.component_graph()
    profile_llm_config = graph.node("llm").component
    profile_dialog = graph.node("dialog").component
    if not isinstance(profile_llm_config, OpenAIChatGeneratorConfig):
        raise SystemExit(f"{profile_path}: the llm node is not openai_chat")
    elif not isinstance(profile_dialog, DialogConfig):
        raise SystemExit(f"{profile_path}: the dialog node is not a dialog")
    else:
        pass
    llm_config = OpenAIChatGeneratorConfig.model_validate(
        {
            **profile_llm_config.model_dump(),
            "base_url": base_url or profile_llm_config.base_url,
            "model": served_model or profile_llm_config.model,
            "extra_body": {**profile_llm_config.extra_body, **extra_body},
        }
    )
    system_prompts: dict[PromptName, str] = {
        "profile": profile_dialog.system_prompt,
        "dialog_default": DialogConfig().system_prompt,
        "no_length_rule": NO_LENGTH_RULE_PROMPT,
        "one_sentence": ONE_SENTENCE_PROMPT,
        "cap_with_exceptions": CAP_WITH_EXCEPTIONS_PROMPT,
        "fitted_length_small_talk": FITTED_LENGTH_SMALL_TALK_PROMPT,
    }
    conversations = ConversationSet.model_validate_json(
        CONVERSATIONS_PATH.read_text()
    ).conversations
    # note (Haoling Pu): the chosen prompts and the sample count stay out of the settings, so a rerun can add prompts or samples to the same output.
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    check_run_settings(
        output_path,
        {
            "llm": llm_label,
            "profile": str(profile_path),
            "commit": commit,
            "llm_config": llm_config.model_dump(mode="json"),
            "system_prompts": dict(system_prompts),
            "conversations_sha256": hashlib.sha256(
                CONVERSATIONS_PATH.read_bytes()
            ).hexdigest(),
        },
    )
    lines = output_path.read_text().splitlines() if output_path.is_file() else []
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
    generator = OpenAIChatGenerator(llm_config)
    async with generator.client:

        async def run_chain(
            prompt_name: PromptName, conversation: Conversation, sample: int
        ) -> None:
            nonlocal finished
            async with limit:
                rows = await replay_conversation(
                    generator,
                    llm_label,
                    prompt_name,
                    system_prompts[prompt_name],
                    conversation,
                    sample,
                )
            with output_path.open("a") as stream:
                stream.write("".join(row.model_dump_json() + "\n" for row in rows))
            finished += 1
            if finished % 20 == 0 or finished == len(chains):
                print(f"{finished}/{len(chains)} chains")
            else:
                pass

        await asyncio.gather(*(run_chain(*chain) for chain in chains))
