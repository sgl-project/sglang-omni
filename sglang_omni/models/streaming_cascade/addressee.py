# SPDX-License-Identifier: Apache-2.0
"""Whether speech heard over a paused reply was addressed to the assistant, asked of the chat service that writes the replies."""

import logging
from collections.abc import Sequence

import httpx
from pydantic import BaseModel

from sglang_omni.audio.floor import AddresseeQuestion, logsumexp
from sglang_omni.models.streaming_cascade.base import (
    ChatMessage,
    bearer_headers,
    message_dict,
)
from sglang_omni.models.streaming_cascade.generators import OpenAIChatGeneratorConfig

logger = logging.getLogger(__name__)

QUESTION = AddresseeQuestion(
    text=(
        "You were speaking when this speech was heard. Answer yes if the speaker "
        "is talking to you: a question, a request, a correction, disagreement, "
        "asking you to stop, or a sudden change of topic, even one about their "
        "own plans or errands. Answer no only if it is clearly a short "
        "acknowledgement that lets you go on (such as mm-hmm, yeah, right, okay), "
        "clearly said to another person nearby, or background speech or noise "
        "such as a TV. If unsure, answer yes. Answer yes or no."
    ),
    addressed=("yes", "Yes"),
    other=("no", "No"),
)
HISTORY_MESSAGES = 4
TOP_LOGPROBS = 20


def addressee_messages(
    history: Sequence[ChatMessage], reply: str, heard: Sequence[str]
) -> list[ChatMessage]:
    recent = [message for message in history if message.role != "system"]
    lines = [
        f"{message.role.capitalize()}: {message.content}"
        for message in recent[-HISTORY_MESSAGES:]
    ]
    lines.append(f"Assistant (speaking): {reply}")
    lines.append(f'Heard while speaking: "{" ".join(heard)}"')
    return [
        ChatMessage(role="system", content=QUESTION.text),
        ChatMessage(role="user", content="\n".join(lines)),
    ]


class TopLogprob(BaseModel):
    token: str
    logprob: float


class TokenLogprobs(BaseModel):
    top_logprobs: list[TopLogprob] = []


class ChoiceLogprobs(BaseModel):
    content: list[TokenLogprobs] | None = None


class JudgeChoice(BaseModel):
    logprobs: ChoiceLogprobs | None = None


class JudgeResponse(BaseModel):
    choices: list[JudgeChoice]


def label_logprobs(top_logprobs: Sequence[TopLogprob]) -> dict[str, float]:
    """The log-probability of each label among the first token's candidates, the tokens that read as the label once stripped summed."""
    found: dict[str, list[float]] = {}
    for entry in top_logprobs:
        label = entry.token.strip()
        if label in QUESTION.labels:
            found.setdefault(label, []).append(entry.logprob)
        else:
            pass
    return {label: logsumexp(values) for label, values in found.items()}


class AddresseeJudge:
    """One question per Ask to the chat service: a single greedy token whose top log-probs give the margin; a service error or timeout gives no evidence."""

    def __init__(self, chat: OpenAIChatGeneratorConfig, timeout_s: float) -> None:
        self.chat = chat
        self.client = httpx.AsyncClient(timeout=timeout_s)

    def request_body(self, messages: list[ChatMessage]) -> dict[str, object]:
        extra = dict(self.chat.extra_body)
        template_kwargs = extra.get("chat_template_kwargs")
        extra["chat_template_kwargs"] = {
            **(template_kwargs if isinstance(template_kwargs, dict) else {}),
            "enable_thinking": False,
        }
        if "min_tokens" in extra:
            extra["min_tokens"] = 0
        else:
            pass
        return {
            **extra,
            "model": self.chat.model,
            "messages": [message_dict(message) for message in messages],
            "stream": False,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": TOP_LOGPROBS,
        }

    async def margin(
        self, history: Sequence[ChatMessage], reply: str, heard: Sequence[str]
    ) -> float | None:
        """log P(addressed) - log P(other) for the speech, or None when the service gave no usable answer."""
        url = f"{self.chat.base_url.rstrip('/')}/chat/completions"
        body = self.request_body(addressee_messages(history, reply, heard))
        try:
            response = await self.client.post(
                url, json=body, headers=bearer_headers(self.chat.api_key_env)
            )
            response.raise_for_status()
            parsed = JudgeResponse.model_validate_json(response.content)
        except (httpx.HTTPError, ValueError) as error:
            logger.warning("addressee judge at %s failed: %r", url, error)
            return None
        logprobs = parsed.choices[0].logprobs if parsed.choices else None
        if logprobs is None or not logprobs.content:
            logger.warning("addressee judge at %s returned no log-probs", url)
            return None
        else:
            return QUESTION.margin(label_logprobs(logprobs.content[0].top_logprobs))

    async def close(self) -> None:
        await self.client.aclose()
