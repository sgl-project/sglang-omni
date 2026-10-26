# SPDX-License-Identifier: Apache-2.0
"""Building blocks shared by the component kinds: the component contract, chunk helpers and service probes."""

import asyncio
import os
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterable
from functools import partial
from typing import Literal, Protocol

import httpx
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import (
    AUDIO,
    CONTROL,
    INTERRUPT,
    MESSAGES,
    PCM16,
    PCM_F32,
    SESSION_CONTROL,
    TEXT,
    TEXT_DELTA,
    TRANSCRIPT,
    TRANSCRIPT_FORMAT,
    control_type,
)
from sglang_omni.scheduling.session import ChunkEmitter

SPEECH_SAMPLE_RATE_HZ = 24000
PROBE_TIMEOUT_S = 10.0
# Note (Junnan Li): A word ending in a period ends a sentence even after an abbreviation such as "Dr.", which then becomes its own short sentence.
SENTENCE_END = re.compile(r"[.!?;。！？；…]+[\"')\]”’」』）]*$")
# Note (Junnan Li): CJK text has no spaces between sentences, so a sentence also ends inside a word once its punctuation run is followed by more text.
CJK_SENTENCE_END = re.compile(r"[。！？；…]+[\"')\]”’」』）]*(?=\S)")


class StreamComponent:
    """One node instance of one Session."""

    def run(self, inputs: AsyncIterator[TimedChunk]) -> AsyncIterator[TimedChunk]:
        return streamed(partial(self.process, inputs))

    async def process(
        self, inputs: AsyncIterator[TimedChunk], emit: ChunkEmitter
    ) -> None:
        raise NotImplementedError

    async def control(self, event: TimedChunk) -> None:
        """Handle a control event of one of the node's control edges; a kind without control semantics lets it pass."""

    async def close(self) -> None:
        return

    def adopt(self, previous: "StreamComponent") -> None:
        """Take over the session state of the instance this one replaces."""

    def can_replace(self) -> bool:
        """Whether replacing this component would discard work in progress."""
        return True

    def state_bytes(self) -> int:
        return 0


class ChatMessage(BaseModel):
    """One history entry; `uncertain` marks an assistant reply cut where the listener may have heard part of the next sentence, and stays out of the model's prompt."""

    role: Literal["system", "user", "assistant"]
    content: str = ""
    uncertain: bool = False


def control_chunk(
    control_type: str, t_start_ms: float = 0, turn: int | None = None, **fields: object
) -> TimedChunk:
    return TimedChunk(
        CONTROL,
        t_start_ms,
        0,
        0,
        {"type": control_type, **fields},
        format=SESSION_CONTROL,
        turn=turn,
    )


def with_language(
    payload: dict[str, object], language: str | None
) -> dict[str, object]:
    """The payload with the user's language when it is known."""
    if language is None:
        return payload
    else:
        return {**payload, "language": language}


def transcript_chunk(
    text: str, is_final: bool, language: str | None = None
) -> TimedChunk:
    return TimedChunk(
        TRANSCRIPT,
        0,
        0,
        0,
        with_language({"text": text, "is_final": is_final}, language),
        format=TRANSCRIPT_FORMAT,
    )


def text_chunk(text: str, turn: int, language: str | None = None) -> TimedChunk:
    return TimedChunk(
        TEXT,
        0,
        0,
        0,
        with_language({"text": text}, language),
        format=TEXT_DELTA,
        turn=turn,
    )


def messages_chunk(
    messages: list[ChatMessage], turn: int, language: str | None = None
) -> TimedChunk:
    return TimedChunk(
        TEXT,
        0,
        0,
        0,
        with_language(
            {"messages": [message_dict(message) for message in messages]}, language
        ),
        format=MESSAGES,
        turn=turn,
    )


def message_dict(message: ChatMessage) -> dict[str, object]:
    return message.model_dump(exclude_none=True, exclude={"uncertain"})


def audio_chunk(samples: np.ndarray, sample_rate_hz: int, turn: int) -> TimedChunk:
    return TimedChunk(
        AUDIO,
        0,
        len(samples) * 1000 / sample_rate_hz,
        0,
        samples.astype("<f4").tobytes(),
        format=PCM_F32,
        turn=turn,
    )


def audio_bytes(chunk: TimedChunk) -> bytes:
    if chunk.modality != AUDIO or chunk.format != PCM16:
        raise ValueError("expected 16 kHz mono pcm16 audio")
    elif not isinstance(chunk.payload, bytes):
        raise ValueError("audio payload must be bytes")
    else:
        return chunk.payload


def messages_of(chunk: TimedChunk) -> list[ChatMessage]:
    if chunk.format != MESSAGES or not isinstance(chunk.payload, dict):
        raise ValueError("generator input must be messages chunks")
    else:
        return [ChatMessage.model_validate(item) for item in chunk.payload["messages"]]


async def iterate(queue: asyncio.Queue[TimedChunk | None]) -> AsyncIterator[TimedChunk]:
    """Chunks from `queue` until a None marks the end."""
    while (chunk := await queue.get()) is not None:
        yield chunk


async def streamed(
    work: Callable[[ChunkEmitter], Awaitable[None]],
) -> AsyncIterator[TimedChunk]:
    """Yield what `work` emits while it runs, then raise what it raised."""
    outputs: asyncio.Queue[TimedChunk] = asyncio.Queue()
    task = asyncio.ensure_future(work(outputs.put_nowait))
    getter: asyncio.Future[TimedChunk] | None = None
    try:
        while not task.done() or not outputs.empty():
            if outputs.empty():
                getter = asyncio.ensure_future(outputs.get())
                await asyncio.wait({getter, task}, return_when=asyncio.FIRST_COMPLETED)
                if getter.done():
                    yield getter.result()
                else:
                    getter.cancel()
                    await asyncio.wait({getter})
                getter = None
            else:
                yield outputs.get_nowait()
        task.result()
    finally:
        for pending in (getter, task):
            if pending is not None and not pending.done():
                pending.cancel()
                await asyncio.wait({pending})
            else:
                pass


async def stop_running(tasks: Iterable[asyncio.Task[None] | None]) -> None:
    """Cancel the tasks still running, wait for them, and raise what one raised instead of stopping."""
    running = [task for task in tasks if task is not None and not task.done()]
    for task in running:
        task.cancel()
    if running:
        await asyncio.wait(running)
    else:
        pass
    for task in running:
        if not task.cancelled() and task.exception() is not None:
            raise task.exception()
        else:
            pass


class Replies:
    """The one reply task a node runs beside its input reader.

    `interrupted` holds the turns whose late chunks the node drops; a turn
    leaves it when its turn.done arrives or a later reply starts.
    """

    def __init__(self) -> None:
        self.task: asyncio.Task[None] | None = None
        self.turn: int | None = None
        self.interrupted: set[int] = set()

    def start(self, turn: int, work: Coroutine[None, None, None]) -> None:
        self.interrupted = {earlier for earlier in self.interrupted if earlier > turn}
        self.turn = turn
        self.task = asyncio.ensure_future(work)

    def is_running(self, turn: int) -> bool:
        return self.task is not None and self.turn == turn

    async def interrupt(self, turn: int) -> None:
        self.interrupted.add(turn)
        task = self.task
        if task is not None and self.turn == turn:
            task.cancel()
            await asyncio.wait({task})
            if self.task is task:
                self.task = None
            else:
                pass
        else:
            pass

    async def finish(self) -> None:
        if self.task is not None:
            await self.task
            self.task = None
        else:
            pass


ChunkHandler = Callable[[TimedChunk, Replies], Awaitable[None]]


class Replier(StreamComponent):
    """A node that runs one reply task beside its input reader; an `interrupt` control cancels the reply of its turn."""

    def __init__(self) -> None:
        self.replies = Replies()

    def can_replace(self) -> bool:
        return self.replies.task is None or self.replies.task.done()

    async def control(self, event: TimedChunk) -> None:
        if control_type(event) == INTERRUPT:
            await self.replies.interrupt(event.turn)
        else:
            pass


async def replying(
    inputs: AsyncIterator[TimedChunk], handle: ChunkHandler, replies: Replies
) -> None:
    """Read the input while a reply task runs.

    A failed reply fails the node at once; a reply cancelled by an interrupt
    control ends quietly; the end of the input waits for the running reply.
    The next input is requested only after the current chunk is handled, so
    work in `handle` counts as unprocessed input of the node.
    """
    pending: asyncio.Task[TimedChunk | None] = asyncio.ensure_future(
        anext(inputs, None)
    )
    try:
        while True:
            waiting = {pending} if replies.task is None else {pending, replies.task}
            await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
            if replies.task is not None and replies.task.done():
                if not replies.task.cancelled():
                    replies.task.result()
                else:
                    pass
                replies.task = None
            else:
                pass
            if not pending.done():
                continue
            else:
                pass
            chunk = pending.result()
            if chunk is None:
                await replies.finish()
                return
            else:
                await handle(chunk, replies)
            pending = asyncio.ensure_future(anext(inputs, None))
    finally:
        for task in (pending, replies.task):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.wait({task})
            else:
                pass


async def texts(inputs: AsyncIterator[TimedChunk]) -> AsyncIterator[str]:
    async for chunk in inputs:
        if chunk.format != TEXT_DELTA or not isinstance(chunk.payload, dict):
            raise ValueError("expected text delta chunks")
        else:
            yield str(chunk.payload["text"])


def sentence_end(text: str, max_words: int) -> int | None:
    """Where the first sentence of `text` ends: after a whole word with terminal punctuation or the max_words-th whole word, after a run of CJK terminal punctuation that something follows, or at the end of a line; None while it may still grow."""
    ends = []
    for count, word in enumerate(re.finditer(r"\S+(?=\s)", text), start=1):
        if SENTENCE_END.search(word.group()) or count >= max_words:
            ends.append(word.end())
            break
        else:
            pass
    cjk = CJK_SENTENCE_END.search(text)
    if cjk is not None:
        ends.append(cjk.end())
    else:
        pass
    line = re.search(r"\S[ \t]*\n", text)
    if line is not None:
        ends.append(line.start() + 1)
    else:
        pass
    return min(ends, default=None)


async def sentences(
    inputs: AsyncIterator[TimedChunk], max_words: int
) -> AsyncIterator[str]:
    """The reply text cut into sentences, each span with its own spacing, so the spans join back to the text."""
    pending = ""
    async for text in texts(inputs):
        pending += text
        while (end := sentence_end(pending, max_words)) is not None:
            yield pending[:end]
            pending = pending[end:]
    if pending:
        yield pending
    else:
        pass


class ComponentConfig(BaseModel):
    """Common base for component options; `kind` selects the implementation."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    kind: str
    timeout_s: float = Field(default=120, gt=0)


def bearer_headers(api_key_env: str) -> dict[str, str]:
    api_key = os.environ.get(api_key_env)
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


async def probe_models(url: str, timeout_s: float, api_key_env: str) -> None:
    """Require `GET url` to answer 200."""
    async with httpx.AsyncClient(
        timeout=min(timeout_s, PROBE_TIMEOUT_S),
        headers=bearer_headers(api_key_env),
    ) as client:
        try:
            response = await client.get(url)
        except httpx.TransportError as error:
            raise RuntimeError(f"cannot reach {url}: {error!r}") from error
    if response.status_code != 200:
        raise RuntimeError(f"{url} answered {response.status_code}")
    else:
        pass


class ServiceConfig(Protocol):
    base_url: str
    timeout_s: float
    api_key_env: str


async def probe_openai_service(config: ServiceConfig) -> None:
    await probe_models(
        f"{config.base_url.rstrip('/')}/models",
        config.timeout_s,
        config.api_key_env,
    )
