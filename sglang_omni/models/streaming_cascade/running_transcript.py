# SPDX-License-Identifier: Apache-2.0
"""Utterance transcripts cut at the speech end or a commit from an ASR stage that streams one running text."""

from collections.abc import AsyncIterator

from pydantic import Field

from sglang_omni.models.streaming_cascade.base import (
    ComponentConfig,
    StreamComponent,
    transcript_chunk,
)
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import (
    COMMIT,
    PCM16,
    RUNNING_TEXT,
    SPEECH_STOPPED,
    control_type,
)
from sglang_omni.scheduling.session import ChunkEmitter

TRAILING_PUNCTUATION = " .,!?;:"


class RunningTranscriptConfig(ComponentConfig):
    """Utterances cut from an ASR stage's one running text at the VAD's speech end or a client commit."""

    kind: str = "running_transcript"
    tail_ms: float = Field(default=400, ge=0)


class RunningTranscript(StreamComponent):
    def __init__(self, config: RunningTranscriptConfig) -> None:
        self.config = config
        self.full_text = ""
        self.committed_chars = 0
        self.clock_ms = 0.0
        self.finalize_at_ms: float | None = None

    def adopt(self, previous: StreamComponent) -> None:
        if isinstance(previous, RunningTranscript):
            self.full_text = previous.full_text
            self.committed_chars = previous.committed_chars
            self.clock_ms = previous.clock_ms
            self.finalize_at_ms = previous.finalize_at_ms
        else:
            pass

    def state_bytes(self) -> int:
        return len(self.full_text) * 4

    def running(self) -> str:
        # Note (Junnan Li): The recognizer closes a sentence only once the next one starts, so an utterance may begin with the previous one's punctuation.
        return (
            self.full_text[self.committed_chars :].lstrip(TRAILING_PUNCTUATION).strip()
        )

    async def process(
        self, inputs: AsyncIterator[TimedChunk], emit: ChunkEmitter
    ) -> None:
        async for chunk in self.run(inputs):
            emit(chunk)
        if self.running():
            emit(transcript_chunk(self.running(), True))
            self.committed_chars = len(self.full_text)
        else:
            pass

    async def run(self, inputs: AsyncIterator[TimedChunk]) -> AsyncIterator[TimedChunk]:
        async for chunk in inputs:
            if chunk.format == RUNNING_TEXT:
                self.full_text = str(chunk.payload["full_text"])
                if chunk.payload["text"] and self.running():
                    yield transcript_chunk(self.running(), False)
                else:
                    pass
            elif chunk.format == PCM16:
                self.clock_ms = chunk.t_start_ms + chunk.duration_ms
            elif control_type(chunk) == SPEECH_STOPPED:
                self.finalize_at_ms = float(chunk.payload["t_ms"]) + self.config.tail_ms
            elif control_type(chunk) == COMMIT:
                self.finalize_at_ms = self.clock_ms + self.config.tail_ms
            else:
                pass
            if self.finalize_at_ms is not None and self.clock_ms >= self.finalize_at_ms:
                self.finalize_at_ms = None
                yield transcript_chunk(self.running(), True)
                self.committed_chars = len(self.full_text)
            else:
                pass


def build_running_transcript(config: RunningTranscriptConfig) -> RunningTranscript:
    return RunningTranscript(config)
