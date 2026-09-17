# SPDX-License-Identifier: Apache-2.0
"""The delayed token timeline one PersonaPlex request runs through.

The model reads one row of 17 tokens per position (text, 8 agent codebooks,
8 user codebooks) and predicts the next row. Frames map onto positions with a
per-stream delay: stream k of frame f sits at position f + delay[k],
and positions 0..delay[k] hold the stream's initial token. This is the
reference's ring cache written out as a table, so the prefill can be one
extend and every decode step is a lookup.

A request's positions are:

    0 .. P-1     prompt (voice prompt, silence, text prompt, silence): forced
    P .. P+U-1   one per frame of the caller's audio: text and agent sampled

Row P-1+j is the input of forward j (j = 0 being the prefill's
last row); its prediction is row P+j. Output frame f is complete once
row f+1 exists: its first codebook lives at f, the rest at f+1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang_omni.models.personaplex.architecture import (
    AGENT_STREAM_OFFSET,
    AUDIO_CODEBOOKS_PER_STREAM,
    AUDIO_INITIAL_ID,
    DELAYS,
    MAX_DELAY,
    NUM_STREAMS,
    PROMPT_SILENCE_FRAMES,
    SILENCE_CODES,
    SINE_CODES,
    TEXT_INITIAL_ID,
    TEXT_PAD_ID,
    USER_STREAM_OFFSET,
)

# Note (wilsonzheng0327): Ring width of the reference generator, and so of a saved voice
# prompt's cache.
REFERENCE_CACHE_POSITIONS = MAX_DELAY + 3

UNKNOWN = -1


def delay_stream(
    frames_FK: torch.Tensor, delays: tuple[int, ...], initial: int
) -> torch.Tensor:
    """Frames [F, K] → positions [F, K] with d[p, k] = frames[p - delay_k, k]
    and the initial token wherever p <= delay_k."""
    num_frames, num_streams = frames_FK.shape
    assert num_streams == len(delays), (num_streams, len(delays))
    out = torch.full_like(frames_FK, initial)
    for k, delay in enumerate(delays):
        if delay < num_frames:
            out[delay + 1 :, k] = frames_FK[1 : num_frames - delay, k]
    return out


@dataclass
class PromptFrames:
    """Frame-level tokens of the prompt phase, in the order they are stepped."""

    text: torch.Tensor
    agent: torch.Tensor
    user: torch.Tensor
    voice_frames: int

    @property
    def num_frames(self) -> int:
        return int(self.text.shape[0])


def build_prompt_frames(
    *,
    voice_frames: int,
    text_prompt_ids: list[int],
    voice_codes: torch.Tensor | None = None,
) -> PromptFrames:
    """Voice prompt → silence → text prompt → silence, as the reference steps it.

    During the voice prompt the agent plays the voice and hears a sine; text
    is PAD. The text prompt is one token per frame over silence.
    """
    silence = torch.tensor(SILENCE_CODES, dtype=torch.long)
    sine = torch.tensor(SINE_CODES, dtype=torch.long)
    num_text = len(text_prompt_ids)
    total = voice_frames + PROMPT_SILENCE_FRAMES + num_text + PROMPT_SILENCE_FRAMES

    text = torch.full((total,), TEXT_PAD_ID, dtype=torch.long)
    start = voice_frames + PROMPT_SILENCE_FRAMES
    text[start : start + num_text] = torch.tensor(text_prompt_ids, dtype=torch.long)

    agent = silence.repeat(total, 1)
    if voice_codes is not None:
        assert voice_codes.shape == (
            voice_frames,
            AUDIO_CODEBOOKS_PER_STREAM,
        ), voice_codes.shape
        agent[:voice_frames] = voice_codes
    else:
        agent[:voice_frames] = UNKNOWN
    user = sine.repeat(total, 1)
    return PromptFrames(text=text, agent=agent, user=user, voice_frames=voice_frames)


def voice_tail_codes_from_cache(cache: torch.Tensor, voice_frames: int) -> torch.Tensor:
    """Recover the last two voice frames' codes from a saved prompt cache.

    A packaged voice ships the fused embeddings of every voice-prompt row plus
    the generator's ring cache at the end of the voice prompt. Only the codes
    of frames V-2 and V-1 still matter after that point, and the ring
    holds them: frame V-1 at position V-1 (delay 0) and V
    (delay 1), frame V-2's delayed codebooks at V-1.

    Returns [2, 8]: frame V-2 (first codebook UNKNOWN) and frame V-1.
    """
    cache = cache.reshape(NUM_STREAMS, -1)
    ring = cache.shape[-1]
    assert ring == REFERENCE_CACHE_POSITIONS, cache.shape
    tail = torch.full((2, AUDIO_CODEBOOKS_PER_STREAM), UNKNOWN, dtype=torch.long)
    for k in range(AUDIO_CODEBOOKS_PER_STREAM):
        stream = AGENT_STREAM_OFFSET + k
        delay = DELAYS[stream]
        tail[1, k] = cache[stream, (voice_frames - 1 + delay) % ring]
        if delay > 0:
            tail[0, k] = cache[stream, (voice_frames - 2 + delay) % ring]
    return tail


@dataclass
class Timeline:
    """Everything the LM stage needs to run one request.

    Attributes:
        prefill_tokens: [P, 17] rows for positions 0..P-1; UNKNOWN where
            a row is supplied as an embedding instead.
        prefill_embeddings: [E, dim] rows replacing prefill_tokens at
            prefill_embedding_positions.
        agent_row_before_start: agent codebooks at position P-1, the first
            codebook of output frame P-1.
        forced_agent_at_start: [8] agent codebooks at position P that
            the prompt already fixed (delayed streams), UNKNOWN where sampled.
        user_rows: [P+U, 8] user codebooks for every position.
        num_prompt_positions: P.
        num_frames: U, the caller's audio frames and the decode budget.
    """

    prefill_tokens: torch.Tensor
    prefill_embeddings: torch.Tensor | None
    prefill_embedding_positions: list[int]
    agent_row_before_start: torch.Tensor
    forced_agent_at_start: torch.Tensor
    user_rows: torch.Tensor
    num_prompt_positions: int
    num_frames: int

    def input_position(self, forward_index: int) -> int:
        """Position of the row forward j consumes (j = 0: last prefill row)."""
        return self.num_prompt_positions - 1 + forward_index


def build_timeline(
    prompt: PromptFrames,
    user_codes_UK: torch.Tensor,
    *,
    voice_embeddings: torch.Tensor | None = None,
    voice_tail_codes: torch.Tensor | None = None,
) -> Timeline:
    """Assemble the delayed timeline for one request.

    With voice_embeddings (a packaged voice) the voice-prompt rows are
    replaced by the stored fused embeddings; voice_tail_codes then fills the
    two voice frames the following rows still read.
    """
    num_prompt = prompt.num_frames
    num_frames = int(user_codes_UK.shape[0])
    agent = prompt.agent.clone()
    if voice_tail_codes is not None:
        assert voice_embeddings is not None
        for offset, frame in enumerate(
            (prompt.voice_frames - 2, prompt.voice_frames - 1)
        ):
            known = voice_tail_codes[offset] != UNKNOWN
            agent[frame, known] = voice_tail_codes[offset, known]

    agent_delays = DELAYS[AGENT_STREAM_OFFSET:USER_STREAM_OFFSET]
    user_delays = DELAYS[USER_STREAM_OFFSET:]
    text_rows = delay_stream(prompt.text[:, None], DELAYS[:1], TEXT_INITIAL_ID)
    agent_rows = delay_stream(agent, agent_delays, AUDIO_INITIAL_ID)
    user_frames = torch.cat([prompt.user, user_codes_UK], dim=0)
    user_rows = delay_stream(user_frames, user_delays, AUDIO_INITIAL_ID)

    prefill_tokens = torch.cat([text_rows, agent_rows, user_rows[:num_prompt]], dim=1)
    embedding_positions: list[int] = []
    if voice_embeddings is not None:
        # Note (wilsonzheng0327): Stored rows are the inputs of positions 0 .. V-2; the
        # reference never forwards the last voice row on its own.
        expected = prompt.voice_frames - 1
        if voice_embeddings.shape[0] != expected:
            raise ValueError(
                f"voice prompt has {voice_embeddings.shape[0]} stored rows but "
                f"{prompt.voice_frames} frames; expected {expected} rows"
            )
        embedding_positions = list(range(expected))
    unknown_rows = (prefill_tokens == UNKNOWN).any(dim=1)
    unknown_rows[embedding_positions] = False
    if bool(unknown_rows.any()):
        raise ValueError(
            "prompt rows without tokens or embeddings at positions "
            f"{unknown_rows.nonzero().flatten().tolist()}"
        )

    forced = torch.full((AUDIO_CODEBOOKS_PER_STREAM,), UNKNOWN, dtype=torch.long)
    for k, delay in enumerate(agent_delays):
        if delay > 0:
            forced[k] = agent[num_prompt - delay, k]

    return Timeline(
        prefill_tokens=prefill_tokens,
        prefill_embeddings=voice_embeddings,
        prefill_embedding_positions=embedding_positions,
        agent_row_before_start=agent_rows[num_prompt - 1].clone(),
        forced_agent_at_start=forced,
        user_rows=user_rows,
        num_prompt_positions=num_prompt,
        num_frames=num_frames,
    )


def output_frame(
    previous_agent_row: torch.Tensor, agent_row: torch.Tensor
) -> torch.Tensor:
    """Codes of the frame whose delayed codebooks arrived with agent_row."""
    frame = agent_row.clone()
    for k, delay in enumerate(DELAYS[AGENT_STREAM_OFFSET:USER_STREAM_OFFSET]):
        if delay == 0:
            frame[..., k] = previous_agent_row[..., k]
    return frame


__all__ = [
    "PromptFrames",
    "REFERENCE_CACHE_POSITIONS",
    "Timeline",
    "UNKNOWN",
    "build_prompt_frames",
    "build_timeline",
    "delay_stream",
    "output_frame",
    "voice_tail_codes_from_cache",
]
