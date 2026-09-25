# SPDX-License-Identifier: Apache-2.0
"""Own session-local flow and vocoder state."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sglang_omni.models.minicpm_o.components.token2wav.vocoder import (
    SpeakerPrompt,
    StreamCaches,
    Token2Wav,
)
from sglang_omni.proto.session import ResourceUsage
from sglang_omni.scheduling.speaker_cache import estimate_cache_bytes

SILENCE_TOKEN_ID = 4218
SILENCE_PREFIX_LENGTH = 3
CODEC_CHUNK_SIZE = 25
OUTPUT_SAMPLE_RATE = 24000


def clone_caches(caches: StreamCaches) -> StreamCaches:
    flow_cache, hift_cache = caches
    return (
        {key: value.clone() for key, value in flow_cache.items()},
        {key: value.clone() for key, value in hift_cache.items()},
    )


@dataclass(kw_only=True)
class MiniCPMOVocoderSessionState:
    prompt: SpeakerPrompt
    # note (Junnan Li): Every turn decodes on a copy so the prompt caches stay pristine for the next reset.
    base_caches: StreamCaches
    caches: StreamCaches
    pre_lookahead: int
    token2wav_buffer: list[int] = field(
        default_factory=lambda: [SILENCE_TOKEN_ID] * SILENCE_PREFIX_LENGTH
    )
    has_pending_turn: bool = False

    def held(self) -> ResourceUsage:
        size = estimate_cache_bytes(
            (self.prompt, self.base_caches, self.caches, self.token2wav_buffer)
        )
        return ResourceUsage(slots={"tts": 1}, bytes=size)


class MiniCPMOVocoderRuntime:
    """Own streaming vocoder state independently per session."""

    def __init__(
        self, token2wav: Token2Wav, *, codec_chunk_size: int = CODEC_CHUNK_SIZE
    ) -> None:
        if codec_chunk_size <= 0:
            raise ValueError("codec_chunk_size must be positive")
        else:
            pass
        self.token2wav = token2wav
        self.codec_chunk_size = codec_chunk_size
        self.sessions: dict[str, MiniCPMOVocoderSessionState] = {}

    def open_session(
        self, session_id: str, *, prompt_wav: str
    ) -> MiniCPMOVocoderSessionState:
        if session_id in self.sessions:
            raise ValueError(f"TTS session {session_id!r} is already open")
        else:
            pass
        prompt = self.token2wav.prepare_prompt(prompt_wav)
        base_caches = self.token2wav.open_stream(prompt)
        state = MiniCPMOVocoderSessionState(
            prompt=prompt,
            base_caches=base_caches,
            caches=clone_caches(base_caches),
            pre_lookahead=self.token2wav.flow.pre_lookahead_len,
        )
        self.sessions[session_id] = state
        return state

    def synthesize(
        self,
        session_id: str,
        codec_tokens: list[int],
        *,
        turn_start: bool,
        end_of_turn: bool = False,
    ) -> np.ndarray | None:
        state = self.sessions[session_id]
        try:
            if codec_tokens:
                state.has_pending_turn = True
            else:
                pass
            if not state.has_pending_turn:
                waveform = None
            else:
                waveform = self.decode_audio_tokens(
                    state,
                    codec_tokens,
                    force_flush=turn_start,
                    is_last_chunk=end_of_turn,
                )
            if end_of_turn:
                self.reset_turn_state(state)
            else:
                pass
            return waveform
        except Exception:
            self.reset_turn_state(state)
            raise

    def close_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    def held(self, session_id: str) -> ResourceUsage:
        state = self.sessions.get(session_id)
        return ResourceUsage() if state is None else state.held()

    def decode_audio_tokens(
        self,
        state: MiniCPMOVocoderSessionState,
        token_ids: list[int],
        *,
        force_flush: bool,
        is_last_chunk: bool,
    ) -> np.ndarray | None:
        state.token2wav_buffer.extend(token_ids)
        chunks: list[bytes] = []
        minimum = state.pre_lookahead + 5
        amount = self.codec_chunk_size + state.pre_lookahead

        if force_flush:
            while len(state.token2wav_buffer) >= minimum:
                current = min(amount, len(state.token2wav_buffer))
                chunks.append(self.stream(state, state.token2wav_buffer[:current]))
                consumed = min(self.codec_chunk_size, current - state.pre_lookahead)
                del state.token2wav_buffer[:consumed]
        else:
            while len(state.token2wav_buffer) >= amount:
                chunks.append(self.stream(state, state.token2wav_buffer[:amount]))
                del state.token2wav_buffer[: self.codec_chunk_size]

        if is_last_chunk and state.token2wav_buffer:
            chunks.append(
                self.stream(state, list(state.token2wav_buffer), last_chunk=True)
            )
            state.token2wav_buffer.clear()
        else:
            pass
        pcm = b"".join(chunks)
        if not pcm:
            return None
        else:
            waveform = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            if not is_last_chunk and waveform.size < OUTPUT_SAMPLE_RATE:
                waveform = np.pad(waveform, (OUTPUT_SAMPLE_RATE - waveform.size, 0))
            else:
                pass
            return waveform

    def stream(
        self,
        state: MiniCPMOVocoderSessionState,
        tokens: list[int],
        *,
        last_chunk: bool = False,
    ) -> bytes:
        pcm, state.caches = self.token2wav.stream(
            tokens, state.prompt, state.caches, last_chunk=last_chunk
        )
        return pcm

    def reset_turn_state(self, state: MiniCPMOVocoderSessionState) -> None:
        state.has_pending_turn = False
        state.token2wav_buffer = [SILENCE_TOKEN_ID] * SILENCE_PREFIX_LENGTH
        state.caches = clone_caches(state.base_caches)


__all__ = [
    "CODEC_CHUNK_SIZE",
    "OUTPUT_SAMPLE_RATE",
    "SILENCE_PREFIX_LENGTH",
    "SILENCE_TOKEN_ID",
    "MiniCPMOVocoderRuntime",
    "MiniCPMOVocoderSessionState",
]
