# SPDX-License-Identifier: Apache-2.0
"""Own session-local flow and vocoder state."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sglang_omni.models.minicpm_o.components.code2wav import (
    OUTPUT_SAMPLE_RATE,
    MiniCPMOCode2Wav,
)
from sglang_omni.models.minicpm_o.components.token2wav.vocoder import (
    SILENCE_TOKEN_ID,
    SpeakerPrompt,
    StreamCaches,
    Token2Wav,
)
from sglang_omni.proto.session import ResourceUsage
from sglang_omni.scheduling.speaker_cache import estimate_cache_bytes

SILENCE_PREFIX_LENGTH = 3
CODEC_CHUNK_SIZE = 25


def clone_caches(caches: StreamCaches) -> StreamCaches:
    flow_cache, hift_cache = caches
    return (
        {name: tensor.clone() for name, tensor in flow_cache.items()},
        {name: tensor.clone() for name, tensor in hift_cache.items()},
    )


@dataclass(kw_only=True)
class SharedSpeaker:
    """Stream caches prefilled from one reference voice, shared by its open sessions."""

    reference_key: str
    prompt: SpeakerPrompt
    # note (Junnan Li): Every turn decodes on a copy, so these caches are never written after prefill.
    base_caches: StreamCaches
    session_ids: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class MiniCPMOVocoderSessionState:
    session_id: str
    speaker: SharedSpeaker
    caches: StreamCaches
    pre_lookahead_tokens: int
    pending_codec_token_ids: list[int] = field(
        default_factory=lambda: [SILENCE_TOKEN_ID] * SILENCE_PREFIX_LENGTH
    )
    has_pending_turn: bool = False

    def held(self) -> ResourceUsage:
        size = estimate_cache_bytes((self.caches, self.pending_codec_token_ids))
        # note (Junnan Li): Shared caches are charged once, to the voice's earliest open session.
        if self.speaker.session_ids[0] == self.session_id:
            size += estimate_cache_bytes(self.speaker.base_caches)
        else:
            pass
        return ResourceUsage(slots={"tts": 1}, bytes=size)


class MiniCPMOVocoderRuntime:
    """Own streaming vocoder state independently per session."""

    def __init__(self, code2wav: MiniCPMOCode2Wav) -> None:
        self.code2wav = code2wav
        self.token2wav: Token2Wav = code2wav.token2wav
        self.sessions: dict[str, MiniCPMOVocoderSessionState] = {}
        self.speakers: dict[str, SharedSpeaker] = {}

    def open_session(
        self, session_id: str, *, reference_audio: bytes
    ) -> MiniCPMOVocoderSessionState:
        if session_id in self.sessions:
            raise ValueError(f"TTS session {session_id!r} is already open")
        else:
            pass
        reference_key, _ = self.code2wav.resolve_reference_key(reference_audio)
        speaker = self.speakers.get(reference_key)
        if speaker is None:
            (prompt,) = self.code2wav.prepare_references([reference_audio])
            speaker = SharedSpeaker(
                reference_key=reference_key,
                prompt=prompt,
                base_caches=self.token2wav.open_stream(prompt),
            )
            self.speakers[reference_key] = speaker
        else:
            pass
        speaker.session_ids.append(session_id)
        state = MiniCPMOVocoderSessionState(
            session_id=session_id,
            speaker=speaker,
            caches=clone_caches(speaker.base_caches),
            pre_lookahead_tokens=self.token2wav.flow.pre_lookahead_len,
        )
        self.sessions[session_id] = state
        return state

    def synthesize(
        self,
        session_id: str,
        codec_tokens: list[int],
        *,
        is_turn_start: bool,
        end_of_turn: bool = False,
    ) -> np.ndarray | None:
        state = self.sessions[session_id]
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
                force_flush=is_turn_start,
                is_last_chunk=end_of_turn,
            )
        if end_of_turn:
            self.reset_turn_state(state)
        else:
            pass
        return waveform

    def close_session(self, session_id: str) -> None:
        speaker = self.sessions.pop(session_id).speaker
        speaker.session_ids.remove(session_id)
        if not speaker.session_ids:
            self.speakers.pop(speaker.reference_key)
        else:
            pass

    def held(self, session_id: str) -> ResourceUsage:
        return self.sessions[session_id].held()

    def decode_audio_tokens(
        self,
        state: MiniCPMOVocoderSessionState,
        token_ids: list[int],
        *,
        force_flush: bool,
        is_last_chunk: bool,
    ) -> np.ndarray | None:
        state.pending_codec_token_ids.extend(token_ids)
        pcm_chunks: list[bytes] = []
        minimum_flush_tokens = state.pre_lookahead_tokens + 5
        window_tokens = CODEC_CHUNK_SIZE + state.pre_lookahead_tokens

        if force_flush:
            while len(state.pending_codec_token_ids) >= minimum_flush_tokens:
                window_length = min(window_tokens, len(state.pending_codec_token_ids))
                pcm_chunks.append(
                    self.stream(state, state.pending_codec_token_ids[:window_length])
                )
                consumed_tokens = min(
                    CODEC_CHUNK_SIZE, window_length - state.pre_lookahead_tokens
                )
                del state.pending_codec_token_ids[:consumed_tokens]
        else:
            while len(state.pending_codec_token_ids) >= window_tokens:
                pcm_chunks.append(
                    self.stream(state, state.pending_codec_token_ids[:window_tokens])
                )
                del state.pending_codec_token_ids[:CODEC_CHUNK_SIZE]

        if is_last_chunk and state.pending_codec_token_ids:
            pcm_chunks.append(
                self.stream(
                    state, list(state.pending_codec_token_ids), is_last_chunk=True
                )
            )
            state.pending_codec_token_ids.clear()
        else:
            pass
        pcm = b"".join(pcm_chunks)
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
        is_last_chunk: bool = False,
    ) -> bytes:
        pcm, state.caches = self.token2wav.stream(
            tokens, state.speaker.prompt, state.caches, is_last_chunk=is_last_chunk
        )
        return pcm

    def reset_turn_state(self, state: MiniCPMOVocoderSessionState) -> None:
        state.has_pending_turn = False
        state.pending_codec_token_ids = [SILENCE_TOKEN_ID] * SILENCE_PREFIX_LENGTH
        state.caches = clone_caches(state.speaker.base_caches)


__all__ = [
    "CODEC_CHUNK_SIZE",
    "SILENCE_PREFIX_LENGTH",
    "MiniCPMOVocoderRuntime",
    "MiniCPMOVocoderSessionState",
    "SharedSpeaker",
]
