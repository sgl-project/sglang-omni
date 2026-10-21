# SPDX-License-Identifier: Apache-2.0
"""Per-session state of the stages around the LM: the prompt handed on with the
first unit, the streaming Mimi encoder and the streaming Mimi decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from sentencepiece import SentencePieceProcessor

from sglang_omni.models.personaplex.architecture import (
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    SILENCE_CODES,
    TEXT_MARKER_IDS,
)
from sglang_omni.models.personaplex.components.mimi import (
    MimiCodec,
    MimiDecodeState,
    MimiEncodeState,
)
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.models.personaplex.timeline import no_codes
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.proto.streams import PCM16, PCM16_BYTES_PER_SAMPLE
from sglang_omni.scheduling.session import SessionContext, SessionHooks
from sglang_omni.serve.realtime.types import samples_to_ms

PCM16_SCALE = 32768.0
PCM16_BYTES_PER_FRAME = SAMPLES_PER_FRAME * PCM16_BYTES_PER_SAMPLE
WORD_BOUNDARY = "▁"
# Note (wilsonzheng0327): A session's first unit returns two agent frames, the
# prompt's last among them, and every later unit one.
MAX_FRAMES_PER_UNIT = 2
WARM_UP_FRAMES_PER_UNIT = (MAX_FRAMES_PER_UNIT, 1)


def is_empty_end_of_input(chunk: TimedChunk) -> bool:
    return chunk.eos and chunk.duration_ms == 0


def pcm16_to_waveform(chunk: TimedChunk) -> torch.Tensor:
    """A unit's PCM16 payload as float32 samples: whole 80 ms frames, at least
    one unless the unit is an empty end of input."""
    payload = chunk.payload
    if chunk.format != PCM16 or not isinstance(payload, bytes):
        raise ValueError(f"PersonaPlex sessions take {PCM16} audio units")
    elif len(payload) % PCM16_BYTES_PER_FRAME or (
        not payload and not is_empty_end_of_input(chunk)
    ):
        raise ValueError(
            f"PersonaPlex session units must hold whole 80 ms frames of "
            f"{SAMPLE_RATE} Hz audio, got {len(payload)} bytes"
        )
    else:
        samples = np.frombuffer(payload, dtype="<i2").astype(np.float32)
        return torch.from_numpy(samples / PCM16_SCALE)


def waveform_to_pcm16(waveform: torch.Tensor) -> bytes:
    samples = (waveform.float().clamp(-1.0, 1.0) * (PCM16_SCALE - 1)).round()
    return samples.to(torch.int16).cpu().numpy().astype("<i2").tobytes()


def encode_waveform(
    codec: MimiCodec, device: torch.device, waveform: torch.Tensor
) -> torch.Tensor:
    """A whole waveform → codes [F, 8] on the CPU."""
    codes = codec.encode(waveform.to(device=device, dtype=torch.float32).view(1, 1, -1))
    return codes[0].T.cpu()


class PromptPreparer(Protocol):
    def __call__(self, request: OmniRequest) -> PersonaPlexState: ...


class PreprocessingHooks(SessionHooks):
    """Prepares the prompt at open; the session's first unit carries it to the LM."""

    def __init__(self, prepare_prompt: PromptPreparer) -> None:
        self.prepare_prompt = prepare_prompt
        self.states: dict[SessionIdentity, PersonaPlexState] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.states[session_identity] = self.prepare_prompt(request)

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        state = self.states.pop(context.session_identity, None) or PersonaPlexState()
        state.waveform = pcm16_to_waveform(chunk)
        payload.data = state.to_dict()
        return payload

    def close(self, session_identity: SessionIdentity) -> None:
        # Note (wilsonzheng0327): The first unit takes the prompt, so a session that
        # carried any audio has nothing left here.
        self.states.pop(session_identity, None)


class MimiEncodeHooks(SessionHooks):
    """Encodes each unit with the session's streaming encoder state."""

    def __init__(self, codec: MimiCodec, device: torch.device) -> None:
        self.codec = codec
        self.device = device
        self.states: dict[SessionIdentity, MimiEncodeState] = {}

    @torch.inference_mode()
    def warm_up_serving_thread(self) -> None:
        """Encode a unit of silence on a throwaway state, so the codec's first use
        on the serving thread happens before the first session."""
        self.codec.encode_step(
            torch.zeros(1, 1, SAMPLES_PER_FRAME, device=self.device),
            self.codec.init_encode_state(batch_size=1),
        )

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.states[session_identity] = self.codec.init_encode_state(batch_size=1)

    @torch.inference_mode()
    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        state = PersonaPlexState.from_dict(payload.data)
        if state.voice_waveform is not None:
            state.voice_codes = encode_waveform(
                self.codec, self.device, state.voice_waveform
            )
            state.voice_waveform = None
        else:
            pass
        if is_empty_end_of_input(chunk):
            state.user_codes = no_codes()
        else:
            codes = self.codec.encode_step(
                state.waveform.to(device=self.device, dtype=torch.float32).view(
                    1, 1, -1
                ),
                self.states[context.session_identity],
            )
            state.user_codes = codes[0].T.cpu()
        state.waveform = None
        payload.data = state.to_dict()
        return payload

    def close(self, session_identity: SessionIdentity) -> None:
        self.states.pop(session_identity)


@dataclass(kw_only=True)
class Code2WavState:
    decode_state: MimiDecodeState
    num_samples: int = 0
    has_spoken: bool = False


class Code2WavHooks(SessionHooks):
    """Decodes each unit's agent frames and emits the audio and spoken words."""

    def __init__(
        self,
        codec: MimiCodec,
        device: torch.device,
        tokenizer: SentencePieceProcessor,
    ) -> None:
        self.codec = codec
        self.device = device
        self.tokenizer = tokenizer
        self.states: dict[SessionIdentity, Code2WavState] = {}

    @torch.inference_mode()
    def warm_up_serving_thread(self) -> None:
        """Decode silence as a session's units arrive, on a throwaway state, so the
        codec's first use on the serving thread happens before the first session."""
        decode_state = self.codec.init_decode_state(
            batch_size=1, max_chunk_frames=MAX_FRAMES_PER_UNIT
        )
        silence = torch.tensor(SILENCE_CODES, dtype=torch.long, device=self.device)
        for num_frames in WARM_UP_FRAMES_PER_UNIT:
            self.codec.decode_step(silence.repeat(num_frames, 1).T[None], decode_state)

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.states[session_identity] = Code2WavState(
            decode_state=self.codec.init_decode_state(
                batch_size=1, max_chunk_frames=MAX_FRAMES_PER_UNIT
            )
        )

    def spoken_words(self, session: Code2WavState, text_ids: list[int]) -> str:
        """Word pieces as they are spoken, frame markers dropped."""
        # Note (wilsonzheng0327): Piece by piece, since decoding a growing id list
        # could rewrite text already sent.
        pieces = []
        for token in text_ids:
            if token in TEXT_MARKER_IDS:
                continue
            else:
                pass
            piece = self.tokenizer.id_to_piece(token).replace(WORD_BOUNDARY, " ")
            if not session.has_spoken:
                piece = piece.lstrip(" ")
                session.has_spoken = bool(piece)
            else:
                pass
            pieces.append(piece)
        return "".join(pieces)

    @torch.inference_mode()
    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        session = self.states[context.session_identity]
        state = PersonaPlexState.from_dict(payload.data)
        codes = state.codes
        start_ms = samples_to_ms(session.num_samples, SAMPLE_RATE)
        num_samples = 0
        if codes.shape[0]:
            waveform = self.codec.decode_step(
                codes.to(device=self.device, dtype=torch.long).T[None],
                session.decode_state,
            )[0, 0]
            num_samples = int(waveform.shape[-1])
            context.emit(
                TimedChunk(
                    "audio",
                    start_ms,
                    samples_to_ms(num_samples, SAMPLE_RATE),
                    chunk.seq,
                    waveform_to_pcm16(waveform),
                    format=PCM16,
                )
            )
        else:
            pass
        words = self.spoken_words(session, state.text_ids)
        if words:
            context.emit(
                TimedChunk(
                    "text",
                    start_ms,
                    samples_to_ms(num_samples, SAMPLE_RATE),
                    chunk.seq,
                    {"text": words},
                )
            )
        else:
            pass
        session.num_samples += num_samples
        if chunk.eos:
            context.emit(
                TimedChunk(
                    "audio",
                    samples_to_ms(session.num_samples, SAMPLE_RATE),
                    0.0,
                    chunk.seq,
                    None,
                    eos=True,
                )
            )
        else:
            pass
        payload.data = {}
        return payload

    def close(self, session_identity: SessionIdentity) -> None:
        self.states.pop(session_identity)


__all__ = [
    "Code2WavHooks",
    "MimiEncodeHooks",
    "PromptPreparer",
    "PreprocessingHooks",
    "encode_waveform",
    "pcm16_to_waveform",
    "waveform_to_pcm16",
]
