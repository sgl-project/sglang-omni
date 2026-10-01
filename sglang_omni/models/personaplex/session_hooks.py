# SPDX-License-Identifier: Apache-2.0
"""Per-call state of the stages around the LM in a full-duplex session.

Each hook serves one stage of the realtime route: preprocessing prepares the
prompt at open and turns each unit's PCM16 into samples, Mimi encode keeps the
call's streaming encoder state, and code2wav keeps its streaming decoder state
and emits the agent's audio and spoken words for each unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from sentencepiece import SentencePieceProcessor

from sglang_omni.models.personaplex.architecture import (
    AUDIO_CODEBOOKS_PER_STREAM,
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    TEXT_MARKER_IDS,
)
from sglang_omni.models.personaplex.components.mimi import (
    MimiCodec,
    MimiDecodeState,
    MimiEncodeState,
)
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import SessionContext, SessionHooks

PCM16_FORMAT = "pcm16"
PCM16_SCALE = 32768.0
PCM16_BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2
MS_PER_SECOND = 1000
WORD_BOUNDARY = "▁"


def pcm16_to_waveform(chunk: TimedChunk) -> torch.Tensor:
    """A unit's PCM16 payload as float32 samples; whole 80 ms frames only."""
    payload = chunk.payload
    if chunk.format != PCM16_FORMAT or not isinstance(payload, bytes):
        raise ValueError(f"PersonaPlex sessions take {PCM16_FORMAT} audio units")
    elif len(payload) % PCM16_BYTES_PER_FRAME:
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


def samples_to_ms(num_samples: int) -> float:
    return num_samples * MS_PER_SECOND / SAMPLE_RATE


def no_codes() -> torch.Tensor:
    return torch.zeros(0, AUDIO_CODEBOOKS_PER_STREAM, dtype=torch.long)


def encode_waveform(
    codec: MimiCodec, device: torch.device, waveform: torch.Tensor
) -> torch.Tensor:
    """A whole waveform → codes [F, 8] on the CPU."""
    codes = codec.encode(waveform.to(device=device, dtype=torch.float32).view(1, 1, -1))
    return codes[0].T.cpu()


class PromptPreparer(Protocol):
    def __call__(self, request: OmniRequest) -> PersonaPlexState: ...


class PromptSessionHooks(SessionHooks):
    """Prepares the prompt at open; the call's first unit carries it to the LM."""

    def __init__(self, prepare_prompt: PromptPreparer) -> None:
        self.prepare_prompt = prepare_prompt
        self.prompts: dict[SessionIdentity, PersonaPlexState] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.prompts[session_identity] = self.prepare_prompt(request)

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        state = self.prompts.pop(context.session_identity, None)
        if state is None:
            state = PersonaPlexState()
        else:
            state.carries_prompt = True
        state.waveform = pcm16_to_waveform(chunk)
        payload.data = state.to_dict()
        return payload

    def close(self, session_identity: SessionIdentity) -> None:
        # Note (wilsonzheng0327): The first unit takes the prompt, so a call that
        # carried any audio has nothing left here.
        self.prompts.pop(session_identity, None)


class MimiEncodeSessionHooks(SessionHooks):
    """Encodes each unit with the call's streaming encoder state."""

    def __init__(self, codec: MimiCodec, device: torch.device) -> None:
        self.codec = codec
        self.device = device
        self.encode_states: dict[SessionIdentity, MimiEncodeState] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.encode_states[session_identity] = self.codec.init_encode_state()

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
        waveform = state.waveform
        assert waveform is not None, "preprocessing sets every unit's waveform"
        if waveform.numel() == 0:
            state.user_codes = no_codes()
        else:
            codes = self.codec.encode_step(
                waveform.to(device=self.device, dtype=torch.float32).view(1, 1, -1),
                self.encode_states[context.session_identity],
            )
            state.user_codes = codes[0].T.cpu()
        state.waveform = None
        payload.data = state.to_dict()
        return payload

    def close(self, session_identity: SessionIdentity) -> None:
        self.encode_states.pop(session_identity)


@dataclass(kw_only=True)
class Code2WavCall:
    decode_state: MimiDecodeState
    num_samples: int = 0
    has_spoken: bool = False


class Code2WavSessionHooks(SessionHooks):
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
        self.calls: dict[SessionIdentity, Code2WavCall] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.calls[session_identity] = Code2WavCall(
            decode_state=self.codec.init_decode_state()
        )

    def spoken_words(self, call: Code2WavCall, text_ids: list[int]) -> str:
        """Word pieces as they are spoken, frame markers dropped."""
        # Note (wilsonzheng0327): Piece by piece, as the reference server streams
        # text; decoding a growing id list could rewrite text already sent.
        pieces = []
        for token in text_ids:
            if token in TEXT_MARKER_IDS:
                continue
            else:
                pass
            piece = self.tokenizer.id_to_piece(token).replace(WORD_BOUNDARY, " ")
            if not call.has_spoken:
                piece = piece.lstrip(" ")
                call.has_spoken = bool(piece)
            else:
                pass
            pieces.append(piece)
        return "".join(pieces)

    @torch.inference_mode()
    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        call = self.calls[context.session_identity]
        state = PersonaPlexState.from_dict(payload.data)
        codes = state.codes
        assert codes is not None, "the LM stage sets every unit's codes"
        start_ms = samples_to_ms(call.num_samples)
        num_samples = 0
        if codes.shape[0]:
            waveform = self.codec.decode_step(
                codes.to(device=self.device, dtype=torch.long).T[None],
                call.decode_state,
            )[0, 0]
            num_samples = int(waveform.shape[-1])
            context.emit(
                TimedChunk(
                    "audio",
                    start_ms,
                    samples_to_ms(num_samples),
                    chunk.seq,
                    waveform_to_pcm16(waveform),
                    format=PCM16_FORMAT,
                )
            )
        else:
            pass
        words = self.spoken_words(call, state.text_ids)
        if words:
            context.emit(
                TimedChunk(
                    "text",
                    start_ms,
                    samples_to_ms(num_samples),
                    chunk.seq,
                    {"text": words},
                )
            )
        else:
            pass
        call.num_samples += num_samples
        if chunk.eos:
            context.emit(
                TimedChunk(
                    "audio",
                    samples_to_ms(call.num_samples),
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
        self.calls.pop(session_identity)


__all__ = [
    "Code2WavSessionHooks",
    "MimiEncodeSessionHooks",
    "PromptPreparer",
    "PromptSessionHooks",
    "encode_waveform",
    "no_codes",
    "pcm16_to_waveform",
    "waveform_to_pcm16",
]
