# SPDX-License-Identifier: Apache-2.0
"""Per-session state of the stages around the LM: the prompt handed on with the
first unit, the streaming Mimi encoder and the streaming Mimi decoder."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from sglang_omni.models.personaplex.architecture import (
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    SILENCE_CODES,
)
from sglang_omni.models.personaplex.components.mimi import MimiCodec, MimiEncodeState
from sglang_omni.models.personaplex.mimi_decode_graph import MimiDecodeSlots
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.models.personaplex.timeline import no_codes
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.proto.streams import AUDIO, PCM16, PCM16_BYTES_PER_SAMPLE
from sglang_omni.scheduling.session import (
    BatchedSessionHooks,
    SessionAppend,
    SessionContext,
    SessionHooks,
)
from sglang_omni.serve.realtime.types import samples_to_ms

PCM16_SCALE = 32768.0
PCM16_BYTES_PER_FRAME = SAMPLES_PER_FRAME * PCM16_BYTES_PER_SAMPLE
# Note (wilsonzheng0327): A session's first unit returns two agent frames, the
# prompt's last among them, and every later unit one.
FRAMES_PER_UNIT = (2, 1)


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
    slot_index: int
    num_samples: int = 0


class Code2WavHooks(BatchedSessionHooks):
    """Decodes the ready sessions' agent frames on their decode slots, one batch per
    frame count, and emits each unit's audio with the words spoken in it."""

    def __init__(self, decode_slots: MimiDecodeSlots) -> None:
        self.decode_slots = decode_slots
        self.states: dict[SessionIdentity, Code2WavState] = {}

    @torch.inference_mode()
    def warm_up_serving_thread(self) -> None:
        """Decode silence as a session's units arrive, on a slot freed again, so the
        codec's first use on the serving thread happens before the first session."""
        slot_index = self.decode_slots.acquire()
        silence = torch.tensor(
            SILENCE_CODES, dtype=torch.long, device=self.decode_slots.codec.device
        )
        for num_frames in FRAMES_PER_UNIT:
            self.decode_slots.decode_step(
                silence.repeat(num_frames, 1).T[None], [slot_index]
            )
        self.decode_slots.release(slot_index)

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        # Note (Junnan Li): The stage opens at most as many sessions as it has slots.
        self.states[session_identity] = Code2WavState(
            slot_index=self.decode_slots.acquire()
        )

    def append_batch(self, appends: list[SessionAppend]) -> list[StagePayload]:
        units = [PersonaPlexState.from_dict(append.payload.data) for append in appends]
        pcm = [b""] * len(appends)
        rows_by_width: dict[int, list[int]] = defaultdict(list)
        for index, unit in enumerate(units):
            if unit.codes.shape[0]:
                rows_by_width[unit.codes.shape[0]].append(index)
            else:
                pass
        for indices in rows_by_width.values():
            if len(indices) == 1:
                codes_BKF = units[indices[0]].codes.T[None]
            else:
                codes_BKF = torch.stack([units[index].codes.T for index in indices])
            waveform = self.decode_slots.decode_step(
                codes_BKF.to(device=self.decode_slots.codec.device),
                [
                    self.states[appends[index].context.session_identity].slot_index
                    for index in indices
                ],
            )
            for row, index in enumerate(indices):
                pcm[index] = waveform_to_pcm16(waveform[row, 0])
        for append, unit, unit_pcm in zip(appends, units, pcm, strict=True):
            session = self.states[append.context.session_identity]
            num_samples = len(unit_pcm) // PCM16_BYTES_PER_SAMPLE
            append.context.emit(
                TimedChunk(
                    AUDIO,
                    samples_to_ms(session.num_samples, SAMPLE_RATE),
                    samples_to_ms(num_samples, SAMPLE_RATE),
                    append.chunk.seq,
                    {"text": unit.text, "pcm": unit_pcm},
                    eos=append.chunk.eos,
                )
            )
            session.num_samples += num_samples
            append.payload.data = {}
        return [append.payload for append in appends]

    def close(self, session_identity: SessionIdentity) -> None:
        self.decode_slots.release(self.states.pop(session_identity).slot_index)


__all__ = [
    "Code2WavHooks",
    "MimiEncodeHooks",
    "PromptPreparer",
    "PreprocessingHooks",
    "encode_waveform",
    "pcm16_to_waveform",
    "waveform_to_pcm16",
]
