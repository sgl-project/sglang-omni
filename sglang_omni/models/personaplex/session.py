# SPDX-License-Identifier: Apache-2.0
"""Full-duplex sessions: each 80 ms unit of caller audio is one frame through
preprocessing, Mimi encode, the LM and Mimi decode.

The LM keeps one SGLang streaming session per conversation. Its first unit
prefills the prompt and steps once; every later unit extends the retained
history by the one position whose text token the previous unit sampled, so
the KV cache grows by exactly one row per frame, as the offline timeline does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from sentencepiece import SentencePieceProcessor
from sglang.srt.managers.schedule_batch import Req

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
from sglang_omni.models.personaplex.prompts import VoicePrompt
from sglang_omni.models.personaplex.request_builders import (
    RequestSampling,
    lm_sampling_params,
    new_model_inputs,
    packaged_voice_for,
    prompt_from_state,
    prompt_input_ids,
    resolve_sampling,
)
from sglang_omni.models.personaplex.timeline import (
    Timeline,
    build_timeline,
    extend_user_rows,
)
from sglang_omni.proto import EXPLICIT_GENERATION_PARAMS_KEY, OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import SessionContext, SessionHooks
from sglang_omni.scheduling.sglang_backend.ar_session import ARSessionAdapter
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData

PCM16_FORMAT = "pcm16"
PCM16_SCALE = 32768.0
PCM16_BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2
MS_PER_SECOND = 1000


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
        pass
    samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / PCM16_SCALE
    return torch.from_numpy(samples)


def waveform_to_pcm16(waveform: torch.Tensor) -> bytes:
    samples = (waveform.float().clamp(-1.0, 1.0) * (PCM16_SCALE - 1)).round()
    return samples.to(torch.int16).cpu().numpy().astype("<i2").tobytes()


def samples_to_ms(num_samples: int) -> float:
    return num_samples * MS_PER_SECOND / SAMPLE_RATE


def no_codes() -> torch.Tensor:
    return torch.zeros(0, AUDIO_CODEBOOKS_PER_STREAM, dtype=torch.long)


class PromptPreparer(Protocol):
    def __call__(self, request: OmniRequest) -> PersonaPlexState: ...


class PromptSessionHooks(SessionHooks):
    """Prepares the prompt at open; the first unit carries it to the LM."""

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
        state.num_samples = int(state.waveform.shape[-1])
        payload.data = state.to_dict()
        return payload

    def close(self, session_identity: SessionIdentity) -> None:
        self.prompts.pop(session_identity, None)


def encode_waveform(
    codec: MimiCodec, device: torch.device, waveform: torch.Tensor
) -> torch.Tensor:
    """A whole waveform → codes [F, 8] on the CPU."""
    codes = codec.encode(waveform.to(device=device, dtype=torch.float32).view(1, 1, -1))
    return codes[0].T.cpu()


class MimiEncodeSessionHooks(SessionHooks):
    """Encodes each unit with the session's streaming encoder state."""

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
        if waveform is None or waveform.numel() == 0:
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
        self.encode_states.pop(session_identity, None)


@dataclass(kw_only=True)
class LMCheckpoint:
    """Where the last finished unit left the session; a failed unit rolls back here."""

    num_forwards: int = 0
    num_caller_frames: int = 0


@dataclass(kw_only=True)
class LMSession:
    sampling: RequestSampling
    timeline: Timeline | None = None
    user_frames: torch.Tensor | None = None
    model_inputs: dict | None = None
    checkpoint: LMCheckpoint | None = None
    pending: LMCheckpoint | None = None


class PersonaPlexSessionAdapter(ARSessionAdapter):
    """One LM request per unit, stepping once per caller frame the unit carries.

    Forward j reads caller frame j - 1, so after a unit that brings the
    caller to F frames the session has run F + 1 forwards: the first unit
    is the prompt prefill plus one decode, every later unit one extend.
    """

    def __init__(self, *, vocab_size: int, context_length: int) -> None:
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.voice_cache: dict[str, VoicePrompt] = {}
        self.sessions: dict[str, LMSession] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.sessions[session_identity.id] = LMSession(
            sampling=resolve_sampling(
                request.params,
                request.metadata.get(EXPLICIT_GENERATION_PARAMS_KEY) or (),
            )
        )

    def close(self, session_identity: SessionIdentity) -> None:
        self.sessions.pop(session_identity.id, None)

    def finish_input(
        self, session_identity: SessionIdentity, payload: StagePayload
    ) -> StagePayload | None:
        payload.data = PersonaPlexState(codes=no_codes()).to_dict()
        return payload

    def start(self, session: LMSession, state: PersonaPlexState) -> None:
        """Lay out the prompt from the first unit's payload."""
        if not state.carries_prompt:
            raise ValueError(
                "PersonaPlex session prompt is missing; the first unit carries it"
            )
        else:
            pass
        prompt, voice = prompt_from_state(
            state, packaged_voice_for(state, self.voice_cache)
        )
        timeline = build_timeline(
            prompt,
            no_codes(),
            voice_embeddings=voice.embeddings,
            voice_tail_codes=voice.tail_codes,
        )
        session.timeline = timeline
        session.user_frames = prompt.user
        session.model_inputs = new_model_inputs(timeline, session.sampling)
        session.checkpoint = LMCheckpoint()

    def roll_back(self, session: LMSession) -> None:
        """Drop what an unfinished unit appended; SGLang kept only finished units."""
        timeline = session.timeline
        model_inputs = session.model_inputs
        checkpoint = session.checkpoint
        assert timeline is not None and model_inputs is not None
        assert session.user_frames is not None and checkpoint is not None
        num_rows = timeline.num_prompt_positions + checkpoint.num_caller_frames
        session.user_frames = session.user_frames[:num_rows]
        timeline.user_rows = timeline.user_rows[:num_rows]
        model_inputs.pop("device_user_rows", None)
        del model_inputs["agent_rows"][checkpoint.num_forwards :]
        model_inputs["frames"].clear()
        model_inputs["pending_frames"].clear()
        session.pending = None

    def build(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> SGLangARRequestData:
        session = self.sessions[session_identity.id]
        state = PersonaPlexState.from_dict(payload.data)
        if session.timeline is None:
            self.start(session, state)
        elif session.pending is not None:
            self.roll_back(session)
        else:
            pass
        timeline = session.timeline
        checkpoint = session.checkpoint
        assert timeline is not None and checkpoint is not None
        assert session.user_frames is not None and session.model_inputs is not None
        caller_codes = state.user_codes if state.user_codes is not None else no_codes()
        if caller_codes.shape[0] == 0:
            raise ValueError("PersonaPlex session unit carries no 80 ms frame")
        else:
            pass

        num_caller_frames = checkpoint.num_caller_frames + int(caller_codes.shape[0])
        num_forwards = num_caller_frames + 1
        positions = timeline.num_prompt_positions + num_forwards
        if positions > self.context_length - 1:
            raise ValueError(
                f"PersonaPlex session needs {positions} positions but the LM context "
                f"holds {self.context_length - 1}; raise the lm stage's context_length"
            )
        else:
            pass
        session.user_frames = torch.cat(
            [session.user_frames, caller_codes.to(torch.long)], dim=0
        )
        timeline.user_rows = extend_user_rows(session.user_frames, timeline.user_rows)
        session.pending = LMCheckpoint(
            num_forwards=num_forwards, num_caller_frames=num_caller_frames
        )

        # Note (wilsonzheng0327): After the first unit SGLang restores the history
        # itself, so a unit sends no new ids: the one position it extends is the
        # previous unit's last sampled token.
        input_ids = prompt_input_ids(timeline) if checkpoint.num_forwards == 0 else []
        max_new_tokens = num_forwards - checkpoint.num_forwards
        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=input_ids,
            sampling_params=lm_sampling_params(session.sampling, max_new_tokens),
            vocab_size=self.vocab_size,
        )
        data = SGLangARRequestData(
            req=req,
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            stage_payload=payload,
            max_new_tokens=max_new_tokens,
            temperature=session.sampling.text_temperature,
        )
        data.talker_model_inputs = session.model_inputs
        return data

    def result(
        self, session_identity: SessionIdentity, request_data: SGLangARRequestData
    ) -> StagePayload:
        session = self.sessions[session_identity.id]
        assert session.model_inputs is not None and session.pending is not None
        frames = session.model_inputs["pending_frames"]
        codes = torch.stack(frames).cpu() if frames else no_codes()
        frames.clear()
        session.model_inputs["frames"].clear()
        session.checkpoint = session.pending
        session.pending = None
        payload = request_data.stage_payload
        payload.data = PersonaPlexState(
            codes=codes, text_ids=[int(i) for i in request_data.output_ids]
        ).to_dict()
        return payload


@dataclass(kw_only=True)
class Code2WavSession:
    decode_state: MimiDecodeState
    num_samples: int = 0
    has_spoken: bool = False


class Code2WavSessionHooks(SessionHooks):
    """Decodes each unit's agent frames and emits the audio and spoken text."""

    def __init__(
        self,
        codec: MimiCodec,
        device: torch.device,
        tokenizer: SentencePieceProcessor,
    ) -> None:
        self.codec = codec
        self.device = device
        self.tokenizer = tokenizer
        self.sessions: dict[SessionIdentity, Code2WavSession] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.sessions[session_identity] = Code2WavSession(
            decode_state=self.codec.init_decode_state()
        )

    def spoken_text(self, session: Code2WavSession, text_ids: list[int]) -> str:
        """Word pieces as they are spoken; markers are dropped."""
        pieces = []
        for token in text_ids:
            if int(token) in TEXT_MARKER_IDS:
                continue
            else:
                pass
            piece = self.tokenizer.id_to_piece(int(token)).replace("▁", " ")
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
        session = self.sessions[context.session_identity]
        state = PersonaPlexState.from_dict(payload.data)
        codes = state.codes if state.codes is not None else no_codes()
        start_ms = samples_to_ms(session.num_samples)
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
                    samples_to_ms(num_samples),
                    chunk.seq,
                    waveform_to_pcm16(waveform),
                    format=PCM16_FORMAT,
                )
            )
        else:
            pass
        text = self.spoken_text(session, state.text_ids)
        if text:
            context.emit(
                TimedChunk(
                    "text",
                    start_ms,
                    samples_to_ms(num_samples),
                    chunk.seq,
                    {"text": text},
                )
            )
        else:
            pass
        session.num_samples += num_samples
        if chunk.eos:
            context.emit(
                TimedChunk(
                    "audio",
                    samples_to_ms(session.num_samples),
                    0.0,
                    chunk.seq,
                    None,
                    format=PCM16_FORMAT,
                    eos=True,
                )
            )
        else:
            pass
        payload.data = {"num_samples": num_samples}
        return payload

    def close(self, session_identity: SessionIdentity) -> None:
        self.sessions.pop(session_identity, None)


__all__ = [
    "Code2WavSessionHooks",
    "MimiEncodeSessionHooks",
    "PersonaPlexSessionAdapter",
    "PromptPreparer",
    "PromptSessionHooks",
    "encode_waveform",
    "pcm16_to_waveform",
    "waveform_to_pcm16",
]
