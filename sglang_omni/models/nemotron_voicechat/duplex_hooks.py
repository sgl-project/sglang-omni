# SPDX-License-Identifier: Apache-2.0
"""Session-owned streaming perception and codec for native VoiceChat."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from sglang_omni.models.nemotron_voicechat.code2wav_stream import (
    DECODE_WINDOW_FRAMES,
    TAIL_HOLDBACK_SAMPLES,
)
from sglang_omni.models.nemotron_voicechat.codec import RVQVAEDecoder
from sglang_omni.models.nemotron_voicechat.conformer import (
    SAMPLES_PER_FRAME,
    AudioPerception,
    GraphPerception,
    StreamingPerception,
)
from sglang_omni.models.nemotron_voicechat.cuda_graph import capture_cuda_graph
from sglang_omni.models.nemotron_voicechat.payload_types import OUTPUT_SAMPLE_RATE
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import ResourceUsage, SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import SessionContext, SessionHooks

PCM16_SAMPLE_BYTES = 2
PCM16_INPUT_SCALE = 32768.0
PCM16_OUTPUT_SCALE = 32767


@dataclass(kw_only=True)
class PerceptionState:
    stream: StreamingPerception
    is_ended: bool = False


class PerceptionHooks(SessionHooks):
    def __init__(self, model: AudioPerception) -> None:
        self.model = model
        self.stream = GraphPerception(model)
        self.states: dict[SessionIdentity, PerceptionState] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.stream.reset()
        self.states[session_identity] = PerceptionState(stream=self.stream)

    @torch.inference_mode()
    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        state = self.states[context.session_identity]
        if chunk.modality == "tool_response":
            # Relayed to the thinker without a perception frame; one that lands after EOS is never consumed.
            payload.data = {"tool_response": chunk.payload, "eos": False}
            return payload
        elif state.is_ended:
            raise ValueError("VoiceChat input already ended")
        else:
            pass
        if chunk.modality != "audio" or chunk.format != "pcm16":
            raise ValueError("VoiceChat requires mono 16 kHz PCM16")
        else:
            pass
        raw = chunk.payload
        if not isinstance(raw, bytes) or len(raw) % PCM16_SAMPLE_BYTES:
            raise ValueError("VoiceChat requires complete PCM16 samples")
        else:
            pass
        if len(raw) not in (0, SAMPLES_PER_FRAME * PCM16_SAMPLE_BYTES):
            raise ValueError("VoiceChat requires 1280 samples per unit; pad at ingress")
        else:
            pass
        if not raw and not chunk.eos:
            raise ValueError("empty VoiceChat input requires EOS")
        else:
            pass
        acoustic_features = None
        if raw:
            waveform = torch.from_numpy(
                np.frombuffer(raw, dtype="<i2").astype(np.float32)
            )
            acoustic_features = state.stream.push(waveform / PCM16_INPUT_SCALE).cpu()
        else:
            pass
        state.is_ended = chunk.eos
        # The encoder's flush row is not a model frame; EOS must not add one.
        payload.data = {"acoustic": acoustic_features, "eos": chunk.eos}
        return payload

    def close(self, session_identity: SessionIdentity) -> None:
        self.states.pop(session_identity, None)

    def usage(self, session_identity: SessionIdentity) -> ResourceUsage:
        state = self.states.get(session_identity)
        if state is None:
            return ResourceUsage()
        else:
            pass
        stream = state.stream
        tensors = [
            stream.sample_buffer,
            stream.preemphasis_carry,
            *stream.sub_caches,
            *stream.key_caches,
            *stream.value_caches,
            *stream.conv_caches,
        ]
        return ResourceUsage(bytes=sum(t.numel() * t.element_size() for t in tensors))


@dataclass(kw_only=True)
class CodecState:
    code_frames: list[torch.Tensor] = field(default_factory=list)
    frame_count: int = 0
    emitted_samples: int = 0
    is_ended: bool = False


class CodecHooks(SessionHooks):
    def __init__(self, decoder: RVQVAEDecoder, device: str | torch.device) -> None:
        self.decoder, self.device = decoder, device
        self.states: dict[SessionIdentity, CodecState] = {}
        self.decode_graph: torch.cuda.CUDAGraph | None = None
        self.decode_input: torch.Tensor | None = None
        self.decode_output: torch.Tensor | None = None

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.device.type != "cuda" or codes.shape[0] != DECODE_WINDOW_FRAMES:
            return self.decoder(codes)
        else:
            pass
        if self.decode_graph is None:
            self.decode_input = codes.clone()
            self.decode_graph, self.decode_output = capture_cuda_graph(
                lambda: self.decoder(self.decode_input), codes.device
            )
        else:
            pass
        assert self.decode_input is not None and self.decode_output is not None
        self.decode_input.copy_(codes)
        self.decode_graph.replay()
        return self.decode_output

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.states[session_identity] = CodecState()

    @torch.inference_mode()
    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        state = self.states[context.session_identity]
        model_output = payload.data
        if "tool_response" in model_output:
            payload.data = {}
            return payload
        elif state.is_ended:
            raise ValueError("VoiceChat codec already ended")
        else:
            pass
        codes = model_output.get("codes")
        if codes is not None:
            state.code_frames.append(codes.reshape(-1).to(self.device))
            state.frame_count += 1
            state.code_frames = state.code_frames[-DECODE_WINDOW_FRAMES:]
        else:
            pass
        eos = bool(model_output.get("eos"))
        new_audio = torch.zeros(0)
        if state.code_frames:
            first_frame_index = state.frame_count - len(state.code_frames)
            frame_samples = self.decoder.samples_per_frame
            available_samples = state.frame_count * frame_samples - (
                0 if eos else TAIL_HOLDBACK_SAMPLES
            )
            audio = self.decode(torch.stack(state.code_frames))
            window_start_sample = first_frame_index * frame_samples
            slice_start = state.emitted_samples - window_start_sample
            slice_end = available_samples - window_start_sample
            new_audio = audio[slice_start:slice_end].float().cpu()
            state.emitted_samples = available_samples
        else:
            pass
        pcm = (
            (new_audio.clamp(-1, 1).numpy() * PCM16_OUTPUT_SCALE)
            .astype("<i2")
            .tobytes()
        )
        state.is_ended = eos
        payload.data = {
            "pcm": pcm,
            "text": model_output.get("text", ""),
            "eos": eos,
            "text_token": model_output.get("text_token"),
            "function_token": model_output.get("function_token"),
        }
        context.emit(
            TimedChunk(
                "audio",
                chunk.t_start_ms,
                len(pcm) * 1000 / (PCM16_SAMPLE_BYTES * OUTPUT_SAMPLE_RATE),
                chunk.seq,
                payload.data,
                format="voicechat",
                eos=eos,
            )
        )
        tool_calls = model_output.get("tool_calls")
        if tool_calls:
            context.emit(
                TimedChunk(
                    "tool_call", chunk.t_start_ms, 0, chunk.seq, {"calls": tool_calls}
                )
            )
        else:
            pass
        return payload

    def close(self, session_identity: SessionIdentity) -> None:
        self.states.pop(session_identity, None)

    def usage(self, session_identity: SessionIdentity) -> ResourceUsage:
        state = self.states.get(session_identity)
        if state is None:
            return ResourceUsage()
        else:
            pass
        return ResourceUsage(
            bytes=sum(t.numel() * t.element_size() for t in state.code_frames)
        )
