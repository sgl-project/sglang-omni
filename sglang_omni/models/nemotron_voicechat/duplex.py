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
    AudioPerception,
    StreamingPerception,
)
from sglang_omni.models.nemotron_voicechat.cuda_graph import capture_cuda_graph
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import ResourceUsage, SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import SessionContext, SessionHooks

INPUT_RATE = 16000
OUTPUT_RATE = 22050
FRAME_SAMPLES = 1280


class GraphPerception(StreamingPerception):
    """Replay only after causal cache shapes have reached their fixed bounds."""

    SINGLE_BUFFERS = ("sample_buffer", "preemphasis_carry")
    LIST_BUFFERS = ("sub_caches", "key_caches", "value_caches", "conv_caches")

    def __init__(self, perception: AudioPerception) -> None:
        super().__init__(perception)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.graph_input: torch.Tensor | None = None
        self.graph_output: torch.Tensor | None = None

    def state_buffers(self) -> list[torch.Tensor]:
        return [getattr(self, name) for name in self.SINGLE_BUFFERS] + [
            tensor for name in self.LIST_BUFFERS for tensor in getattr(self, name)
        ]

    @torch.inference_mode()
    def push(self, samples: torch.Tensor) -> torch.Tensor:
        if self.device.type != "cuda" or len(self.key_caches[0]) < self.max_keys:
            return super().push(samples)
        else:
            pass
        if self.graph is None:
            inputs = self.state_buffers()
            saved = [value.clone() for value in inputs]
            attrs = {name: getattr(self, name) for name in self.SINGLE_BUFFERS}
            attrs.update(
                {name: list(getattr(self, name)) for name in self.LIST_BUFFERS}
            )
            self.graph_input = samples.to(device=self.device, dtype=self.dtype).clone()

            def restore_state() -> None:
                for target, value in zip(inputs, saved):
                    target.copy_(value)
                for name, value in attrs.items():
                    setattr(
                        self, name, list(value) if name in self.LIST_BUFFERS else value
                    )

            def forward() -> torch.Tensor:
                output = super(GraphPerception, self).push(self.graph_input)
                # Keep captured cache addresses fixed across subsequent replays.
                for target, value in zip(inputs, self.state_buffers()):
                    target.copy_(value)
                return output

            self.graph, self.graph_output = capture_cuda_graph(
                forward, self.device, restore_state=restore_state
            )
            for name, value in attrs.items():
                setattr(self, name, list(value) if name in self.LIST_BUFFERS else value)
        else:
            pass

        assert self.graph_input is not None and self.graph_output is not None
        self.graph_input.copy_(samples)
        self.graph.replay()
        return self.graph_output


@dataclass(kw_only=True)
class PerceptionState:
    stream: StreamingPerception
    ended: bool = False


class PerceptionHooks(SessionHooks):
    def __init__(self, model: AudioPerception) -> None:
        self.model = model
        self.states: dict[SessionIdentity, PerceptionState] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.states[session_identity] = PerceptionState(
            stream=GraphPerception(self.model)
        )

    @torch.inference_mode()
    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        state = self.states[context.session_identity]
        if state.ended:
            raise ValueError("VoiceChat input already ended")
        else:
            pass
        if chunk.modality != "audio" or chunk.format != "pcm16":
            raise ValueError("VoiceChat requires mono 16 kHz PCM16")
        else:
            pass
        raw = chunk.payload
        if not isinstance(raw, bytes) or len(raw) % 2:
            raise ValueError("VoiceChat requires complete PCM16 samples")
        else:
            pass
        if len(raw) not in (0, FRAME_SAMPLES * 2):
            raise ValueError("VoiceChat requires 1280 samples per unit; pad at ingress")
        else:
            pass
        if not raw and not chunk.eos:
            raise ValueError("empty VoiceChat input requires EOS")
        else:
            pass
        row = None
        if raw:
            waveform = torch.from_numpy(
                np.frombuffer(raw, dtype="<i2").astype(np.float32)
            )
            row = state.stream.push(waveform / 32768.0).cpu()
        else:
            pass
        state.ended = chunk.eos
        # Match the offline pipeline: the encoder's extra flush row is not a
        # model frame. EOS only drains the codec, it does not synthesize input.
        payload.data = {"acoustic": row, "eos": chunk.eos}
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
        tensors = [stream.sample_buffer, stream.preemphasis_carry]
        for key in ("sub_caches", "key_caches", "value_caches", "conv_caches"):
            tensors.extend(getattr(stream, key))
        return ResourceUsage(bytes=sum(t.numel() * t.element_size() for t in tensors))


@dataclass(kw_only=True)
class CodecState:
    code_frames: list[torch.Tensor] = field(default_factory=list)
    frame_count: int = 0
    emitted_samples: int = 0
    ended: bool = False


class CodecHooks(SessionHooks):
    """A bounded rolling decoder window with the offline codec's tail holdback."""

    def __init__(self, decoder: RVQVAEDecoder, device: str | torch.device) -> None:
        self.decoder, self.device = decoder, device
        self.states: dict[SessionIdentity, CodecState] = {}
        self.decode_graph: torch.cuda.CUDAGraph | None = None
        self.decode_input: torch.Tensor | None = None
        self.decode_output: torch.Tensor | None = None

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        # Once the rolling window is full, its shape never changes. Reuse a
        # graph for codec kernels without changing the window or audio samples.
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
        if state.ended:
            raise ValueError("VoiceChat codec already ended")
        else:
            pass
        data = payload.data
        codes = data.get("codes")
        if codes is not None:
            state.code_frames.append(codes.reshape(-1).to(self.device))
            state.frame_count += 1
            state.code_frames = state.code_frames[-DECODE_WINDOW_FRAMES:]
        else:
            pass
        eos = bool(data.get("eos"))
        fresh = torch.zeros(0)
        if state.code_frames:
            first_frame_index = state.frame_count - len(state.code_frames)
            frame_samples = self.decoder.samples_per_frame
            available_samples = state.frame_count * frame_samples - (
                0 if eos else TAIL_HOLDBACK_SAMPLES
            )
            audio = self.decode(torch.stack(state.code_frames))
            window_start = first_frame_index * frame_samples
            start = state.emitted_samples - window_start
            end = available_samples - window_start
            fresh = audio[start:end].float().cpu()
            state.emitted_samples = available_samples
        else:
            pass
        pcm = (fresh.clamp(-1, 1).numpy() * 32767).astype("<i2").tobytes()
        state.ended = eos
        payload.data = {
            "pcm": pcm,
            "text": data.get("text", ""),
            "eos": eos,
            "text_token": data.get("text_token"),
            "function_token": data.get("function_token"),
        }
        context.emit(
            TimedChunk(
                "audio",
                chunk.t_start_ms,
                len(pcm) * 500 / OUTPUT_RATE,
                chunk.seq,
                payload.data,
                format="voicechat",
                eos=eos,
            )
        )
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
