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
from sglang_omni.models.nemotron_voicechat.conformer import StreamingPerception
from sglang_omni.models.nemotron_voicechat.cuda_graph import capture_cuda_graph
from sglang_omni.proto.session import ResourceUsage, TimedChunk
from sglang_omni.scheduling.session import SessionHooks

INPUT_RATE = 16000
OUTPUT_RATE = 22050
FRAME_SAMPLES = 1280


class GraphPerception(StreamingPerception):
    """Replay only after causal cache shapes have reached their fixed bounds."""

    _single = ("sample_buffer", "preemphasis_carry")
    _lists = ("sub_caches", "key_caches", "value_caches", "conv_caches")

    def _buffers(self) -> list[torch.Tensor]:
        return [getattr(self, name) for name in self._single] + [
            tensor for name in self._lists for tensor in getattr(self, name)
        ]

    @torch.inference_mode()
    def push(self, samples: torch.Tensor) -> torch.Tensor:
        if self.device.type != "cuda" or len(self.key_caches[0]) < self.max_keys:
            return super().push(samples)
        if not hasattr(self, "_graph"):
            inputs = self._buffers()
            saved = [value.clone() for value in inputs]
            attrs = {name: getattr(self, name) for name in self._single}
            attrs.update({name: list(getattr(self, name)) for name in self._lists})
            self._input = samples.to(device=self.device, dtype=self.dtype).clone()

            def restore_state():
                for target, value in zip(inputs, saved):
                    target.copy_(value)
                for name, value in attrs.items():
                    setattr(self, name, list(value) if name in self._lists else value)

            def forward():
                output = super(GraphPerception, self).push(self._input)
                # Keep captured cache addresses fixed across subsequent replays.
                for target, value in zip(inputs, self._buffers()):
                    target.copy_(value)
                return output

            self._graph, self._output = capture_cuda_graph(
                forward, self.device, restore_state=restore_state
            )
            for name, value in attrs.items():
                setattr(self, name, list(value) if name in self._lists else value)

        self._input.copy_(samples)
        self._graph.replay()
        return self._output


@dataclass
class PerceptionState:
    stream: StreamingPerception | None
    ended: bool = False


class PerceptionHooks(SessionHooks):
    def __init__(self, model):
        self.model = model

    def open(self, ref, request):
        return PerceptionState(GraphPerception(self.model))

    @torch.inference_mode()
    def append(self, state, chunk, payload, context):
        if state.ended:
            raise ValueError("VoiceChat input already ended")
        if chunk.modality != "audio" or chunk.format != "pcm16":
            raise ValueError("VoiceChat requires mono 16 kHz PCM16")
        raw = chunk.payload
        if not isinstance(raw, bytes) or len(raw) % 2:
            raise ValueError("VoiceChat requires complete PCM16 samples")
        if len(raw) not in (0, FRAME_SAMPLES * 2):
            raise ValueError("VoiceChat requires 1280 samples per unit; pad at ingress")
        if not raw and not chunk.eos:
            raise ValueError("empty VoiceChat input requires EOS")
        row = None
        if raw:
            wave = torch.from_numpy(np.frombuffer(raw, dtype="<i2").astype(np.float32))
            row = state.stream.push(wave / 32768.0).cpu()
        state.ended = chunk.eos
        # Match the offline pipeline: the encoder's extra flush row is not a
        # model frame. EOS only drains the codec, it does not synthesize input.
        payload.data = {"acoustic": row, "eos": chunk.eos}
        return payload

    def abort(self, state, ref):
        pass  # Output fencing must not reset causal perception history.

    def close(self, state):
        state.stream = None

    def usage(self, state):
        if state.stream is None:
            return ResourceUsage()
        stream = state.stream
        tensors = [stream.sample_buffer, stream.preemphasis_carry]
        for key in ("sub_caches", "key_caches", "value_caches", "conv_caches"):
            tensors.extend(getattr(stream, key))
        return ResourceUsage(bytes=sum(t.numel() * t.element_size() for t in tensors))


@dataclass
class CodecState:
    rows: list[torch.Tensor] = field(default_factory=list)
    frames: int = 0
    emitted: int = 0
    ended: bool = False


class CodecHooks(SessionHooks):
    """A bounded rolling decoder window with the offline codec's tail holdback."""

    def __init__(self, decoder, device):
        self.decoder, self.device = decoder, device
        self._decode_graph = None

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        # Once the rolling window is full, its shape never changes. Reuse a
        # graph for codec kernels without changing the window or audio samples.
        if codes.device.type != "cuda" or codes.shape[0] != DECODE_WINDOW_FRAMES:
            return self.decoder(codes)
        if self._decode_graph is None:
            self._decode_input = codes.clone()
            self._decode_graph, self._decode_output = capture_cuda_graph(
                lambda: self.decoder(self._decode_input), codes.device
            )
        self._decode_input.copy_(codes)
        self._decode_graph.replay()
        return self._decode_output

    def open(self, ref, request):
        return CodecState()

    @torch.inference_mode()
    def append(self, state, chunk, payload, context):
        if state.ended:
            raise ValueError("VoiceChat codec already ended")
        data = payload.data
        codes = data.get("codes")
        if codes is not None:
            state.rows.append(codes.reshape(-1).to(self.device))
            state.frames += 1
            state.rows = state.rows[-DECODE_WINDOW_FRAMES:]
        eos = bool(data.get("eos"))
        fresh = torch.zeros(0)
        if state.rows:
            first = state.frames - len(state.rows)
            frame_samples = self.decoder.samples_per_frame
            available = state.frames * frame_samples - (
                0 if eos else TAIL_HOLDBACK_SAMPLES
            )
            audio = self.decode(torch.stack(state.rows))
            window_start = first * frame_samples
            start = state.emitted - window_start
            end = available - window_start
            fresh = audio[start:end].float().cpu()
            state.emitted = available
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

    def abort(self, state, ref):
        pass  # Retain synthesis history when old output is suppressed.

    def close(self, state):
        state.rows.clear()

    def usage(self, state):
        return ResourceUsage(
            bytes=sum(t.numel() * t.element_size() for t in state.rows)
        )
