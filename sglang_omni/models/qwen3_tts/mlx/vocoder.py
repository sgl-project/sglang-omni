# SPDX-License-Identifier: Apache-2.0
"""Request-local audio decoding behind the shared MLX vocoder stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

from sglang_omni.models.qwen3_tts.mlx.decoder import (
    Qwen3TTSMlxSpeechDecoder,
    load_qwen3_tts_mlx_decoder,
)
from sglang_omni.models.qwen3_tts.mlx.decoder_stream import Qwen3TTSMlxDecoderStream
from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload


@dataclass(kw_only=True)
class MlxAudioState:
    is_streaming: bool
    decoder_stream: Qwen3TTSMlxDecoderStream
    pending_waveform: mx.array = field(default_factory=lambda: mx.zeros((0,)))
    code_chunks: list[np.ndarray] = field(default_factory=list)
    sample_count: int = 0


class Qwen3TTSMlxVocoder(StreamingSimpleScheduler):
    """Decode streamed codec chunks, buffering audio for non-streaming clients."""

    def __init__(self, model_dir: Path) -> None:
        super().__init__(None)
        self.decoder: Qwen3TTSMlxSpeechDecoder = load_qwen3_tts_mlx_decoder(model_dir)
        self.audio_states: dict[str, MlxAudioState] = {}

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        # note (Codex): Codec input streams for both buffered and streaming HTTP requests.
        return True

    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        if request_id in self.completed_non_streaming_request_ids:
            return []
        else:
            pass
        if not isinstance(item.data, np.ndarray) or not isinstance(item.metadata, dict):
            raise TypeError("Qwen3-TTS MLX vocoder requires codec arrays and metadata")
        else:
            codes = item.data
        is_streaming = item.metadata.get("is_streaming")
        if item.metadata.get("modality") != "audio_codes" or not isinstance(
            is_streaming, bool
        ):
            raise ValueError("Qwen3-TTS MLX vocoder requires the audio_codes contract")
        elif codes.ndim != 3 or codes.shape[0] != 1:
            raise ValueError(
                "Qwen3-TTS MLX vocoder requires one request per codec chunk"
            )
        else:
            pass
        if request_id not in self.audio_states:
            self.audio_states[request_id] = MlxAudioState(
                is_streaming=is_streaming,
                decoder_stream=Qwen3TTSMlxDecoderStream(self.decoder),
            )
        else:
            pass
        audio_state = self.audio_states[request_id]
        if audio_state.is_streaming != is_streaming:
            raise ValueError("Qwen3-TTS MLX streaming mode changed within a request")
        elif not is_streaming:
            audio_state.code_chunks.append(codes)
            return []
        else:
            waveform, lengths = audio_state.decoder_stream.decode(mx.array(codes))
        audio_state.pending_waveform = mx.concatenate(
            [audio_state.pending_waveform, waveform[0]]
        )
        valid_samples = int(lengths[0].item())
        audio = np.asarray(
            audio_state.pending_waveform[:valid_samples], dtype=np.float32
        )
        audio_state.pending_waveform = audio_state.pending_waveform[valid_samples:]
        audio_state.sample_count += audio.size
        if audio.size == 0:
            return []
        else:
            return [
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data=audio_waveform_payload(
                        audio,
                        sample_rate=self.decoder.output_sample_rate,
                        modality="audio",
                        source_hint="Qwen3-TTS MLX",
                    ),
                    metadata={"modality": "audio"},
                )
            ]

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        payload: StagePayload = self.stream_payloads[request_id]
        state = Qwen3TTSState.from_dict(payload.data)
        audio_state = self.audio_states[request_id]
        if audio_state.is_streaming:
            payload.data = {
                "sample_rate": self.decoder.output_sample_rate,
                "modality": "audio",
            }
        else:
            codes = mx.array(np.concatenate(audio_state.code_chunks, axis=1))
            waveform, lengths = self.decoder.decode(codes)
            mx.eval(waveform, lengths)
            audio = np.asarray(waveform[0, : int(lengths[0].item())], dtype=np.float32)
            audio_state.sample_count = audio.size
            payload.data = audio_waveform_payload(
                audio,
                sample_rate=self.decoder.output_sample_rate,
                modality="audio",
                source_hint="Qwen3-TTS MLX",
            )
        if audio_state.sample_count == 0:
            raise RuntimeError("Qwen3-TTS MLX returned an empty or invalid waveform")
        else:
            pass
        usage = build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        else:
            pass
        self.record_completed_non_streaming_request_id(request_id)
        return [OutgoingMessage(request_id=request_id, type="result", data=payload)]

    def clear_stream_state(self, request_id: str) -> None:
        self.audio_states.pop(request_id, None)

    def start(self) -> None:
        try:
            super().start()
        finally:
            with self.state_lock:
                for request_id in list(self.audio_states):
                    self.clear_request_state(request_id)
