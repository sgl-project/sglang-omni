# SPDX-License-Identifier: Apache-2.0
"""Session-resident streaming perception for MiniCPM-o native duplex."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, TypedDict

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.minicpm_o.components.audio_encoder import (
    AudioEncoderState,
    MiniCPMOAudioEncoder,
)
from sglang_omni.preprocessing.audio import load_audio_path
from sglang_omni.proto.session import ResourceUsage
from sglang_omni.scheduling.speaker_cache import estimate_cache_bytes

SAMPLE_RATE = 16000
UNIT_MS = 1000
FIRST_CHUNK_MS = 1035
UNIT_DECODE_BUDGET = 20


class StreamingMelProcessor(Protocol):
    def get_config(self) -> Mapping[str, object]:
        pass


class StreamingAudioProcessor(Protocol):
    """The checkpoint processor's streaming surface used by one session."""

    _streaming_mel_processor: StreamingMelProcessor

    def set_streaming_mode(
        self,
        *,
        mode: str,
        chunk_ms: int,
        first_chunk_ms: int,
        cnn_redundancy_ms: int,
        enable_sliding_window: bool,
        slide_trigger_seconds: float,
        slide_stride_seconds: float,
    ) -> None:
        pass

    def get_streaming_chunk_size(self) -> int:

        pass

    def process_audio(
        self, audio: np.ndarray, *, sampling_rate: int
    ) -> Mapping[str, object]:
        pass

    def process_audio_streaming(
        self, audio: np.ndarray, *, reset: bool, return_batch_feature: bool
    ) -> Mapping[str, object]:
        pass


class ProcessorFactory(Protocol):
    def __call__(self) -> StreamingAudioProcessor:
        pass


class EmbeddingSpanPlan(TypedDict):
    modality: str
    token_start: int
    token_end: int
    embed_start: int
    embed_end: int


class PerceptionStepPlan(TypedDict):
    token_ids: list[int]
    input_embeds: torch.Tensor
    embedding_spans: list[EmbeddingSpanPlan]
    prefill_schema: list[tuple[str, int]]
    decode_budget: int


@dataclass(kw_only=True)
class AudioFeatureBatch:
    audio_features: torch.Tensor
    audio_feature_lens: torch.Tensor


def audio_feature_batch(processor_output: Mapping[str, object]) -> AudioFeatureBatch:
    features = processor_output["audio_features"]
    lengths = processor_output["audio_feature_lens"]
    if not isinstance(features, torch.Tensor):
        raise TypeError("processor audio_features must be a tensor")
    else:
        pass
    if isinstance(lengths, list | tuple):
        lengths = torch.cat([torch.as_tensor(length).reshape(-1) for length in lengths])
    elif not isinstance(lengths, torch.Tensor):
        raise TypeError("processor audio_feature_lens must be a tensor or list")
    else:
        pass
    return AudioFeatureBatch(
        audio_features=features, audio_feature_lens=lengths.reshape(-1)
    )


@dataclass(kw_only=True)
class MiniCPMOPerceptionState:
    """All mutable checkpoint perception state owned by one session."""

    tokenizer: PreTrainedTokenizerBase
    processor: StreamingAudioProcessor
    audio_encoder: MiniCPMOAudioEncoder
    audio_buffer: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    audio_chunk_idx: int = 0
    audio_encoder_state: AudioEncoderState | None = None
    prefix_token_ids: list[int] = field(default_factory=list)
    prefix_embeds: torch.Tensor | None = None
    prefix_schema: list[tuple[str, int]] = field(default_factory=list)
    is_open: bool = True

    @classmethod
    def open(
        cls,
        *,
        tokenizer: PreTrainedTokenizerBase,
        processor: StreamingAudioProcessor,
        audio_encoder: MiniCPMOAudioEncoder,
        prompt: str,
        reference_audio: str | None,
    ) -> MiniCPMOPerceptionState:
        processor.set_streaming_mode(
            mode="exact",
            chunk_ms=UNIT_MS,
            first_chunk_ms=FIRST_CHUNK_MS,
            cnn_redundancy_ms=20,
            enable_sliding_window=True,
            slide_trigger_seconds=30.0,
            slide_stride_seconds=10.0,
        )
        state = cls(
            tokenizer=tokenizer, processor=processor, audio_encoder=audio_encoder
        )
        prompt_ids = list(
            tokenizer.encode(
                f"<|im_start|>system\n{prompt}\n", add_special_tokens=False
            )
        )
        im_end_ids = list(tokenizer.encode("<|im_end|>", add_special_tokens=False))
        state.prefix_token_ids = list(prompt_ids)
        if reference_audio is None:
            state.prefix_token_ids.extend(im_end_ids)
            state.prefix_schema = [("tok", len(prompt_ids) + len(im_end_ids))]
        else:
            state.prefix_token_ids.append(
                tokenizer.convert_tokens_to_ids("<|audio_start|>")
            )
            waveform = np.asarray(
                load_audio_path(reference_audio, target_sr=SAMPLE_RATE),
                dtype=np.float32,
            ).reshape(-1)
            batch = audio_feature_batch(
                processor.process_audio(waveform, sampling_rate=SAMPLE_RATE)
            )
            state.prefix_embeds = audio_encoder(
                audio_features=batch.audio_features,
                audio_feature_lens=batch.audio_feature_lens,
            )["audio_embeds"]
            count = int(state.prefix_embeds.shape[0])
            state.prefix_token_ids.extend([tokenizer.unk_token_id] * count)
            state.prefix_token_ids.append(
                tokenizer.convert_tokens_to_ids("<|audio_end|>")
            )
            state.prefix_token_ids.extend(im_end_ids)
            state.prefix_schema = [
                ("tok", len(prompt_ids) + 1),
                ("audio", count),
                ("tok", 1 + len(im_end_ids)),
            ]
        return state

    def close(self) -> None:
        self.is_open = False
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.audio_encoder_state = None
        self.prefix_embeds = None

    def held(self) -> ResourceUsage:
        if not self.is_open:
            return ResourceUsage()
        else:
            size = (
                int(self.audio_buffer.nbytes)
                + (
                    self.audio_encoder_state.nbytes
                    if self.audio_encoder_state is not None
                    else 0
                )
                + estimate_cache_bytes(self.prefix_embeds)
            )
            # note (Junnan Li): Count the live processor even when every buffer is empty.
            return ResourceUsage(slots={"perception": 1}, bytes=max(size, 1))

    def encode_audio(self, pcm: np.ndarray) -> torch.Tensor:
        need_samples = self.processor.get_streaming_chunk_size()
        # note (Junnan Li): The checkpoint front-pads the first chunk to 1035 ms so the encoder's CNN context is full.
        if self.audio_chunk_idx == 0:
            first_chunk_samples = FIRST_CHUNK_MS * SAMPLE_RATE // 1000
            padding = max(first_chunk_samples - self.audio_buffer.size - pcm.size, 0)
        else:
            padding = 0
        self.audio_buffer = np.concatenate(
            [np.zeros(padding, dtype=np.float32), self.audio_buffer, pcm]
        )
        assert self.audio_buffer.size >= need_samples, (
            self.audio_buffer.size,
            need_samples,
        )
        batch = audio_feature_batch(
            self.processor.process_audio_streaming(
                self.audio_buffer[:need_samples].copy(),
                reset=False,
                return_batch_feature=True,
            )
        )
        audio_embeds, self.audio_encoder_state = self.audio_encoder.forward_streaming(
            audio_features=batch.audio_features,
            audio_feature_lens=batch.audio_feature_lens,
            state=self.audio_encoder_state,
            prefix_extra_frames=0 if self.audio_chunk_idx == 0 else 2,
            suffix_extra_frames=2,
        )
        if self.audio_chunk_idx == 0:
            # note (Junnan Li): The checkpoint processor owns this private attribute.
            config = (
                self.processor._streaming_mel_processor.get_config()
            )  # noqa: leading-underscore
            consumed_ms = int(config.get("effective_first_chunk_ms", FIRST_CHUNK_MS))
            consumed_samples = consumed_ms * SAMPLE_RATE // 1000
        else:
            consumed_samples = need_samples
        self.audio_buffer = self.audio_buffer[consumed_samples:].copy()
        self.audio_chunk_idx += 1
        return audio_embeds

    def build_step_plan(self, audio_embeds: torch.Tensor) -> PerceptionStepPlan:
        token_ids: list[int] = []
        embed_blocks: list[torch.Tensor] = []
        spans: list[EmbeddingSpanPlan] = []

        def add_embeds(values: torch.Tensor) -> None:
            start = len(token_ids)
            count = int(values.shape[0])
            embed_start = sum(int(block.shape[0]) for block in embed_blocks)
            token_ids.extend([self.tokenizer.unk_token_id] * count)
            embed_blocks.append(values)
            spans.append(
                EmbeddingSpanPlan(
                    modality="audio",
                    token_start=start,
                    token_end=start + count,
                    embed_start=embed_start,
                    embed_end=embed_start + count,
                )
            )

        # note (Junnan Li): The reference implementation prefills the system prefix apart from the first unit; it enters this plan while prefill_schema reports only the unit, so oracle comparison keeps that boundary.
        if self.audio_chunk_idx == 1 and self.prefix_token_ids:
            cursor = 0
            embed_cursor = 0
            for kind, count in self.prefix_schema:
                if kind == "tok":
                    token_ids.extend(self.prefix_token_ids[cursor : cursor + count])
                else:
                    assert self.prefix_embeds is not None
                    add_embeds(self.prefix_embeds[embed_cursor : embed_cursor + count])
                    embed_cursor += count
                cursor += count
        else:
            pass

        token_ids.append(self.tokenizer.convert_tokens_to_ids("<unit>"))
        add_embeds(audio_embeds)
        return PerceptionStepPlan(
            token_ids=token_ids,
            input_embeds=torch.cat(embed_blocks, dim=0),
            embedding_spans=spans,
            prefill_schema=[("tok", 1), ("audio", int(audio_embeds.shape[0]))],
            decode_budget=UNIT_DECODE_BUDGET,
        )
