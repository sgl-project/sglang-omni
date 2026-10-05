# SPDX-License-Identifier: Apache-2.0
"""Session-resident streaming perception for MiniCPM-o native duplex."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Literal, Protocol, TypedDict

import numpy as np
import torch
from PIL import Image
from transformers import PreTrainedTokenizerBase, WhisperFeatureExtractor

from sglang_omni.models.minicpm_o.components.audio_encoder import (
    MiniCPMOAudioEncoder,
    StreamingAudioChunk,
)
from sglang_omni.models.minicpm_o.components.whisper_encoder import AudioEncoderState
from sglang_omni.proto.session import ResourceUsage
from sglang_omni.scheduling.speaker_cache import estimate_cache_bytes

SAMPLE_RATE = 16000
UNIT_MS = 1000
FIRST_CHUNK_MS = 1035
IMAGE_TOKENS = 64
MAX_FRAME_PIXELS = 4096 * 4096
# note (Junnan Li): The constants below reproduce the checkpoint's exact streaming mel in its duplex configuration.
N_FFT = 400
HOP_LENGTH = 160
UNIT_SAMPLES = UNIT_MS * SAMPLE_RATE // 1000
UNIT_FRAMES = UNIT_SAMPLES // HOP_LENGTH
# note (Junnan Li): The checkpoint aligns the first chunk down to whole frames, so it consumes 1030 ms.
FIRST_CHUNK_SAMPLES = FIRST_CHUNK_MS * SAMPLE_RATE // 1000 // HOP_LENGTH * HOP_LENGTH
# note (Junnan Li): 20 ms of mel frames on each side of a unit feed the encoder's convolutions and are cut before attention.
CONTEXT_FRAMES = 2
# note (Junnan Li): Frames this close to a buffer end see the STFT's reflect padding, so a window starts this many frames early.
EDGE_FRAMES = -(-(N_FFT // 2) // HOP_LENGTH)
EDGE_WINDOW_SAMPLES = 2 * EDGE_FRAMES * HOP_LENGTH
SLIDE_TRIGGER_SAMPLES = 30 * SAMPLE_RATE
SLIDE_STRIDE_SAMPLES = 10 * SAMPLE_RATE
MEL_BUFFER_SAMPLES = SLIDE_TRIGGER_SAMPLES + UNIT_SAMPLES
# note (Junnan Li): Under 5 s of buffer the checkpoint clips log10 mel at a fixed floor, from 5 s on at the buffer's peak minus a fixed range.
DYNAMIC_NORM_MIN_SAMPLES = 5 * SAMPLE_RATE
LOG_FLOOR_DB = -10.0
DYNAMIC_RANGE_DB = 8.0


class ImageEncoder(Protocol):
    def __call__(
        self, *, pixel_values: list[torch.Tensor], tgt_sizes: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        pass


class ImageFeatureBatch(TypedDict):
    pixel_values: list[list[torch.Tensor]]
    tgt_sizes: list[torch.Tensor]


class ProcessorAudioFeatures(TypedDict):
    audio_features: torch.Tensor
    audio_feature_lens: list[torch.Tensor]


class StreamingAudioProcessor(Protocol):
    """The checkpoint processor surface shared by all sessions."""

    def process_image(
        self, images: list[Image.Image], *, max_slice_nums: int
    ) -> ImageFeatureBatch:
        pass

    def process_audio(
        self, audio: np.ndarray, *, sampling_rate: int
    ) -> ProcessorAudioFeatures:
        pass


class EmbeddingSpanPlan(TypedDict):
    modality: Literal["audio", "image"]
    token_start: int
    token_end: int
    embedding_start: int
    embedding_end: int


class PerceptionStepPlan(TypedDict):
    token_ids: list[int]
    input_embeds: torch.Tensor
    embedding_spans: list[EmbeddingSpanPlan]


@dataclass(kw_only=True)
class AudioFeatureBatch:
    audio_features: torch.Tensor
    audio_feature_lens: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class LogMelFilterBank:
    """Batched STFT and mel projection of the checkpoint feature extractor."""

    mel_filters: torch.Tensor
    window: torch.Tensor

    @classmethod
    def from_feature_extractor(
        cls, feature_extractor: WhisperFeatureExtractor
    ) -> LogMelFilterBank:
        assert (
            feature_extractor.n_fft,
            feature_extractor.hop_length,
            feature_extractor.dither,
        ) == (N_FFT, HOP_LENGTH, 0.0)
        return cls(
            mel_filters=torch.from_numpy(feature_extractor.mel_filters).to(
                torch.float32
            ),
            window=torch.hann_window(N_FFT),
        )

    def log_mel(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Unclipped log10 mel (windows, mels, frames) of (windows, samples) waveforms."""
        stft = torch.stft(
            waveforms,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            window=self.window,
            return_complex=True,
        )
        # note (Junnan Li): Whisper drops the last STFT frame.
        magnitudes = stft[..., :-1].abs() ** 2
        return torch.clamp(self.mel_filters.T @ magnitudes, min=1e-10).log10()


def audio_feature_batch(processor_output: ProcessorAudioFeatures) -> AudioFeatureBatch:
    return AudioFeatureBatch(
        audio_features=processor_output["audio_features"],
        audio_feature_lens=torch.cat(
            [length.reshape(-1) for length in processor_output["audio_feature_lens"]]
        ),
    )


@dataclass(kw_only=True)
class MiniCPMOPerceptionState:
    """All mutable checkpoint perception state owned by one session."""

    tokenizer: PreTrainedTokenizerBase
    processor: StreamingAudioProcessor
    audio_encoder: MiniCPMOAudioEncoder
    max_slice_nums: int
    image_encoder: ImageEncoder
    mel_filter_bank: LogMelFilterBank
    audio_buffer: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    audio_chunk_index: int = 0
    audio_encoder_state: AudioEncoderState | None = None
    prefix_token_ids: list[int] = field(default_factory=list)
    prefix_embeds: torch.Tensor | None = None
    prefix_schema: list[tuple[Literal["token", "audio"], int]] = field(
        default_factory=list
    )
    # note (Junnan Li): The checkpoint's streaming mel buffer, the peak log10 mel of each of its frames, and the next unit's first core frame.
    mel_samples: np.ndarray = field(
        default_factory=lambda: np.zeros(MEL_BUFFER_SAMPLES, dtype=np.float32)
    )
    mel_length: int = 0
    frame_maxima: np.ndarray = field(
        default_factory=lambda: np.zeros(
            MEL_BUFFER_SAMPLES // HOP_LENGTH, dtype=np.float32
        )
    )
    core_frame: int = 0
    window_start_frame: int = 0
    unit_token_id: int = field(init=False)
    image_marker_ids: dict[str, int] = field(init=False)
    is_open: bool = True

    def __post_init__(self) -> None:
        self.unit_token_id = self.tokenizer.convert_tokens_to_ids("<unit>")
        self.image_marker_ids = {
            marker: self.tokenizer.convert_tokens_to_ids(marker)
            for marker in ("<image>", "</image>", "<slice>", "</slice>")
        }

    @classmethod
    def open(
        cls,
        *,
        tokenizer: PreTrainedTokenizerBase,
        processor: StreamingAudioProcessor,
        audio_encoder: MiniCPMOAudioEncoder,
        prompt: str,
        reference_embeds: torch.Tensor,
        image_encoder: ImageEncoder,
        max_slice_nums: int,
        mel_filter_bank: LogMelFilterBank,
    ) -> MiniCPMOPerceptionState:
        state = cls(
            tokenizer=tokenizer,
            processor=processor,
            audio_encoder=audio_encoder,
            image_encoder=image_encoder,
            max_slice_nums=max_slice_nums,
            mel_filter_bank=mel_filter_bank,
        )
        prompt_ids = list(
            tokenizer.encode(
                f"<|im_start|>system\n{prompt}\n", add_special_tokens=False
            )
        )
        im_end_ids = list(tokenizer.encode("<|im_end|>", add_special_tokens=False))
        state.prefix_token_ids = list(prompt_ids)
        state.prefix_token_ids.append(
            tokenizer.convert_tokens_to_ids("<|audio_start|>")
        )
        state.prefix_embeds = reference_embeds
        count = int(state.prefix_embeds.shape[0])
        state.prefix_token_ids.extend([tokenizer.unk_token_id] * count)
        state.prefix_token_ids.append(tokenizer.convert_tokens_to_ids("<|audio_end|>"))
        state.prefix_token_ids.extend(im_end_ids)
        state.prefix_schema = [
            ("token", len(prompt_ids) + 1),
            ("audio", count),
            ("token", 1 + len(im_end_ids)),
        ]
        return state

    def close(self) -> None:
        self.is_open = False
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.mel_samples = np.zeros(0, dtype=np.float32)
        self.audio_encoder_state = None
        self.prefix_embeds = None

    def held(self) -> ResourceUsage:
        if not self.is_open:
            return ResourceUsage()
        else:
            size = (
                int(self.audio_buffer.nbytes)
                + int(self.mel_samples.nbytes)
                + (
                    self.audio_encoder_state.nbytes
                    if self.audio_encoder_state is not None
                    else 0
                )
                + estimate_cache_bytes(self.prefix_embeds)
            )
            return ResourceUsage(slots={"perception": 1}, bytes=max(size, 1))

    def prepare_audio(self, waveform: np.ndarray) -> np.ndarray:
        """Consume one unit of audio and return the samples whose STFT holds its new mel frames.

        Only the new frames are computed; the clipping peak is kept per frame.
        """
        # note (Junnan Li): The checkpoint front-pads the first chunk to 1035 ms so the encoder's CNN context is full.
        if self.audio_chunk_index == 0:
            chunk_samples = FIRST_CHUNK_SAMPLES
            padding = max(
                FIRST_CHUNK_MS * SAMPLE_RATE // 1000
                - self.audio_buffer.size
                - waveform.size,
                0,
            )
        else:
            chunk_samples = UNIT_SAMPLES
            padding = 0
        self.audio_buffer = np.concatenate(
            [np.zeros(padding, dtype=np.float32), self.audio_buffer, waveform]
        )
        assert self.audio_buffer.size >= chunk_samples, (
            self.audio_buffer.size,
            chunk_samples,
        )
        self.mel_samples[self.mel_length : self.mel_length + chunk_samples] = (
            self.audio_buffer[:chunk_samples]
        )
        self.mel_length += chunk_samples
        self.audio_buffer = self.audio_buffer[chunk_samples:].copy()
        if self.mel_length >= SLIDE_TRIGGER_SAMPLES:
            self.mel_length -= SLIDE_STRIDE_SAMPLES
            self.mel_samples[: self.mel_length] = self.mel_samples[
                SLIDE_STRIDE_SAMPLES : SLIDE_STRIDE_SAMPLES + self.mel_length
            ]
            dropped_frames = SLIDE_STRIDE_SAMPLES // HOP_LENGTH
            kept_frames = self.mel_length // HOP_LENGTH
            self.frame_maxima[:kept_frames] = self.frame_maxima[
                dropped_frames : dropped_frames + kept_frames
            ]
            self.core_frame -= dropped_frames
            # note (Junnan Li): The new first frames see reflect padding at the new buffer start; once per 10 s, so computed alone.
            edge_mel = self.mel_filter_bank.log_mel(
                torch.from_numpy(self.mel_samples[None, :EDGE_WINDOW_SAMPLES])
            )
            self.frame_maxima[:EDGE_FRAMES] = (
                edge_mel[0, :, :EDGE_FRAMES].amax(dim=0).numpy()
            )
        else:
            pass
        self.window_start_frame = max(self.core_frame - CONTEXT_FRAMES - EDGE_FRAMES, 0)
        return self.mel_samples[self.window_start_frame * HOP_LENGTH : self.mel_length]

    def mel_chunk(self, window_log_mel: torch.Tensor) -> StreamingAudioChunk:
        """Clip the new frames as the checkpoint does for the whole buffer and cut the encoder's chunk, which takes the attention history."""
        emit_start = max(self.core_frame - CONTEXT_FRAMES, 0)
        frame_count = self.mel_length // HOP_LENGTH
        self.frame_maxima[emit_start:frame_count] = (
            window_log_mel[:, emit_start - self.window_start_frame :]
            .amax(dim=0)
            .numpy()
        )
        if self.mel_length < DYNAMIC_NORM_MIN_SAMPLES:
            threshold = np.float32(LOG_FLOOR_DB)
        else:
            threshold = self.frame_maxima[:frame_count].max() - np.float32(
                DYNAMIC_RANGE_DB
            )
        emit_end = self.core_frame + UNIT_FRAMES + CONTEXT_FRAMES
        features = (
            torch.maximum(
                window_log_mel[
                    :,
                    emit_start
                    - self.window_start_frame : emit_end
                    - self.window_start_frame,
                ],
                torch.from_numpy(np.asarray(threshold)),
            )
            + 4.0
        ) / 4.0
        chunk = StreamingAudioChunk(
            audio_features=features[None],
            state=self.audio_encoder_state,
            prefix_extra_frames=self.core_frame - emit_start,
            suffix_extra_frames=CONTEXT_FRAMES,
        )
        self.audio_encoder_state = None
        self.core_frame += UNIT_FRAMES
        return chunk

    def finish_audio(self, audio_encoder_state: AudioEncoderState) -> None:
        """Keep the attention history of the unit the encoder just ran."""
        self.audio_encoder_state = audio_encoder_state
        self.audio_chunk_index += 1

    def encode_image(self, encoded_image: bytes) -> torch.Tensor:
        with Image.open(BytesIO(encoded_image)) as image:
            if image.format not in ("JPEG", "PNG"):
                raise ValueError("unit image must be JPEG or PNG")
            elif image.width * image.height > MAX_FRAME_PIXELS:
                raise ValueError("unit image exceeds pixel limit")
            else:
                frame = image.convert("RGB")
        processed = self.processor.process_image(
            [frame], max_slice_nums=self.max_slice_nums
        )
        image_embeds = self.image_encoder(
            pixel_values=processed["pixel_values"][0],
            tgt_sizes=processed["tgt_sizes"][0],
        )["image_embeds"]
        assert image_embeds.ndim == 2 and image_embeds.shape[0] % IMAGE_TOKENS == 0
        return image_embeds

    def build_step_plan(
        self, audio_embeds: torch.Tensor, image_embeds: tuple[torch.Tensor, ...] = ()
    ) -> PerceptionStepPlan:
        token_ids: list[int] = []
        embedding_blocks: list[torch.Tensor] = []
        spans: list[EmbeddingSpanPlan] = []

        def add_embeds(
            embeddings: torch.Tensor, modality: Literal["audio", "image"] = "audio"
        ) -> None:
            token_start = len(token_ids)
            row_count = int(embeddings.shape[0])
            embedding_start = sum(int(block.shape[0]) for block in embedding_blocks)
            token_ids.extend([self.tokenizer.unk_token_id] * row_count)
            embedding_blocks.append(embeddings)
            spans.append(
                EmbeddingSpanPlan(
                    modality=modality,
                    token_start=token_start,
                    token_end=token_start + row_count,
                    embedding_start=embedding_start,
                    embedding_end=embedding_start + row_count,
                )
            )

        if self.audio_chunk_index == 1 and self.prefix_token_ids:
            token_cursor = 0
            embedding_cursor = 0
            for segment_kind, segment_length in self.prefix_schema:
                if segment_kind == "token":
                    token_ids.extend(
                        self.prefix_token_ids[
                            token_cursor : token_cursor + segment_length
                        ]
                    )
                else:
                    assert self.prefix_embeds is not None
                    add_embeds(
                        self.prefix_embeds[
                            embedding_cursor : embedding_cursor + segment_length
                        ]
                    )
                    embedding_cursor += segment_length
                token_cursor += segment_length
        else:
            pass

        token_ids.append(self.unit_token_id)
        for frame_embeds in image_embeds:
            assert (
                frame_embeds.ndim == 2
                and frame_embeds.shape[1] == audio_embeds.shape[1]
            )
            assert (
                frame_embeds.shape[0] > 0 and frame_embeds.shape[0] % IMAGE_TOKENS == 0
            )
            for slice_index, slice_embeds in enumerate(
                frame_embeds.split(IMAGE_TOKENS)
            ):
                marker = "image" if slice_index == 0 else "slice"
                token_ids.append(self.image_marker_ids[f"<{marker}>"])
                add_embeds(slice_embeds, "image")
                token_ids.append(self.image_marker_ids[f"</{marker}>"])
        add_embeds(audio_embeds)
        return PerceptionStepPlan(
            token_ids=token_ids,
            input_embeds=torch.cat(embedding_blocks, dim=0),
            embedding_spans=spans,
        )
