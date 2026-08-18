# SPDX-License-Identifier: Apache-2.0
"""Non-streaming vocoder implementation for MOSS-TTS Delay."""

from __future__ import annotations

import logging
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import accumulate
from typing import Any, cast

import torch
from torch.nn.utils.rnn import pad_sequence

from sglang_omni.models.moss_tts.audio_tokenizer import (
    MossAudioTokenizerVocoderDecoder,
    MossTTSAudioTokenizer,
)
from sglang_omni.models.moss_tts.delay_pattern import split_moss_audio_segments
from sglang_omni.models.moss_tts.payload_types import (
    MossTTSState,
    load_moss_tts_state,
    resolve_moss_audio_pad_code,
    store_moss_tts_state,
)
from sglang_omni.models.moss_tts.vocoder_quantizer import (
    MossAudioTokenizerQuantizerDecoder,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)


def _codec_device(codec: Any, fallback: str) -> torch.device:
    try:
        return next(codec.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device(fallback)


def _module_dtype(module: Any) -> torch.dtype | None:
    try:
        return next(
            parameter.dtype
            for parameter in module.parameters()
            if parameter.is_floating_point()
        )
    except (AttributeError, StopIteration):
        return None


@contextmanager
def _autocast_if_supported(
    device: torch.device,
    dtype: torch.dtype | None,
) -> Iterator[None]:
    enabled = (device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)) or (
        device.type == "cpu" and dtype is torch.bfloat16
    )
    if enabled:
        with torch.autocast(
            device_type=device.type,
            dtype=cast(torch.dtype, dtype),
        ):
            yield
    else:
        yield


def _join_waveforms(waveforms: list[torch.Tensor]) -> torch.Tensor:
    if not waveforms:
        raise ValueError("waveforms must not be empty")
    return waveforms[0] if len(waveforms) == 1 else torch.cat(waveforms, dim=0)


def _copy_valid_waveforms_to_cpu(
    audio: torch.Tensor,
    output_lengths: list[int],
) -> list[torch.Tensor]:
    if audio.ndim != 3 or int(audio.shape[1]) != 1:
        raise ValueError(
            "MOSS-TTS Delay decoder audio must have shape [B, 1, T], got "
            f"{tuple(audio.shape)}"
        )
    if len(output_lengths) != int(audio.shape[0]):
        raise ValueError("output_lengths must match the decoder batch size")
    max_length = int(audio.shape[2])
    if any(length < 0 or length > max_length for length in output_lengths):
        raise ValueError("output_lengths must be within the decoded waveform")

    valid_waveforms = [
        audio[index, 0, :length] for index, length in enumerate(output_lengths)
    ]
    flat_audio = _join_waveforms(valid_waveforms)
    if flat_audio.device.type == "cuda":
        try:
            flat_cpu = torch.empty(
                flat_audio.shape,
                device="cpu",
                dtype=torch.float32,
                pin_memory=True,
            )
        except RuntimeError:
            flat_cpu = flat_audio.detach().to(device="cpu", dtype=torch.float32)
        else:
            flat_cpu.copy_(flat_audio.detach(), non_blocking=True)
            torch.cuda.current_stream(flat_audio.device).synchronize()
    else:
        flat_cpu = flat_audio.detach().to(device="cpu", dtype=torch.float32)
    offsets = [0, *accumulate(output_lengths)]
    return [
        flat_cpu[start:end].contiguous()
        for start, end in zip(offsets[:-1], offsets[1:], strict=True)
    ]


class MossTTSVocoder(BatchVocoderBase):
    def __init__(
        self,
        processor: Any,
        audio_tokenizer: MossTTSAudioTokenizer,
        device: str,
        *,
        compute_dtype: torch.dtype | None = None,
        max_segment_batch_size: int = 8,
    ) -> None:
        self._processor = processor
        self._audio_tokenizer = audio_tokenizer
        self._device = device
        self._compute_dtype = compute_dtype
        self._max_segment_batch_size = max(int(max_segment_batch_size), 1)
        self._codec = getattr(audio_tokenizer, "model", None)
        self._quantizer = getattr(self._codec, "quantizer", None)
        self._quantizer_decoder = None
        self._nonstream_decoder = None
        if (
            self._compute_dtype is not None
            and self._codec is not None
            and callable(getattr(self._quantizer, "decode_codes", None))
            and hasattr(self._codec, "decoder")
        ):
            codec_device = _codec_device(self._codec, self._device)
            try:
                nonstream_decoder = MossAudioTokenizerVocoderDecoder(
                    self._codec.decoder
                )
                supports_packed_flash = nonstream_decoder.supports_packed_flash(
                    codec_device,
                    self._compute_dtype,
                )
            except (AttributeError, AssertionError, TypeError, ValueError):
                logger.exception(
                    "MOSS-TTS Delay codec is incompatible with the batched "
                    "packed decoder; falling back to standalone codec decode"
                )
            else:
                if supports_packed_flash:
                    self._quantizer.to(dtype=torch.float32)
                    try:
                        self._quantizer_decoder = MossAudioTokenizerQuantizerDecoder(
                            self._quantizer
                        )
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        logger.exception(
                            "MOSS-TTS Delay quantizer is incompatible with the "
                            "cached decoder; using source quantizer decode"
                        )
                    self._nonstream_decoder = nonstream_decoder
                    logger.info(
                        "MOSS-TTS Delay vocoder enabled batched packed decoder "
                        "stages=%d compute_dtype=%s",
                        len(self._nonstream_decoder),
                        self._compute_dtype,
                    )
                else:
                    logger.info(
                        "MOSS-TTS Delay packed decoder is unavailable for "
                        "device=%s compute_dtype=%s; using standalone codec decode",
                        codec_device,
                        self._compute_dtype,
                    )

    def prepare_item(self, payload: StagePayload) -> tuple[MossTTSState, torch.Tensor]:
        state = load_moss_tts_state(payload)
        if state.delayed_audio_codes is None:
            raise RuntimeError("MOSS-TTS vocoder requires delayed_audio_codes")
        delayed_codes = torch.as_tensor(state.delayed_audio_codes, dtype=torch.long)
        if delayed_codes.numel() == 0:
            raise RuntimeError("MOSS-TTS generated no delayed audio codes")
        return state, delayed_codes

    def _decode_audio(
        self,
        state: MossTTSState,
        delayed_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        delayed_codes = delayed_codes.to(device=self._device, dtype=torch.long)
        audio_pad_code = resolve_moss_audio_pad_code(
            getattr(self._processor, "model_config", None)
        )
        segments = split_moss_audio_segments(
            delayed_codes,
            audio_pad_code=audio_pad_code,
            assistant_start_length=int(state.assistant_start_length),
        )
        codec_decoder = getattr(self._codec, "decoder", self._codec)
        codec_dtype = _module_dtype(codec_decoder) or _module_dtype(self._codec)
        codec_device = _codec_device(self._codec, self._device)
        decoded = []
        with _autocast_if_supported(codec_device, codec_dtype):
            for segment in segments:
                decoded.extend(self._audio_tokenizer.decode_codes([segment]))
        if not decoded:
            raise RuntimeError("MOSS-TTS vocoder decoded no audio segments")
        waveforms = [
            torch.as_tensor(wav).detach().reshape(-1).to("cpu") for wav in decoded
        ]
        waveform = _join_waveforms(waveforms)
        return waveform, self._resolve_sample_rate(state)

    def _resolve_sample_rate(self, state: MossTTSState) -> int:
        return int(
            getattr(self._audio_tokenizer, "sample_rate", 0)
            or getattr(getattr(self._codec, "config", None), "sampling_rate", 0)
            or getattr(
                getattr(self._processor, "model_config", None), "sampling_rate", 0
            )
            or state.sample_rate
            or 24000
        )

    def _decode_segments_batch(
        self,
        segments: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        codec = self._codec
        if codec is None:
            raise RuntimeError("batched MOSS-TTS Delay codec path is unavailable")
        packed_decoder = self._nonstream_decoder
        if packed_decoder is None:
            raise RuntimeError("packed MOSS-TTS Delay codec path is unavailable")
        if not segments:
            return []

        if any(segment.ndim != 2 for segment in segments):
            raise ValueError("MOSS-TTS Delay codec segments must be 2D")
        device = _codec_device(codec, self._device)
        n_vq = int(segments[0].shape[1])
        if n_vq <= 0 or any(int(segment.shape[1]) != n_vq for segment in segments):
            raise ValueError("MOSS-TTS Delay codec segments must share one n_vq")

        host_segments = [
            segment.to(device="cpu", dtype=torch.long) for segment in segments
        ]
        input_lengths_cpu = [int(segment.shape[0]) for segment in host_segments]
        padded_codes = pad_sequence(
            host_segments,
            batch_first=True,
            padding_value=0,
        )
        audio_codes = (
            padded_codes.permute(2, 0, 1)
            .contiguous()
            .to(device=device, non_blocking=True)
        )
        input_lengths = torch.tensor(
            input_lengths_cpu,
            device=device,
            dtype=torch.int32,
        )
        output_lengths_cpu = packed_decoder.output_lengths(input_lengths_cpu)

        quantizer = self._quantizer
        if quantizer is None or not callable(getattr(quantizer, "decode_codes", None)):
            raise RuntimeError(
                "MOSS-TTS Delay audio tokenizer has no supported "
                "quantizer.decode_codes"
            )
        with torch.inference_mode():
            # Keep codebook math in FP32. Only decoder stages run under the
            # configured compute autocast, independent of the attention backend.
            with torch.autocast(device_type=device.type, enabled=False):
                quantizer_decoder = self._quantizer_decoder
                decoder_hidden_states = (
                    quantizer.decode_codes(audio_codes)
                    if quantizer_decoder is None
                    else quantizer_decoder.decode_codes(audio_codes)
                ).float()
            with _autocast_if_supported(device, self._compute_dtype):
                audio, audio_lengths = packed_decoder(
                    decoder_hidden_states,
                    input_lengths,
                    input_lengths_cpu=input_lengths_cpu,
                )
        if audio is None or audio_lengths is None:
            raise RuntimeError(
                "MOSS-TTS Delay audio tokenizer returned empty audio/audio_lengths"
            )
        if len(output_lengths_cpu) != int(audio.shape[0]):
            raise RuntimeError(
                "MOSS-TTS Delay decoder returned an unexpected output batch size"
            )
        return _copy_valid_waveforms_to_cpu(audio, output_lengths_cpu)

    def _decode_segment_batches(
        self,
        segments: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        decoded: list[torch.Tensor] = []
        for start in range(0, len(segments), self._max_segment_batch_size):
            decoded.extend(
                self._decode_segments_batch(
                    segments[start : start + self._max_segment_batch_size],
                )
            )
        return decoded

    async def decode_batch(
        self, items: list[tuple[MossTTSState, torch.Tensor]]
    ) -> list[tuple[torch.Tensor, int]]:
        if self._nonstream_decoder is None:
            return [self._decode_audio(state, codes) for state, codes in items]

        audio_pad_code = resolve_moss_audio_pad_code(
            getattr(self._processor, "model_config", None)
        )
        request_segments: list[list[int]] = [[] for _ in items]
        flat_segments: list[torch.Tensor] = []
        for request_index, (state, delayed_codes) in enumerate(items):
            segments = split_moss_audio_segments(
                delayed_codes.to(dtype=torch.long),
                audio_pad_code=audio_pad_code,
                assistant_start_length=int(state.assistant_start_length),
            )
            for segment in segments:
                request_segments[request_index].append(len(flat_segments))
                flat_segments.append(segment)

        if any(not indices for indices in request_segments):
            raise RuntimeError("MOSS-TTS vocoder decoded no audio segments")

        failure_traceback = None
        try:
            decoded_segments = self._decode_segment_batches(flat_segments)
        except Exception:
            # Materialize the traceback as text, then leave the exception scope
            # before retrying. The exception traceback can otherwise retain the
            # failed batch's CUDA tensors throughout the fallback.
            failure_traceback = traceback.format_exc()
        if failure_traceback is not None:
            self._nonstream_decoder = None
            self._quantizer_decoder = None
            logger.error(
                "MOSS-TTS Delay packed codec decode failed; disabling the "
                "packed path and falling back to standalone codec decode:\n%s",
                failure_traceback.rstrip(),
            )
            return [self._decode_audio(state, codes) for state, codes in items]

        return [
            (
                _join_waveforms([decoded_segments[index] for index in indices]),
                self._resolve_sample_rate(items[request_index][0]),
            )
            for request_index, indices in enumerate(request_segments)
        ]

    def store_result(
        self,
        payload: StagePayload,
        state: MossTTSState,
        wav: torch.Tensor,
        sample_rate: int,
    ) -> StagePayload:
        audio_payload = audio_waveform_payload(wav, source_hint="MOSS-TTS")
        state.delayed_audio_codes = None
        state.sample_rate = int(sample_rate)
        payload = store_moss_tts_state(payload, state)
        payload.data.update(audio_payload)
        payload.data["sample_rate"] = state.sample_rate
        payload.data["modality"] = "audio"
        usage = build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload


__all__ = ["MossTTSVocoder"]
