# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the MOSS-TTS Delay pipeline."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer

from sglang_omni.models.moss_tts.audio_tokenizer import (
    DEFAULT_MOSS_TTS_AUDIO_TOKENIZER,
    MossTTSAudioTokenizer,
    load_moss_tts_audio_tokenizer,
)
from sglang_omni.models.moss_tts.delay_pattern import split_moss_audio_segments
from sglang_omni.models.moss_tts.engine_builder import MossTtsEngineBuilder
from sglang_omni.models.moss_tts.hf_loading import (
    load_moss_processor_class,
    moss_transformers_processor_compat,
)
from sglang_omni.models.moss_tts.payload_types import (
    MossTTSState,
    moss_tts_special_token_defaults,
    resolve_moss_audio_pad_code,
)
from sglang_omni.models.moss_tts.request_builders import (
    cleanup_prepared_moss_tts_request,
    preprocess_moss_tts_payload,
    set_moss_tts_preprocessing_context,
)
from sglang_omni.models.moss_tts.streaming_vocoder import MossStreamingVocoderScheduler
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.pipeline_state import load_state as _load_pipeline_state
from sglang_omni.scheduling.pipeline_state import store_state as _store_pipeline_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from sglang_omni.utils.audio import load_audio
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

_MOSS_TTS_INSTALL_HINT = (
    "MOSS-TTS support requires the upstream custom Transformers code. "
    "Launch with trust_remote_code=True and make sure the checkpoint can load "
    "OpenMOSS-Team/MOSS-Audio-Tokenizer."
)


def load_state(payload: StagePayload) -> MossTTSState:
    return _load_pipeline_state(payload, MossTTSState)


def store_state(payload: StagePayload, state: MossTTSState) -> StagePayload:
    return _store_pipeline_state(payload, state)


def _normalize_moss_processor_config(processor: Any) -> None:
    model_config = getattr(processor, "model_config", None)
    if model_config is None:
        return
    audio_vocab_size = int(getattr(model_config, "audio_vocab_size", 1024) or 1024)
    for attr, default in moss_tts_special_token_defaults(audio_vocab_size):
        if getattr(model_config, attr, None) is None:
            setattr(model_config, attr, default)


def _audio_tokenizer_model_path_from_processor_dict(
    processor_dict: dict[str, Any],
) -> str | None:
    model_path = processor_dict.get("audio_tokenizer_name_or_path")
    audio_tokenizer_dict = processor_dict.get("audio_tokenizer")
    if isinstance(audio_tokenizer_dict, dict):
        model_path = (
            audio_tokenizer_dict.get("audio_tokenizer_name_or_path") or model_path
        )
    return str(model_path) if model_path else None


def _load_moss_processor(
    model_path: str,
) -> Any:
    logger.info(f"Loading MOSS-TTS processor from {model_path} without codec")
    try:
        with moss_transformers_processor_compat():
            processor_cls = load_moss_processor_class(model_path)
            processor_dict, _ = processor_cls.get_processor_dict(model_path)
            audio_tokenizer_model_path = (
                _audio_tokenizer_model_path_from_processor_dict(processor_dict)
            )
            model_config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            if audio_tokenizer_model_path:
                # processor_cls.from_pretrained normally resolves this metadata
                # before loading the codec. Preserve the same selection while
                # constructing the processor without a codec instance.
                model_config.audio_tokenizer_name_or_path = audio_tokenizer_model_path
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            processor = processor_cls(
                tokenizer=tokenizer,
                audio_tokenizer=None,
                model_config=model_config,
            )
    except Exception as exc:
        raise RuntimeError(_MOSS_TTS_INSTALL_HINT) from exc

    _normalize_moss_processor_config(processor)
    return processor


def _resolve_audio_tokenizer_model_path(
    processor: Any,
    codec_model_path: str | None,
) -> str:
    return str(
        codec_model_path
        or getattr(processor.model_config, "audio_tokenizer_name_or_path", None)
        or DEFAULT_MOSS_TTS_AUDIO_TOKENIZER
    )


def _resolve_codec_device(device: str | None, gpu_id: int | None) -> str:
    if device:
        return device
    if gpu_id is not None:
        return f"cuda:{int(gpu_id)}"
    return "cuda:0"


class _MossTTSReferenceEncoder:
    """Reference-encode boundary backed by a standalone audio tokenizer."""

    def __init__(self, audio_tokenizer: MossTTSAudioTokenizer, *, n_vq: int) -> None:
        self.audio_tokenizer = audio_tokenizer
        self.n_vq = int(n_vq)

    def encode(self, source: str | os.PathLike[str]) -> torch.Tensor:
        waveform = load_audio(
            os.fsdecode(source),
            source_name="MOSS-TTS reference",
            target_sample_rate=self.audio_tokenizer.sample_rate,
            mono=True,
        )
        return self.audio_tokenizer.encode_waveforms(
            [(torch.from_numpy(waveform), self.audio_tokenizer.sample_rate)],
            num_quantizers=self.n_vq,
        )[0]


def create_preprocessing_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "float32",
    codec_model_path: str | None = None,
    max_concurrency: int = 8,
) -> SimpleScheduler:
    device = _resolve_codec_device(device, gpu_id)
    processor = _load_moss_processor(model_path)
    audio_tokenizer = load_moss_tts_audio_tokenizer(
        _resolve_audio_tokenizer_model_path(processor, codec_model_path),
        device=device,
        dtype=dtype,
    )
    reference_encoder = _MossTTSReferenceEncoder(
        audio_tokenizer,
        n_vq=int(processor.model_config.n_vq),
    )
    set_moss_tts_preprocessing_context(
        processor=processor,
        reference_encoder=reference_encoder,
    )
    # Reference loading and text tokenization remain CPU-side, while the codec
    # forward follows the preprocessing stage's GPU placement. Keep multiple
    # workers so those CPU portions do not starve the AR engine.
    return SimpleScheduler(
        preprocess_moss_tts_payload,
        abort_callback=cleanup_prepared_moss_tts_request,
        max_concurrency=max_concurrency,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, Any] | None = None,
) -> Any:
    return MossTtsEngineBuilder().build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


create_tts_engine_executor = create_sglang_tts_engine_executor


class _MossTTSVocoder(BatchVocoderBase):
    def __init__(
        self,
        processor: Any,
        audio_tokenizer: MossTTSAudioTokenizer,
        device: str,
    ) -> None:
        self._processor = processor
        self._audio_tokenizer = audio_tokenizer
        self._device = device

    def prepare_item(self, payload: StagePayload) -> tuple[MossTTSState, torch.Tensor]:
        state = load_state(payload)
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
        decoded = []
        for segment in segments:
            decoded.extend(self._audio_tokenizer.decode_codes([segment]))
        if not decoded:
            raise RuntimeError("MOSS-TTS vocoder decoded no audio segments")
        waveforms = [
            torch.as_tensor(wav).detach().reshape(-1).to("cpu") for wav in decoded
        ]
        waveform = torch.cat(waveforms, dim=0)
        sample_rate = int(
            getattr(self._audio_tokenizer, "sample_rate", 0)
            or getattr(
                getattr(self._processor, "model_config", None), "sampling_rate", 0
            )
            or state.sample_rate
            or 24000
        )
        return waveform, sample_rate

    async def decode_batch(
        self, items: list[tuple[MossTTSState, torch.Tensor]]
    ) -> list[tuple[torch.Tensor, int]]:
        return [self._decode_audio(state, codes) for state, codes in items]

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
        payload = store_state(payload, state)
        payload.data.update(audio_payload)
        payload.data["sample_rate"] = state.sample_rate
        payload.data["modality"] = "audio"
        usage = build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "float32",
    codec_model_path: str | None = None,
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
    stream_stride: int = 8,
    stream_followup_stride: int = 8,
    stream_overlap_tokens: int = 8,
    stream_holdback_tokens: int = 1,
    initial_chunk_frames: int = 0,
) -> MossStreamingVocoderScheduler:
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    processor = _load_moss_processor(model_path)
    audio_tokenizer = load_moss_tts_audio_tokenizer(
        _resolve_audio_tokenizer_model_path(processor, codec_model_path),
        device=device,
        dtype=dtype,
    )

    vocoder = _MossTTSVocoder(processor, audio_tokenizer, device)
    return MossStreamingVocoderScheduler(
        vocoder,
        stream_stride=stream_stride,
        stream_followup_stride=stream_followup_stride,
        stream_overlap_tokens=stream_overlap_tokens,
        stream_holdback_tokens=stream_holdback_tokens,
        initial_chunk_frames=initial_chunk_frames,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
