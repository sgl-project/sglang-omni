# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the MOSS-TTS Delay pipeline."""

from __future__ import annotations

import logging
from typing import Any

import torch

from sglang_omni.models.moss_tts.audio_tokenizer import (
    DEFAULT_MOSS_AUDIO_TOKENIZER,
    load_moss_audio_tokenizer,
)
from sglang_omni.models.moss_tts.codec import split_moss_audio_segments
from sglang_omni.models.moss_tts.hf_loading import (
    load_moss_processor_class,
    moss_transformers_processor_compat,
    resolve_moss_checkpoint,
)
from sglang_omni.models.moss_tts.payload_types import (
    MossTTSState,
    moss_tts_special_token_defaults,
)
from sglang_omni.models.moss_tts.request_builders import (
    cleanup_prepared_moss_tts_request,
    preprocess_moss_tts_payload,
    set_moss_tts_preprocessing_context,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.pipeline_state import load_state as _load_pipeline_state
from sglang_omni.scheduling.pipeline_state import store_state as _store_pipeline_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

_MOSS_TTS_INSTALL_HINT = (
    "MOSS-TTS processor support requires the upstream custom Transformers code. "
    "Launch the MOSS-TTS checkpoint with trust_remote_code=True. The codec "
    "model implementation is local; make sure the codec config and safetensors "
    "artifacts are available."
)


def load_state(payload: StagePayload) -> MossTTSState:
    return _load_pipeline_state(payload, MossTTSState)


def store_state(payload: StagePayload, state: MossTTSState) -> StagePayload:
    return _store_pipeline_state(payload, state)


def _torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    return getattr(torch, dtype) if isinstance(dtype, str) else dtype


def _normalize_moss_model_config(model_config: Any) -> None:
    if model_config is None:
        return
    audio_vocab_size = int(getattr(model_config, "audio_vocab_size", 1024) or 1024)
    for attr, default in moss_tts_special_token_defaults(audio_vocab_size):
        if getattr(model_config, attr, None) is None:
            setattr(model_config, attr, default)


def _normalize_moss_processor_config(processor: Any) -> None:
    model_config = getattr(processor, "model_config", None)
    _normalize_moss_model_config(model_config)


def _load_moss_model_config(model_path: str) -> Any:
    from transformers import AutoConfig

    checkpoint_dir = resolve_moss_checkpoint(model_path)
    with moss_transformers_processor_compat():
        model_config = AutoConfig.from_pretrained(
            checkpoint_dir,
            trust_remote_code=True,
        )
    _normalize_moss_model_config(model_config)
    return model_config


def _load_moss_processor(
    model_path: str,
    *,
    device: str | None = None,
    dtype: str | torch.dtype | None = None,
) -> Any:
    del device, dtype
    checkpoint_dir = resolve_moss_checkpoint(model_path)
    logger.info("Loading MOSS-TTS processor from %s without codec", checkpoint_dir)
    try:
        from transformers import AutoConfig, AutoTokenizer

        with moss_transformers_processor_compat():
            processor_cls = load_moss_processor_class(checkpoint_dir)
            model_config = AutoConfig.from_pretrained(
                checkpoint_dir,
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_dir,
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


def _resolve_codec_device(device: str) -> str:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; loading MOSS audio tokenizer on CPU")
        return "cpu"
    return device


def create_preprocessing_executor(
    model_path: str,
    *,
    max_concurrency: int = 8,
    codec_model_path: str = DEFAULT_MOSS_AUDIO_TOKENIZER,
    device: str = "cuda:0",
    dtype: str = "float32",
) -> SimpleScheduler:
    processor = _load_moss_processor(model_path)
    device = _resolve_codec_device(device)
    audio_tokenizer = load_moss_audio_tokenizer(
        codec_model_path,
        device=device,
        dtype=dtype,
    )
    set_moss_tts_preprocessing_context(
        processor=processor,
        audio_tokenizer=audio_tokenizer,
    )
    # Preprocessing tokenizes text and encodes reference audio through the MOSS
    # audio tokenizer. Run several in parallel so the AR OmniScheduler receives
    # a steady, batchable request stream.
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
    from sglang_omni.models.moss_tts.engine_builder import MossTtsEngineBuilder

    return MossTtsEngineBuilder().build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


def create_tts_engine_executor(*args, **kwargs) -> Any:
    return create_sglang_tts_engine_executor(*args, **kwargs)


class _MossTTSVocoder(BatchVocoderBase):
    def __init__(self, model_config: Any, audio_tokenizer: Any, device: str) -> None:
        self._model_config = model_config
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
        audio_pad_code = int(getattr(self._model_config, "audio_pad_code", 1024))
        segments = split_moss_audio_segments(
            delayed_codes,
            audio_pad_code=audio_pad_code,
            assistant_start_length=int(state.assistant_start_length),
        )
        decoded = self._audio_tokenizer.decode_codes(segments)
        if not decoded:
            raise RuntimeError("MOSS-TTS vocoder decoded no audio segments")
        waveforms = [
            torch.as_tensor(wav).detach().reshape(-1).to("cpu") for wav in decoded
        ]
        waveform = torch.cat(waveforms, dim=0)
        sample_rate = int(
            getattr(self._audio_tokenizer, "sample_rate", 0)
            or getattr(self._model_config, "sampling_rate", 0)
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
    codec_model_path: str = DEFAULT_MOSS_AUDIO_TOKENIZER,
    dtype: str = "float32",
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
) -> SimpleScheduler:
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    device = _resolve_codec_device(device)
    model_config = _load_moss_model_config(model_path)
    audio_tokenizer = load_moss_audio_tokenizer(
        codec_model_path,
        device=device,
        dtype=dtype,
    )

    return _MossTTSVocoder(model_config, audio_tokenizer, device).build_scheduler(
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
