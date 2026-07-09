# SPDX-License-Identifier: Apache-2.0
"""StagePayload <-> SGLang request adapters for Higgs-Audio-v3-STT.

Like Qwen3-ASR, higgs is a causal LM ingesting audio as multimodal
embeddings — but the audio front end differs: the waveform is split into
fixed 4 s chunks, each chunk gets its own mel spectrogram (padded to the
longest chunk in the clip) and contributes ``(mel-1)//2//2 -> conv``
audio tokens (12.5/s). All chunks' embeddings form one contiguous span
between ``<|audio_bos|>`` and ``<|audio_eos|>`` in the ChatML prompt.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
)
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData
from sglang_omni.utils.audio import audio_fingerprint, audio_fingerprint_int, load_audio

from .configuration_higgs_audio_asr import higgs_audio_token_lengths

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000

_AUDIO_BOS = "<|audio_bos|>"
_AUDIO_PAD = "<|AUDIO|>"
_AUDIO_EOS = "<|audio_eos|>"

# The reference transcribe.py prompt (lowercase, no punctuation output).
DEFAULT_TRANSCRIBE_PROMPT = (
    "Transcribe the speech. Output only the spoken words in lowercase "
    "with no punctuation."
)
# Empty think block: the transcript starts immediately (enable_thinking=False
# in the reference pipeline).
_THINK_SUFFIX = "<think>\n\n</think>\n\n"


@dataclass
class HiggsAudioASRRequestData(SGLangARRequestData):
    prompt_token_ids: list[int] | None = None
    output_ids: list[int] | None = None
    audio_duration_s: float = 0.0
    language: str = "en"
    engine_start_s: float = 0.0


def _audio_source_from_payload(payload: StagePayload) -> Any:
    inputs = payload.request.inputs
    if isinstance(inputs, dict):
        for key in ("audio_bytes", "bytes", "file"):
            value = inputs.get(key)
            if value is not None:
                return value
        for key in ("audio_path", "path", "url"):
            value = inputs.get(key)
            if value is not None:
                return value
    return inputs


def make_higgs_audio_asr_scheduler_adapters(
    *,
    tokenizer: Any,
    feature_extractor: Any,
    max_new_tokens: int,
    chunk_size_seconds: float = 4.0,
) -> tuple[
    Callable[[StagePayload], HiggsAudioASRRequestData], Callable[[Any], StagePayload]
]:
    if feature_extractor is None:
        raise ValueError("Higgs-Audio-ASR is missing a feature_extractor")

    audio_pad_token_id = int(tokenizer.convert_tokens_to_ids(_AUDIO_PAD))
    stop_token_ids = [
        int(tokenizer.convert_tokens_to_ids("<|im_end|>")),
        int(tokenizer.convert_tokens_to_ids("<|endoftext|>")),
    ]
    vocab_size = int(tokenizer.vocab_size)
    chunk_samples = int(chunk_size_seconds * _SAMPLE_RATE)

    def _encode(text: str) -> list[int]:
        return list(tokenizer.encode(text, add_special_tokens=False))

    def _build_prompt_ids(num_audio_tokens: int, user_prompt: str) -> list[int]:
        prompt = (
            f"<|im_start|>user\n"
            f"{user_prompt}"
            f"{_AUDIO_BOS}{_AUDIO_PAD * num_audio_tokens}{_AUDIO_EOS}"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"{_THINK_SUFFIX}"
        )
        return _encode(prompt)

    def request_builder(payload: StagePayload) -> HiggsAudioASRRequestData:
        params = payload.request.params or {}
        audio = load_audio(
            _audio_source_from_payload(payload),
            source_name="Higgs-Audio-ASR",
            target_sample_rate=_SAMPLE_RATE,
        )
        audio_duration_s = float(len(audio) / _SAMPLE_RATE)
        fingerprint = audio_fingerprint(audio)

        # Fixed 4 s chunking (the reference pipeline's non-VAD path); each
        # chunk gets its own mel, padded to the longest chunk in the clip.
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        chunks = [
            audio[i : i + chunk_samples]
            for i in range(0, max(len(audio), 1), chunk_samples)
        ]
        extracted = feature_extractor(
            chunks,
            sampling_rate=_SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
            padding="longest",
            truncation=False,
        )
        features = extracted.input_features  # (num_chunks, 128, T_longest)
        feature_attention_mask = getattr(extracted, "attention_mask", None)
        if feature_attention_mask is None:
            feature_attention_mask = torch.ones(
                (features.shape[0], features.shape[-1]), dtype=torch.long
            )
        mel_lens = feature_attention_mask.sum(dim=-1)
        per_chunk_tokens = higgs_audio_token_lengths(mel_lens)
        num_audio_tokens = int(per_chunk_tokens.sum().item())
        logger.debug(
            f"[higgs-asr] chunks={len(chunks)} mel_lens={mel_lens.tolist()} "
            f"audio_tokens={per_chunk_tokens.tolist()} total={num_audio_tokens}"
        )

        user_prompt = str(params.get("prompt") or DEFAULT_TRANSCRIBE_PROMPT)
        input_ids = _build_prompt_ids(num_audio_tokens, user_prompt)

        audio_item = MultimodalDataItem(
            modality=Modality.AUDIO,
            hash=audio_fingerprint_int(fingerprint),
            feature=features,
            model_specific_data={
                "feature_attention_mask": feature_attention_mask,
            },
        )
        # general_mm_embed_routine locates audio positions by matching the
        # item's pad_value against input_ids; the omni scheduler does not
        # run pad_input_ids, so scatter the pad_value ourselves and record
        # the (inclusive) placeholder span.
        audio_item.set_pad_value()
        audio_start = input_ids.index(audio_pad_token_id)
        input_ids = [
            audio_item.pad_value if tok == audio_pad_token_id else tok
            for tok in input_ids
        ]
        audio_item.offsets = [(audio_start, audio_start + num_audio_tokens - 1)]

        mm_inputs = MultimodalInputs(
            mm_items=[audio_item],
            num_image_tokens=num_audio_tokens,
        )
        mm_inputs.audio_token_id = audio_pad_token_id

        # Reference eval decodes greedily; temperature 0 is fine for higgs.
        temperature = float(params.get("temperature") or 0.0)
        request_max_new_tokens = int(params.get("max_new_tokens") or max_new_tokens)
        sampling_params = SamplingParams(
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            stop_token_ids=stop_token_ids,
        )
        sampling_params.normalize(tokenizer=None)

        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=input_ids,
            sampling_params=sampling_params,
            vocab_size=vocab_size,
            extra_key=fingerprint,
        )
        req.multimodal_inputs = mm_inputs
        req._codec_suppress_tokens = None

        return HiggsAudioASRRequestData(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            req=req,
            prompt_token_ids=input_ids,
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            audio_duration_s=audio_duration_s,
            language=str(params.get("language") or "en"),
            engine_start_s=time.perf_counter(),
            stage_payload=payload,
        )

    def result_adapter(data: HiggsAudioASRRequestData) -> StagePayload:
        payload = data.stage_payload
        output_ids = list(data.output_ids or [])
        text = tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        # The prompt forces an empty think block, but strip a stray one if
        # the model opens its own anyway.
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        text = text.strip()

        engine_time_s = (
            time.perf_counter() - data.engine_start_s if data.engine_start_s else 0.0
        )
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                "text": text,
                "language": data.language,
                "duration_s": data.audio_duration_s,
                "asr_latency_s": engine_time_s,
                "usage": {"engine_time_s": engine_time_s},
                "modality": "text",
            },
        )

    return request_builder, result_adapter


__all__ = [
    "DEFAULT_TRANSCRIBE_PROMPT",
    "HiggsAudioASRRequestData",
    "make_higgs_audio_asr_scheduler_adapters",
]
