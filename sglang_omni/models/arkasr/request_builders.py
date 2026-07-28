# SPDX-License-Identifier: Apache-2.0
"""StagePayload <-> SGLang request adapters for ARK-ASR-3B.

Mirrors the checkpoint's processor: mel features via WhisperFeatureExtractor,
prompt = ``<|user|>...<|begin_of_audio|>{N audio tokens}<|end_of_audio|>
Please transcribe this audio.<|assistant|>``, then the ``<|audio|>`` (id
151663) placeholders are scattered with encoder output by the model's
general_mm_embed_routine. ARK's LM is a dense Qwen2 (1-D RoPE, no MRoPE).
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

from .audio_lengths import arkasr_num_audio_tokens

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_AUDIO_TOKEN = "<|audio|>"
_BOA = "<|begin_of_audio|>"
_EOA = "<|end_of_audio|>"
_USER = "<|user|>"
_ASSISTANT = "<|assistant|>"
_DEFAULT_INSTRUCTION = "Please transcribe this audio."


@dataclass
class ArkASRRequestData(SGLangARRequestData):
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


def _load_audio(source: Any) -> np.ndarray:
    return load_audio(source, source_name="ARK-ASR", target_sample_rate=_SAMPLE_RATE)


def _decode_token_ids(
    tokenizer: Any, token_ids: list[int], skip_special_tokens: bool
) -> str:
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def _build_suppressed_token_ids(tokenizer: Any) -> list[int]:
    """All special / ``<...>`` added marker token ids except EOS.

    The checkpoint ships no ``bad_words_ids`` in its generation config, so plain
    ``skip_special_tokens=True`` decoding leaks the non-special added markers
    (e.g. ``<tool_call>``, ``<|audio|>``, ``<|fim_*|>``) verbatim into
    transcripts on adversarial / OOD audio. We defensively suppress every
    reserved marker (all ``all_special_ids`` plus ``<...>``-wrapped added
    tokens) except EOS at generation. Returned sorted for determinism.
    """
    eos = tokenizer.eos_token_id
    keep = {int(eos)} if isinstance(eos, int) else set(int(x) for x in (eos or []))
    bad: set[int] = set(int(i) for i in (tokenizer.all_special_ids or []))
    try:
        added = tokenizer.get_added_vocab()
    except Exception:
        added = {}
    for tok, tid in added.items():
        if isinstance(tok, str) and tok.startswith("<") and tok.endswith(">"):
            bad.add(int(tid))
    bad -= keep
    return sorted(bad)


def make_arkasr_scheduler_adapters(
    *,
    tokenizer: Any,
    max_new_tokens: int,
    feature_extractor: Any = None,
    merge_factor: int = 4,
    audio_token_id: int = 151663,
) -> tuple[Callable[[StagePayload], ArkASRRequestData], Callable[[Any], StagePayload]]:
    if feature_extractor is None:
        raise ValueError("ARK-ASR processor is missing a feature_extractor")

    eos_token_id = int(tokenizer.eos_token_id)
    vocab_size = int(tokenizer.vocab_size)

    # Defensively suppress every reserved marker (special / ``<...>`` added
    # token) except EOS. The checkpoint ships no bad_words_ids, so without this,
    # adversarial / OOD audio can leak markers like ``<tool_call>`` or
    # ``<|audio|>`` into transcripts (``skip_special_tokens`` only strips the few
    # "special" ones, not the non-special added tokens). We suppress at sampling
    # (hard-negative logit_bias) and strip on decode as belt-and-suspenders.
    _suppressed_ids = _build_suppressed_token_ids(tokenizer)

    def _build_prompt_ids(num_audio_tokens: int) -> list[int]:
        prompt = (
            f"{_USER}"
            f"{_BOA}{_AUDIO_TOKEN * num_audio_tokens}{_EOA}"
            f"{_DEFAULT_INSTRUCTION}"
            f"{_ASSISTANT}"
        )
        return list(tokenizer(prompt, add_special_tokens=False).input_ids)

    def request_builder(payload: StagePayload) -> ArkASRRequestData:
        params = payload.request.params or {}
        audio = _load_audio(_audio_source_from_payload(payload))
        audio_duration_s = float(len(audio) / _SAMPLE_RATE)
        fingerprint = audio_fingerprint(audio)

        # mel: pad to the clip's true length (short clips do not pay the full
        # 30s of FFT). ARK's WhisperEncoder is variable-length; conv2 stride-2
        # then merge_factor determines the audio-token count.
        extracted = feature_extractor(
            audio,
            sampling_rate=_SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
            padding="longest",
            truncation=True,
        )
        features = extracted.input_features  # [num_mel_bins, T]
        feature_attention_mask = getattr(extracted, "attention_mask", None)
        if feature_attention_mask is None:
            feature_attention_mask = torch.ones(
                (features.shape[0], features.shape[-1]), dtype=torch.long
            )
        num_mel_frames = int(feature_attention_mask.sum().item())
        num_audio_tokens = arkasr_num_audio_tokens(num_mel_frames, merge_factor)

        input_ids = _build_prompt_ids(num_audio_tokens)

        audio_item = MultimodalDataItem(
            modality=Modality.AUDIO,
            hash=audio_fingerprint_int(fingerprint),
            feature=features,
            model_specific_data={"feature_attention_mask": feature_attention_mask},
        )
        # scatter contract (same as qwen3_asr): replace <|audio|> placeholders
        # with the item's pad_value and record the span as inclusive offsets.
        audio_item.set_pad_value()
        audio_start = input_ids.index(audio_token_id)
        input_ids = [
            audio_item.pad_value if tok == audio_token_id else tok for tok in input_ids
        ]
        audio_item.offsets = [(audio_start, audio_start + num_audio_tokens - 1)]

        mm_inputs = MultimodalInputs(
            mm_items=[audio_item], num_image_tokens=num_audio_tokens
        )
        mm_inputs.audio_token_id = audio_token_id

        temperature = float(params.get("temperature") or 0.0)
        request_max_new_tokens = int(params.get("max_new_tokens") or max_new_tokens)
        sampling_params = SamplingParams(
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            stop_token_ids=[eos_token_id],
            logit_bias=(
                {str(tid): -100.0 for tid in _suppressed_ids}
                if _suppressed_ids
                else None
            ),
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

        return ArkASRRequestData(
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

    def result_adapter(data: ArkASRRequestData) -> StagePayload:
        payload = data.stage_payload
        output_ids = list(data.output_ids or [])
        # belt-and-suspenders: drop any suppressed marker tokens that slipped
        # through before decoding (logit_bias suppresses them at sampling, but a
        # non-greedy request could still surface one).
        if _suppressed_ids:
            _drop = set(_suppressed_ids)
            output_ids = [t for t in output_ids if t not in _drop]
        text = _decode_token_ids(
            tokenizer, output_ids, skip_special_tokens=True
        ).strip()
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


__all__ = ["ArkASRRequestData", "load_audio", "make_arkasr_scheduler_adapters"]
