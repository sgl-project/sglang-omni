# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
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
from sglang_omni.utils.audio import audio_fingerprint, audio_fingerprint_int
from sglang_omni.utils.audio import load_audio as _shared_load_audio

from .configuration_fun_asr import AUDIO_PLACEHOLDER_TOKEN as _AUDIO_PAD
from .tool_funcs.audio_lengths import fun_asr_low_frame_rate_length

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_MAX_AUDIO_DURATION_S = 30.0
_MAX_GENERATION_TOKENS_AT_MAX_DURATION = 200
_MIN_GENERATION_TOKENS = 16


@dataclass
class FunASRRequestData(SGLangARRequestData):
    prompt_token_ids: list[int] | None = None
    output_ids: list[int] | None = None
    audio_duration_s: float = 0.0
    language: str | None = None
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
    return _shared_load_audio(
        source,
        source_name="Fun-ASR",
        target_sample_rate=_SAMPLE_RATE,
    )


def _default_token_budget(audio_duration_s: float, max_new_tokens: int) -> int:
    proportional = math.ceil(
        audio_duration_s
        / _MAX_AUDIO_DURATION_S
        * _MAX_GENERATION_TOKENS_AT_MAX_DURATION
    )
    return min(max_new_tokens, max(_MIN_GENERATION_TOKENS, proportional))


def _request_token_budget(
    params: dict[str, Any], audio_duration_s: float, max_new_tokens: int
) -> int:
    explicit = params.get("max_new_tokens")
    if explicit is None:
        return _default_token_budget(audio_duration_s, max_new_tokens)

    try:
        requested = int(explicit)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_new_tokens must be an integer") from exc
    if requested < 1 or requested > max_new_tokens:
        raise ValueError(
            f"max_new_tokens must be between 1 and {max_new_tokens}, got {requested}"
        )
    return requested


def _decode_token_ids(
    tokenizer: Any, token_ids: list[int], *, skip_special_tokens: bool
) -> str:
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def _resolve_language(lang_raw: str | None) -> str | None:

    if lang_raw is None:
        return None
    lang = lang_raw.strip().lower()
    if lang in ("", "auto", "null", "none"):
        return None
    if lang in ("zh", "cn", "chinese", "中文"):
        return None
    if lang in ("en", "english", "英文"):
        return "英文"
    return lang_raw.strip()


def _build_prompt_text(language: str | None, itn: bool, hotwords: list[str]) -> str:

    prompt = ""
    if hotwords:
        joined = ", ".join(hotwords)
        prompt += (
            "请结合上下文信息，更加准确地完成语音转写任务。"
            "如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n"
        )
        prompt += f"热词列表：[{joined}]\n"
    if language is None:
        prompt += "语音转写"
    else:
        prompt += f"语音转写成{language}"
    if not itn:
        prompt += "，不进行文本规整"
    return prompt + "："


def _prompt_template(prompt_text: str, num_audio_tokens: int) -> str:

    return (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{prompt_text}{_AUDIO_PAD * num_audio_tokens}"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def fun_asr_prompt_overhead_tokens(
    tokenizer: Any,
    *,
    language: str | None = None,
    itn: bool = True,
    hotwords: tuple[str, ...] = (),
) -> int:
    """Token count of the non-audio prompt preamble (wrappers + prompt_text).

    Audio placeholders are sized separately via ``encoder_token_count``; the
    context-length budget needs only this overhead plus the per-request
    prompt tokens. Tokenizing without audio pads is exact because
    ``_AUDIO_PAD`` is a special token with atomic boundaries.
    """
    prompt_text = _build_prompt_text(language, itn, list(hotwords))
    return len(
        tokenizer(_prompt_template(prompt_text, 0), add_special_tokens=False).input_ids
    )


def make_fun_asr_scheduler_adapters(
    *,
    tokenizer: Any,
    max_new_tokens: int,
    feature_extractor: Any = None,
    context_length: int | None = None,
) -> tuple[
    Callable[[StagePayload], FunASRRequestData],
    Callable[[FunASRRequestData], StagePayload],
]:
    if feature_extractor is None:
        raise ValueError("Fun-ASR processor is missing a feature_extractor")
    max_new_tokens = int(max_new_tokens)
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")

    audio_pad_token_id = int(tokenizer.convert_tokens_to_ids(_AUDIO_PAD))
    eos_token_id = int(tokenizer.eos_token_id)
    vocab_size = int(tokenizer.vocab_size)

    def _build_prompt_ids(num_audio_tokens: int, prompt_text: str) -> list[int]:
        return tokenizer(
            _prompt_template(prompt_text, num_audio_tokens),
            add_special_tokens=False,
        ).input_ids

    def request_builder(payload: StagePayload) -> FunASRRequestData:
        params = payload.request.params or {}
        audio = _load_audio(_audio_source_from_payload(payload))
        audio_duration_s = float(len(audio) / _SAMPLE_RATE)
        if audio_duration_s > _MAX_AUDIO_DURATION_S:
            raise ValueError(
                "Fun-ASR accepts audio up to 30.0 seconds because its official "
                "VAD segment limit is 30 seconds; split longer audio before inference."
            )
        fingerprint = audio_fingerprint(audio)

        extracted = feature_extractor(
            audio,
            sampling_rate=_SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
            padding="longest",
        )
        features = extracted["input_features"]  # [1, 560, T_lfr]
        feature_attention_mask = extracted.get("attention_mask")
        if feature_attention_mask is None:
            feature_attention_mask = torch.ones(
                (features.shape[0], features.shape[-1]), dtype=torch.long
            )
        num_lfr_frames = int(feature_attention_mask.sum().item())
        num_audio_tokens = int(fun_asr_low_frame_rate_length(num_lfr_frames))
        logger.debug(
            f"[fun-asr] lfr_frames={num_lfr_frames} "
            f"num_audio_tokens={num_audio_tokens} feat_shape={tuple(features.shape)}"
        )

        lang_raw = params.get("language")
        language = _resolve_language(lang_raw)
        itn = bool(params.get("itn", True))
        hotwords_raw = params.get("hotwords") or []
        # A bare string would be split into characters by list(...); wrap it as
        # a single hotword so each user-supplied entry stays intact.
        if isinstance(hotwords_raw, str):
            hotwords = [hotwords_raw]
        else:
            hotwords = list(hotwords_raw)
        prompt_text = _build_prompt_text(language, itn, hotwords)
        input_ids = _build_prompt_ids(num_audio_tokens, prompt_text)

        audio_item = MultimodalDataItem(
            modality=Modality.AUDIO,
            hash=audio_fingerprint_int(fingerprint),
            feature=features,
            model_specific_data={
                "feature_attention_mask": feature_attention_mask,
            },
        )

        audio_item.set_pad_value()
        if audio_pad_token_id not in input_ids:
            raise RuntimeError(
                f"Fun-ASR prompt missing audio placeholder {_AUDIO_PAD!r} "
                f"(id {audio_pad_token_id}); prompt_text={prompt_text!r}"
            )
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

        seq_len = len(input_ids)
        positions = torch.arange(seq_len, dtype=torch.long)
        mm_inputs.mrope_positions = positions.unsqueeze(0).expand(3, -1).clone()
        mm_inputs.mrope_position_delta = torch.tensor([0], dtype=torch.long)

        temperature = float(params.get("temperature") or 0.0)
        request_max_new_tokens = _request_token_budget(
            params, audio_duration_s, max_new_tokens
        )
        if (
            context_length is not None
            and len(input_ids) + request_max_new_tokens > context_length
        ):
            raise ValueError(
                f"Fun-ASR request is longer than the model's context length "
                f"({len(input_ids)} prompt/audio tokens + {request_max_new_tokens} "
                f"max_new_tokens > {context_length}); reduce hotwords or split the audio"
            )
        logger.debug(
            f"[fun-asr] sampling temp={temperature} "
            f"max_new_tokens={request_max_new_tokens} params={dict(params)}"
        )
        sampling_params = SamplingParams(
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            stop_token_ids=[eos_token_id],
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

        return FunASRRequestData(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            req=req,
            prompt_token_ids=input_ids,
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            audio_duration_s=audio_duration_s,
            language=lang_raw,
            engine_start_s=time.perf_counter(),
            stage_payload=payload,
        )

    def result_adapter(data: FunASRRequestData) -> StagePayload:
        payload = data.stage_payload
        output_ids = list(data.output_ids or [])

        text = _decode_token_ids(tokenizer, output_ids, skip_special_tokens=True)
        engine_time_s = (
            time.perf_counter() - data.engine_start_s if data.engine_start_s else 0.0
        )
        logger.debug(
            f"[fun-asr] n_out={len(output_ids)} ids={output_ids[:40]} text={text!r}"
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
    "FunASRRequestData",
    "make_fun_asr_scheduler_adapters",
]
