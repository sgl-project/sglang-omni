# SPDX-License-Identifier: Apache-2.0
"""StagePayload adapters for Kimi-Audio text generation."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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

from .processor import KimiAudioProcessor


@dataclass
class KimiAudioRequestData(SGLangARRequestData):
    output_ids: list[int] | None = None
    engine_start_s: float = 0.0


def _request_inputs(payload: StagePayload) -> tuple[list[dict[str, Any]], list[Any]]:
    inputs = payload.request.inputs
    if isinstance(inputs, dict):
        messages = inputs.get("messages")
        audios = list(inputs.get("audios") or [])
    else:
        messages = inputs
        audios = []
    if not isinstance(messages, list) or not messages:
        raise ValueError("Kimi-Audio requires a non-empty messages list")
    return messages, audios


def make_kimi_audio_scheduler_adapters(
    *,
    processor: KimiAudioProcessor,
    max_new_tokens: int,
    context_length: int,
) -> tuple[
    Callable[[StagePayload], KimiAudioRequestData], Callable[[Any], StagePayload]
]:
    tokenizer = processor.tokenizer
    text_eos = processor.special.text_eos
    text_vocab_size = int(processor.model_config.kimia_text_output_vocab)

    class _SamplingTokenizer:
        @staticmethod
        def encode(text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return processor._encode_text(text)

    def request_builder(payload: StagePayload) -> KimiAudioRequestData:
        output_modalities = payload.request.metadata.get("output_modalities", ["text"])
        if any(modality != "text" for modality in output_modalities):
            raise ValueError(
                "Invalid Kimi-Audio request: text output is currently the only "
                "supported modality"
            )
        try:
            messages, audios = _request_inputs(payload)
            prompt = processor.build_prompt(messages, audios)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Kimi-Audio request: {exc}") from exc
        params = payload.request.params or {}

        def param_or_default(name: str, default: Any) -> Any:
            value = params.get(name)
            return default if value is None else value

        request_max_new_tokens = int(param_or_default("max_new_tokens", max_new_tokens))
        if len(prompt.audio_ids) + request_max_new_tokens > context_length:
            raise ValueError(
                "Invalid Kimi-Audio request: request exceeds the context length "
                f"({len(prompt.audio_ids)} prompt tokens + {request_max_new_tokens} "
                f"generated tokens > {context_length})"
            )

        features = (
            torch.cat(prompt.continuous_features, dim=0)
            if prompt.continuous_features
            else torch.empty((0, 5120), dtype=torch.bfloat16)
        )
        item = MultimodalDataItem(
            modality=Modality.AUDIO,
            feature=features,
            offsets=[(0, len(prompt.audio_ids) - 1)],
            model_specific_data={
                "text_input_ids": torch.tensor(prompt.text_ids, dtype=torch.long),
                "continuous_mask": torch.tensor(
                    prompt.continuous_mask, dtype=torch.bool
                ),
            },
        )
        item.set_pad_value()
        mm_inputs = MultimodalInputs(mm_items=[item], num_image_tokens=0)

        sampling = SamplingParams(
            max_new_tokens=request_max_new_tokens,
            temperature=float(param_or_default("temperature", 0.0)),
            top_p=float(param_or_default("top_p", 1.0)),
            top_k=int(param_or_default("top_k", -1)),
            min_p=float(param_or_default("min_p", 0.0)),
            repetition_penalty=float(param_or_default("repetition_penalty", 1.0)),
            stop=params.get("stop") or [],
            stop_token_ids={
                text_eos,
                *(int(token) for token in (params.get("stop_token_ids") or [])),
            },
            sampling_seed=params.get("seed"),
        )
        try:
            sampling.normalize(_SamplingTokenizer())
            sampling.verify(text_vocab_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Kimi-Audio request: {exc}") from exc
        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=prompt.audio_ids,
            sampling_params=sampling,
            vocab_size=text_vocab_size,
            extra_key=str(item.hash),
        )
        req.tokenizer = tokenizer
        req.multimodal_inputs = mm_inputs
        return KimiAudioRequestData(
            input_ids=torch.tensor(prompt.audio_ids, dtype=torch.long),
            req=req,
            max_new_tokens=request_max_new_tokens,
            temperature=sampling.temperature,
            engine_start_s=time.perf_counter(),
            stage_payload=payload,
        )

    def result_adapter(data: KimiAudioRequestData) -> StagePayload:
        output_ids = list(data.output_ids or [])
        text = tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        latency = time.perf_counter() - data.engine_start_s
        prompt_tokens = int(data.input_ids.numel())
        completion_tokens = len(output_ids)
        payload = data.stage_payload
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                "text": text,
                "modality": "text",
                "token_ids": output_ids,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "finish_reason": getattr(data, "finish_reason", None),
                "weight_version": getattr(data, "weight_version", None),
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "engine_time_s": latency,
                },
            },
        )

    return request_builder, result_adapter


__all__ = ["KimiAudioRequestData", "make_kimi_audio_scheduler_adapters"]
