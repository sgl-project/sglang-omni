# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2.5-ASR StagePayload adapters."""

from __future__ import annotations

import hashlib
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

from sglang_omni.models.mimo_v25_asr.prompt import (
    EMPTY,
    build_asr_sft_prompt,
    strip_asr_markers,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData


def audio_source_from_payload(payload: StagePayload) -> Any:
    inputs = payload.request.inputs
    if isinstance(inputs, dict):
        for key in ("audio_bytes", "bytes", "file"):
            if inputs.get(key) is not None:
                return inputs[key]
        for key in ("audio_path", "path", "url"):
            if inputs.get(key) is not None:
                return inputs[key]
    if inputs is None:
        raise ValueError("MiMo-V2.5-ASR request is missing audio input")
    return inputs


def validate_text_only_request(payload: StagePayload) -> None:
    modalities = payload.request.metadata.get("output_modalities", ["text"])
    if isinstance(modalities, str):
        modalities = [modalities]
    normalized = {str(value).strip().lower() for value in modalities}
    if "audio" in normalized or normalized - {"text"}:
        raise ValueError(
            "MiMo-V2.5-ASR supports text transcription only; audio output is "
            "not implemented"
        )


@dataclass
class MiMoV25ASRRequestData(SGLangARRequestData):
    output_ids: list[int] | None = None
    prompt_token_ids: list[int] | None = None
    audio_duration_s: float = 0.0
    language: str = "auto"
    engine_start_s: float = 0.0


def make_mimo_v25_asr_scheduler_adapters(
    *,
    tokenizer: Any,
    max_new_tokens: int,
) -> tuple[
    Callable[[StagePayload], MiMoV25ASRRequestData],
    Callable[[MiMoV25ASRRequestData], StagePayload],
]:
    eos_id = int(tokenizer.eos_token_id)
    im_end_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    empty_id = int(tokenizer.convert_tokens_to_ids(EMPTY))
    if hasattr(tokenizer, "vocab_size"):
        vocab_size = int(tokenizer.vocab_size)
    else:
        vocab_size = int(len(tokenizer))

    def request_builder(payload: StagePayload) -> MiMoV25ASRRequestData:
        validate_text_only_request(payload)
        if not isinstance(payload.data, dict) or "audio_codes" not in payload.data:
            raise ValueError("MiMo-V2.5-ASR requires preprocessed audio codes")
        codes = torch.as_tensor(payload.data["audio_codes"], dtype=torch.long)
        params = payload.request.params or {}
        language = str(params.get("language") or "auto")
        prompt_override = params.get("prompt")
        prompt = build_asr_sft_prompt(
            tokenizer,
            codes,
            language=language,
            instruction=str(prompt_override) if prompt_override else None,
        )
        input_ids = list(prompt.input_ids)
        fingerprint = str(
            payload.data.get(
                "audio_fingerprint",
                hashlib.sha256(codes.numpy().tobytes()).hexdigest(),
            )
        )
        item = MultimodalDataItem(
            modality=Modality.AUDIO,
            hash=int(fingerprint[:16], 16),
            feature=prompt.audio_codes,
        )
        item.set_pad_value()
        for index in range(prompt.audio_start, prompt.audio_end):
            input_ids[index] = item.pad_value
        item.offsets = [(prompt.audio_start, prompt.audio_end - 1)]
        mm_inputs = MultimodalInputs(
            mm_items=[item],
            num_image_tokens=prompt.audio_end - prompt.audio_start,
        )
        mm_inputs.audio_token_id = empty_id

        temperature = float(params.get("temperature") or 0.0)
        request_max_new_tokens = int(params.get("max_new_tokens") or max_new_tokens)
        sampling = SamplingParams(
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            stop_token_ids=[eos_id, im_end_id],
            # `<|empty|>` selects official output-audio generation. Suppress it
            # so ASR cannot enter a delayed-RVQ path.
            logit_bias={str(empty_id): -1.0e9},
        )
        sampling.normalize(tokenizer=None)
        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=input_ids,
            sampling_params=sampling,
            vocab_size=vocab_size,
            extra_key=fingerprint,
        )
        req.multimodal_inputs = mm_inputs
        req._codec_suppress_tokens = None
        return MiMoV25ASRRequestData(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            req=req,
            prompt_token_ids=input_ids,
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            audio_duration_s=float(payload.data.get("audio_duration_s", 0.0)),
            language=language,
            engine_start_s=time.perf_counter(),
            stage_payload=payload,
        )

    def result_adapter(data: MiMoV25ASRRequestData) -> StagePayload:
        payload = data.stage_payload
        output_ids = list(data.output_ids or [])
        text = tokenizer.decode(
            output_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        latency = (
            time.perf_counter() - data.engine_start_s if data.engine_start_s else 0.0
        )
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                "text": strip_asr_markers(text),
                "language": data.language,
                "duration_s": data.audio_duration_s,
                "asr_latency_s": latency,
                "usage": {"engine_time_s": latency},
                "modality": "text",
            },
        )

    return request_builder, result_adapter


__all__ = [
    "MiMoV25ASRRequestData",
    "audio_source_from_payload",
    "make_mimo_v25_asr_scheduler_adapters",
    "validate_text_only_request",
]
