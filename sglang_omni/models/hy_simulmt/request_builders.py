# SPDX-License-Identifier: Apache-2.0
"""Audio request and text response adapters for Hy-SimulMT."""

import time
from dataclasses import dataclass
from typing import Literal

import torch
from pydantic import TypeAdapter
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from transformers import PreTrainedTokenizerBase, WhisperFeatureExtractor

from sglang_omni.models.hy_simulmt.streaming import TranslationTurn
from sglang_omni.preprocessing.transcription import prepare_audio
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.token_text_streaming import (
    make_token_text_stream_output_builder,
)
from sglang_omni.scheduling.types import StreamOutputBuilder
from sglang_omni.utils.audio import audio_fingerprint, audio_fingerprint_int

SAMPLE_RATE = 16000
HOP_LENGTH = 160
MAX_AUDIO_SECONDS = 30
AUDIO_TOKEN = "<｜hy_place▁holder▁no▁5｜>"
AUDIO_START = "<｜hy_place▁holder▁no▁6｜>"
AUDIO_END = "<｜hy_place▁holder▁no▁8｜>"
PROMPT_PREFIX = (
    "<｜hy_begin▁of▁sentence｜><｜reasoning_mode｜>reasoning_effort:no_think"
)
USER_TOKEN = "<｜hy_User｜>"
ASSISTANT_PREFIX = "<｜hy_Assistant｜><think></think>"
TRANSLATION_INSTRUCTIONS = {
    "en": "把这段语音翻译成英文，要全部输出，不要分块。",
    "zh": "把这段语音翻译成中文，要全部输出，不要分块。",
}
SIMULTANEOUS_INSTRUCTIONS = {
    "en": "请完成下面的同声传译任务，将陆续收到的语音翻译成英文。请结合上下文边听边译，输出能够确定的译文，不必等待整句结束。",
    "zh": "请完成下面的同声传译任务，将陆续收到的语音翻译成中文。请结合上下文边听边译，输出能够确定的译文，不必等待整句结束。",
}
TURN_END = "<｜hy_EOT｜>"


def audio_token_count(mel_frames: int) -> int:
    # note (JingwenGu0829): Each 100-frame convolution block has its own right padding.
    encoder_frames = mel_frames // 100 * 13 + (mel_frames % 100 + 7) // 8
    return (encoder_frames + 1) // 2


@dataclass(kw_only=True)
class HySimulMTRequestData(SGLangARRequestData):
    output_ids: list[int] | None = None
    audio_duration_s: float = 0.0
    target_language: Literal["en", "zh"] = "en"
    engine_start_s: float = 0.0


@dataclass(kw_only=True)
class HySimulMTRequestBuilder:
    tokenizer: PreTrainedTokenizerBase
    feature_extractor: WhisperFeatureExtractor
    max_new_tokens: int
    context_length: int
    audio_token_id: int
    stop_token_ids: list[int]

    def __call__(self, payload: StagePayload) -> HySimulMTRequestData:
        parameters = payload.request.params
        stage_parameters = parameters.get("stage_params", {})
        if not isinstance(stage_parameters, dict):
            raise ValueError("stage_params must be an object")
        else:
            translation_parameters = stage_parameters.get("asr", {})
        if not isinstance(translation_parameters, dict):
            raise ValueError("stage_params.asr must be an object")
        else:
            parameters = {**parameters, **translation_parameters}
        if parameters.get("detect_language"):
            raise ValueError("Hy-SimulMT does not expose language-detection tokens")
        else:
            pass
        source_language = parameters.get("language")
        if source_language not in (None, "en", "zh"):
            raise ValueError("Hy-SimulMT supports Chinese (zh) and English (en)")
        else:
            pass
        default_target = "zh" if source_language == "en" else "en"
        if parameters.get("task") == "translate":
            default_target = "en"
        else:
            pass
        target_language = parameters.get("target_language", default_target)
        if target_language not in ("en", "zh"):
            raise ValueError("target_language must be en or zh")
        else:
            pass
        prepared = prepare_audio(
            payload,
            source_name="Hy-SimulMT",
            target_sample_rate=SAMPLE_RATE,
            max_duration_s=MAX_AUDIO_SECONDS,
        )
        history_value = parameters.get("hy_simulmt_history")
        if history_value is None:
            history: list[TranslationTurn] = []
            instruction = TRANSLATION_INSTRUCTIONS[target_language]
        else:
            history = TypeAdapter(list[TranslationTurn]).validate_python(history_value)
            instruction = SIMULTANEOUS_INSTRUCTIONS[target_language]
        prompt = parameters.get("prompt")
        if prompt:
            if not isinstance(prompt, str):
                raise ValueError("prompt must be a string")
            else:
                instruction += prompt
        else:
            pass
        input_ids = self.tokenizer.encode(PROMPT_PREFIX, add_special_tokens=False)
        audio_items: list[MultimodalDataItem] = []
        start_sample = 0
        turns = [*history, TranslationTurn(end_sample=len(prepared.waveform), text="")]
        for turn_index, turn in enumerate(turns):
            if not start_sample < turn.end_sample <= len(prepared.waveform):
                raise ValueError(
                    "Audio history must contain increasing boundaries before the new audio"
                )
            else:
                pass
            waveform = prepared.waveform[start_sample : turn.end_sample]
            extracted = self.feature_extractor(
                waveform,
                sampling_rate=SAMPLE_RATE,
                padding="longest",
                pad_to_multiple_of=HOP_LENGTH,
                truncation=False,
                return_attention_mask=True,
                return_tensors="pt",
            )
            mel_frames = int(extracted.attention_mask.sum())
            audio_tokens = audio_token_count(mel_frames)
            features = extracted.input_features[0, :, :mel_frames].contiguous()
            prompt_text = (
                USER_TOKEN + (instruction if turn_index == 0 else "") + AUDIO_START
            )
            input_ids.extend(
                self.tokenizer.encode(prompt_text, add_special_tokens=False)
            )
            audio_start = len(input_ids)
            audio_item = MultimodalDataItem(
                modality=Modality.AUDIO,
                hash=audio_fingerprint_int(audio_fingerprint(waveform)),
                feature=features,
            )
            audio_item.set_pad_value()
            audio_item.offsets = [(audio_start, audio_start + audio_tokens - 1)]
            input_ids.extend([audio_item.pad_value] * audio_tokens)
            input_ids.extend(
                self.tokenizer.encode(
                    AUDIO_END + ASSISTANT_PREFIX, add_special_tokens=False
                )
            )
            if turn_index < len(history):
                input_ids.extend(
                    self.tokenizer.encode(
                        turn.text + TURN_END, add_special_tokens=False
                    )
                )
            else:
                pass
            audio_items.append(audio_item)
            start_sample = turn.end_sample
        max_new_tokens = int(parameters.get("max_new_tokens") or self.max_new_tokens)
        if len(input_ids) + max_new_tokens > self.context_length:
            raise ValueError("Hy-SimulMT input and output exceed the context length")
        else:
            pass
        multimodal_inputs = MultimodalInputs(
            mm_items=audio_items,
            num_image_tokens=sum(
                end - start + 1 for item in audio_items for start, end in item.offsets
            ),
        )
        multimodal_inputs.audio_token_id = self.audio_token_id
        temperature = float(parameters.get("temperature") or 0.0)
        sampling = SamplingParams(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            stop_token_ids=self.stop_token_ids,
        )
        sampling.normalize(tokenizer=None)
        request = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=input_ids,
            sampling_params=sampling,
            vocab_size=len(self.tokenizer),
            extra_key=prepared.fingerprint,
        )
        request.multimodal_inputs = multimodal_inputs
        return HySimulMTRequestData(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            req=request,
            stage_payload=payload,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            return_logprob=bool(parameters.get("return_logprob", False)),
            audio_duration_s=prepared.duration_s,
            target_language=target_language,
            engine_start_s=time.perf_counter(),
        )


@dataclass(kw_only=True)
class HySimulMTResultAdapter:
    tokenizer: PreTrainedTokenizerBase

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def __call__(self, request: HySimulMTRequestData) -> StagePayload:
        payload = request.stage_payload
        assert payload is not None
        elapsed_seconds = time.perf_counter() - request.engine_start_s
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                "text": self.decode(request.output_ids or []).strip(),
                "token_ids": list(request.output_ids or []),
                "output_token_logprobs": (
                    request.output_token_logprobs if request.return_logprob else None
                ),
                "finish_reason": request.finish_reason,
                "language": request.target_language,
                "duration_s": request.audio_duration_s,
                "asr_latency_s": elapsed_seconds,
                "usage": {
                    "engine_time_s": elapsed_seconds,
                    "prompt_tokens": len(request.input_ids),
                    "completion_tokens": len(request.output_ids or []),
                },
                "modality": "text",
            },
        )

    def stream_builder(self) -> StreamOutputBuilder[HySimulMTRequestData]:
        return make_token_text_stream_output_builder(
            decode_fn=self.decode,
            build_message_data=lambda text: {
                "text": text,
                "modality": "text",
                "stage_name": "asr",
            },
            build_message_metadata=lambda token_id: {
                "modality": "text",
                "token_id": token_id,
            },
            pending_ids_attr="hy_simulmt_pending_token_ids",
            last_emit_attr="hy_simulmt_last_emit_seconds",
            eos_token_id=None,
            allow_terminal_flush=True,
        )
